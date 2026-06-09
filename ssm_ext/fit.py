# fit.py

import numpy as np
import numpy.random as npr
import ssm
from sklearn.cluster import KMeans
from scipy.special import logsumexp

from .init import fit_kmeans

import autograd.numpy as _anp
from autograd.scipy.special import logsumexp as _ag_logsumexp
from ssm.transitions import RecurrentOnlyTransitions as _RecOnly


# ---------------------------------------------------------------
# Stickiness
# ---------------------------------------------------------------

# Fixed self-transition stickiness for fit_rSLDS_restricted_em (Fox 2011
# sticky prior applied to Linderman 2017 recurrent_only transitions).
# Empirical safe band across baseline / long-dwell / short-dwell synthetic
# price DGPs is ~[5, 15]; midpoint 10 is comfortably interior so the chain
# neither collapses (kappa < 5) nor freezes (kappa >> 20). Structural
# hyperparameter of the method, not a user-tunable knob.
_STICKY_KAPPA = 10.0


class StickyRecurrentOnly(_RecOnly):
    """Recurrent-only rSLDS transitions with fixed self-transition stickiness.

    Linderman et al. ("Bayesian Learning and Inference in Recurrent Switching
    Linear Dynamical Systems", AISTATS 2017) define `recurrent_only`:
    log p(z_t=j | x) = R_j.x + r_j, with NO dependence on z_{t-1}, so the
    discrete state carries no temporal persistence of its own. This subclass
    adds Fox-style self-transition stickiness ("A sticky HDP-HMM with
    application to speaker diarization", Ann. Appl. Stat. 2011): a fixed
    scalar kappa added to the diagonal of the per-step log transition
    matrix before normalisation,

        log p(z_t=j | z_{t-1}=i, x) = R_j.x + r_j + kappa * 1[i==j]

    so regime persistence lives in the DISCRETE transitions and the
    continuous AR coefficient A can stay bounded (stationary). kappa is a
    constant; not added to params, not optimised in the M-step.
    """

    def __init__(self, K, D, M, kappa):
        super(StickyRecurrentOnly, self).__init__(K, D, M=M)
        self.kappa = float(kappa)

    def log_transition_matrices(self, data, input, mask, tag):
        lp = super(StickyRecurrentOnly, self).log_transition_matrices(
            data, input, mask, tag)
        lp = lp + self.kappa * _anp.eye(self.K)[None, :, :]
        return lp - _ag_logsumexp(lp, axis=2, keepdims=True)


def _estimate_masked_loadings(C, d, y, C_mask):
    """Closed-form (EM-free) estimate of the FREE emission loadings for the
    restricted factor models, used by the EM-free fit_*_restricted paths.

    Cells with C_mask != 0 (e.g. the factor1 / factor1_vix return loadings c_i)
    are learned from data by least squares against the latent proxy. Cells with
    C_mask == 0 (the fixed fundamental rows, the VIX structural row) are left
    EXACTLY as passed. Fund / factor2 carry an all-zero mask, so their C is
    returned unchanged (locked, never learned).

    A free loading left at its 0 init makes its latent column dead in
    x_proxy = C^+(y - d), so it can never be recovered (this is what produced
    the all-zero return loading). We therefore (a) give each free cell a
    non-zero data-scale value so the column is identified, (b) build the proxy
    latent, (c) refine the free cells by regressing (y - d) on the proxy and
    re-imposing the mask so fixed cells are untouched.
    """
    if C_mask is None:
        return C
    C_mask = np.asarray(C_mask, dtype=float)
    if not np.any(C_mask != 0):
        return C
    C = np.array(C, dtype=float, copy=True)
    yc = np.asarray(y, dtype=float) - np.asarray(d, dtype=float)
    N, D = C.shape
    y_std = yc.std(axis=0)
    free = C_mask != 0
    # (a) non-zero data-scale init for any free cell still at 0
    for i in range(N):
        for j in range(D):
            if free[i, j] and C[i, j] == 0.0:
                C[i, j] = y_std[i] if y_std[i] > 1e-8 else 1.0
    # (b) proxy latent under the identified C
    U, s, Vt = np.linalg.svd(C, full_matrices=False)
    s_inv = np.where(s > 1e-8, 1.0 / s, 0.0)
    x0 = yc @ ((Vt.T * s_inv) @ U.T).T              # (T, D)
    # (c) least-squares refine of the free cells; mask re-imposed
    XtX = x0.T @ x0 + 1e-8 * np.eye(D)
    C_ls = (yc.T @ x0) @ np.linalg.inv(XtX)          # (N, D)
    C = C_mask * C_ls + (1.0 - C_mask) * C
    return C


# ---------------------------------------------------------------
# Fit rSLDS
# ---------------------------------------------------------------

def fit_rSLDS(y, params, n_iter_em=50, seed=None):
    
    """
    params: dict(n_regimes, dim_latent, single_subspace)
    """

    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)
        
    # unpack params
    N = y.shape[1]
    D = params["dim_latent"]
    K = params["n_regimes"]
    single_subspace = params["single_subspace"]

    # observation noise
    def safe_log_inv(var, lo=1e-6, hi=1e6):
        var = np.nan_to_num(var, nan=hi, posinf=hi, neginf=hi)
        var = np.clip(var, lo, hi)
        log_inv = np.log(1.0 / var)
        return log_inv

    var = np.var(y, 0, keepdims=True)
    log_inv = safe_log_inv(var)

    # OPTIMAL CLUSTERS
    if K is None:
        K, cluster_stats = fit_kmeans(y, Ks=[2, 3, 4], display=False)
        print(cluster_stats)

    # INSTANTIATE MODEL
    mdl = ssm.SLDS(N, K, D,
                   transitions="recurrent_only",
                   dynamics="diagonal_gaussian",
                   emissions="gaussian",
                   single_subspace=single_subspace)

    # INITIALIZE MODEL

    if N < D * K or D >= N:

        # ssm.SLDS.initialize cannot be used here.
        # The `D >= N` clause also covers single_subspace K=1 with N == D: there
        # the PCA in ssm's initializer explains all of y, noise_variance_ = 0,
        # and inv_etas = log(0) = -inf (emissions.py:394). That case satisfies
        # N >= D*K (so the N < D*K guard misses it) yet still must skip PCA. It runs PCA on y, then
        # builds C via (random_orthogonal @ pca.components_). When D*Keff > N
        # that matmul fails with a shape mismatch; when D*Keff == N it runs
        # but PCA's noise_variance_ is 0, so ssm sets inv_etas = log(0) = -inf,
        # silently corrupting the emission noise. We bypass it and initialise
        # manually. See ssm_init_note.md.
        # (Keff = 1 for single_subspace=True, else K.)

        # discrete-state prior π
        mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
 
        # continuous-state prior 𝒩(μ_init, diag(σ^2_init))
        mdl.dynamics.mu_init = np.zeros((K, D))
        mdl.dynamics.sigmasq_init = np.ones((K, D))

        # recurrent transition weights R, r
        mdl.transitions.Rs = 0.01 * np.random.randn(K, D)
        mdl.transitions.r = np.zeros(K)

        # The init below mirrors what ssm.SLDS.initialize does in the
        # well-specified case (data-informed per-regime b_k anchoring), but
        # uses k-means directly on y (since C = I and d = 0 below, so the
        # latent estimate equals y). This is not an algorithmic change: it
        # supplies the per-regime separation that ssm's initialiser would
        # have produced, in the regime where ssm's initialiser cannot run.
        # The package's own fallback set identical (A, b, Q) across all K
        # regimes, a symmetric attractor EM cannot escape; this anchors each
        # regime's stationary mean to a distinct k-means centroid instead.
        km = KMeans(n_clusters=K, n_init=10, random_state=seed if seed is not None else 0)
        km.fit(y)
        order = np.argsort(km.cluster_centers_[:, 0])
        cluster_means = km.cluster_centers_[order]   # (K, N)
        # Per-cluster variance of y, ordered to match cluster_means. This
        # initialises sigmasq to the data scale of each regime. A flat tiny
        # sigmasq (e.g. 1e-4) makes the iteration-1 E-step treat the
        # high-variance regime's points as gross outliers, so all
        # responsibility collapses onto one regime and the regimes degenerate.
        
        
        labels = km.labels_
        # Map old label -> rank in the sorted order so all per-cluster
        # quantities line up with cluster_means.
        old_to_new = np.empty(K, dtype=int)
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx

        # Per-cluster variance of y (data scale of each regime).
        cluster_vars = np.zeros((K, N))
        # Per-cluster lag-1 autocorrelation of y, estimated from TIME-ADJACENT
        # within-cluster pairs only (t, t+1 both in the same cluster). Using
        # the naive concatenation y[labels==k] would pair points that are not
        # adjacent in time (separated by intervening other-regime steps),
        # which inflates the autocorrelation and biases A toward the unit
        # root. For financial log-returns the true within-regime a is ~0, so
        # this estimator initialises A near zero and lets b carry the regime
        # mean directly (b_k = (1 - a_k) * centroid_k -> ~centroid_k).
        cluster_ac1 = np.zeros((K, N))
        for old_idx in range(K):
            new_idx = old_to_new[old_idx]
            pts = y[labels == old_idx]
            if pts.shape[0] >= 2:
                cluster_vars[new_idx] = np.var(pts, axis=0)
            else:
                cluster_vars[new_idx] = np.var(y, axis=0)
            # time-adjacent within-cluster pairs
            adj = (labels[:-1] == old_idx) & (labels[1:] == old_idx)
            if adj.sum() >= 2:
                y0 = y[:-1][adj]   # (n_pairs, N)
                y1 = y[1:][adj]    # (n_pairs, N)
                mu = 0.5 * (y0.mean(axis=0) + y1.mean(axis=0))
                num = ((y0 - mu) * (y1 - mu)).sum(axis=0)
                den = ((y0 - mu) ** 2).sum(axis=0) + 1e-12
                cluster_ac1[new_idx] = np.clip(num / den, -0.999, 0.999)
            else:
                cluster_ac1[new_idx] = 0.0

        # Per-regime persistence from data; pad D>N dims with 0 (no info).
        rho_init = np.zeros((K, D))
        rho_init[:, :N] = cluster_ac1
        # b_k = (1 - a_k) * mu_k so stationary mean = k-means centroid.
        cluster_means_D = np.zeros((K, D))
        cluster_means_D[:, :N] = cluster_means
        bs_init = (1.0 - rho_init) * cluster_means_D
        # Per-cluster sigmasq, padded over D, floored to avoid degeneracy.
        sigmasq_init = np.full((K, D), 1e-4)
        sigmasq_init[:, :N] = np.maximum(cluster_vars, 1e-6)
        # Diagonal A_k = a_k with small per-regime perturbation for symmetry
        # breaking; clip for stability.
        As_init = np.zeros((K, D, D))
        for k in range(K):
            diag_k = rho_init[k] + 0.01 * npr.randn(D)
            np.fill_diagonal(As_init[k], np.clip(diag_k, -0.999, 0.999))
        mdl.dynamics.As = As_init
        mdl.dynamics.bs = bs_init
        mdl.dynamics.sigmasq = sigmasq_init

        # emissions C, d, log-variance
        if single_subspace:
            mdl.emissions.Cs = np.eye(N, D)[None, :, :]  # shape (1,N,D)
            mdl.emissions.ds = np.zeros((1, N))
            mdl.emissions.inv_etas = log_inv
        else:
            mdl.emissions.Cs = np.tile(np.eye(N, D), (K, 1, 1))  # shape (K,N,D)
            mdl.emissions.ds = np.zeros((K, N))
            mdl.emissions.inv_etas = np.tile(log_inv, (K, 1))

    else:
        mdl.initialize(y)  #, discrete_state_init_method="kmeans")  # default

    # --- emission-variance floor (critical for the K=1 T1b nulls) ---
    # ssm stores the emission VARIANCE as etas = exp(inv_etas). It can reach 0
    # (-> inv_etas = -inf -> 1/exp(inv_etas) = inf in the Laplace E-step Hessian
    # at emissions.py:400 -> non-finite curvature -> AssertionError -> the fit is
    # skipped) two ways, both seen for K=1:
    #   (1) init: mdl.initialize runs PCA; when the latent fully explains y
    #       (K=1 with N == D) PCA noise_variance_ = 0 -> inv_etas = log(0). The
    #       manual-init branch only guards N < D*K, so N == D slips through here.
    #   (2) EM: with one regime the emissions M-step can drive the residual
    #       variance to ~0.
    # Floor inv_etas from below. Variance floor 1e-4 sits far under the z-scored
    # data scale (~1), so it binds only on degenerate collapse and leaves the
    # K>=2 fits (residual variance O(0.01-1)) unchanged.
    _INV_ETAS_MIN = float(np.log(1e-4))
    mdl.emissions.inv_etas = np.maximum(
        np.nan_to_num(mdl.emissions.inv_etas, neginf=_INV_ETAS_MIN, nan=_INV_ETAS_MIN),
        _INV_ETAS_MIN)
    _emi_mstep_base_uns = mdl.emissions.m_step
    def _emi_mstep_floored(*a, **k):
        _emi_mstep_base_uns(*a, **k)
        mdl.emissions.inv_etas = np.maximum(mdl.emissions.inv_etas, _INV_ETAS_MIN)
    mdl.emissions.m_step = _emi_mstep_floored

    # FIT MODEL
    elbo, q = mdl.fit(y,
                      method="laplace_em",
                      variational_posterior="structured_meanfield",
                      num_iters=n_iter_em,
                      alpha=0.0,  # default: 0.0. Laplace-EM param: new params=(1−α)⋅M-step params+α⋅old 
                      initialize=False)

    xhat = q.mean_continuous_states[0]
    zhat = mdl.most_likely_states(xhat, y)
    
    return xhat, zhat, elbo, q, mdl


def fit_rSLDS_restricted(y, params, C=None, d=None, n_iter_em=10, seed=None,
    b_pattern=None, enforce_diag_A=True,
    lam_dyn=None, lam_trn=None, delta=None, q_min=1e-6,
    C_mask=None, d_mask=None):
    """
    Restricted rSLDS via EM-free closed-form estimation.

    The model is a genuine recurrent SLDS (RecurrentOnlyTransitions: softmax
    gate over Rs.x + r, used for switching at inference). Emissions are fixed
    (C, d). It is fit WITHOUT variational EM, using closed-form per-regime
    moment estimates from a k-means partition of the latent proxy.

    Three-stage fit (all on the training batch only; leak-free):
      1. Assignment   : k-means on x_proxy = C^+(y - d).
      2. Dynamics     : per-regime (A_k, b_k, sigmasq_k) read off from each
                        cluster's within-cluster moments — time-adjacent lag-1
                        autocorrelation -> A, (1-A).centroid -> b, variance ->
                        sigmasq. No re-estimation.
      3. Gate         : Rs, r set by multinomial logistic regression of the
                        next-step k-means label on x_proxy (supervised on the
                        FIXED labels, so no E/M feedback loop -> no collapse).
                        This is the RecurrentOnlyTransitions softmax gate.

    Outputs match the EM signature: (xhat, zhat, elbo, q, mdl).
      xhat : x_proxy (= y when C=I, d=0).
      zhat : mdl.most_likely_states(xhat, y) — gate-consistent Viterbi path,
             so IS and OOS inference use the same switching rule.
      elbo : length-1 array [data log-likelihood of the Viterbi path] (finite;
             there is no iterative trace, so elbo_start == elbo_end).
      q    : minimal posterior shim exposing mean_continuous_states.

    Legacy EM knobs (n_iter_em, lam_dyn, lam_trn, delta, q_min) are accepted
    for signature compatibility and ignored. b_pattern is accepted; "zero"
    entries zero the corresponding b dimension, "mu_form"/"free" leave it.
    """
    from sklearn.linear_model import LogisticRegression

    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)

    y = np.asarray(y, dtype=float)
    T, N = y.shape
    K = int(params["n_regimes"])
    D = int(params["dim_latent"])
    assert bool(params.get("single_subspace", True)), "Require single_subspace=True."
    if b_pattern is None:
        b_pattern = ["mu_form"] * D
    assert len(b_pattern) == D and all(m in {"free", "zero", "mu_form"} for m in b_pattern)

    fixed_emissions = (C is not None) and (d is not None)
    assert fixed_emissions, "fit_rSLDS_restricted requires fixed emissions C, d."
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float)
    assert C.shape == (N, D), f"C must be (N,D)=({N},{D}), got {C.shape}"
    assert d.shape == (N,),   f"d must be (N,), got {d.shape}"

    # ----- model (genuine recurrent-only SLDS; gate used at inference)
    mdl = ssm.SLDS(
        N, K, D,
        transitions="recurrent_only",
        dynamics="diagonal_gaussian",
        emissions="gaussian",
        single_subspace=True,)

    # learn the free factor loadings closed-form (fund/factor2: mask all-zero
    # -> C returned unchanged, stays locked). Fixes the all-zero return loading.
    C = _estimate_masked_loadings(C, d, y, C_mask)

    # lock emissions to (C, d) with data-driven observation noise
    obs_var = np.var(y, axis=0)
    obs_var = np.clip(np.nan_to_num(obs_var, nan=1.0, posinf=1e6, neginf=1e6), 1e-8, 1e6)
    inv_etas = np.tile(np.log(1.0 / obs_var)[None, :], (K, 1))
    mdl.emissions.Cs = C[None, :, :]
    mdl.emissions.ds = d[None, :]
    mdl.emissions.inv_etas = inv_etas

    # latent proxy x = C^+(y - d)
    U_svd, s_svd, Vt_svd = np.linalg.svd(C, full_matrices=False)
    s_safe = np.where(s_svd > 1e-8, s_svd, 1.0)
    s_inv  = np.where(s_svd > 1e-8, 1.0 / s_safe, 0.0)
    C_pinv = (Vt_svd.T * s_inv) @ U_svd.T      # (D, N)
    x_proxy = (y - d) @ C_pinv.T               # (T, D)

    # ----- stage 1: assignment (k-means on [level, squared successive-diff])
    # The clustering feature space spans BOTH regime axes:
    #   - level   = x_proxy_t                          -> separates drift regimes
    #   - vol     = (x_proxy_t - x_proxy_{t-1})^2      -> separates diffusion regimes
    # Squared successive difference isolates local movement (volatility)
    # independent of level, so a turbulent regime is separable from a calm one
    # even at identical mean. Both feature groups are standardised to unit
    # variance so k-means weights drift and vol comparably; with that, k-means
    # finds whatever direction (drift, vol, or their diagonal) best separates
    # the K clusters. Causal and parameter-free. Leak-free: batch data only.
    vol_feat = np.zeros((T, D))
    vol_feat[1:] = (x_proxy[1:] - x_proxy[:-1]) ** 2
    vol_feat[0] = vol_feat[1] if T > 1 else 0.0       # first point has no diff; copy next
    feat = np.concatenate([x_proxy, vol_feat], axis=1)   # (T, 2D)
    feat_mean = feat.mean(axis=0, keepdims=True)
    feat_std = feat.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std > 1e-12, feat_std, 1.0)  # avoid /0 on constant dims
    feat_z = (feat - feat_mean) / feat_std               # standardised features

    km = KMeans(n_clusters=K, n_init=10, random_state=seed if seed is not None else 0)
    km.fit(feat_z)
    # Deterministic regime order by the LEVEL (drift) centroid, recovered in
    # original units from the standardised level dimension (first D cols).
    level_centroid_z = km.cluster_centers_[:, :D]                 # (K, D) standardised
    level_centroid = level_centroid_z * feat_std[:, :D] + feat_mean[:, :D]
    order = np.argsort(level_centroid[:, 0])          # order by first level dim
    old_to_new = np.empty(K, dtype=int)
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
    labels = np.array([old_to_new[l] for l in km.labels_], dtype=int)   # (T,) in new order
    # cluster_means: per-regime mean of x_proxy under the new assignment (the
    # quantity stage 2 needs; recompute from labels rather than from feature
    # centroids so it is exactly the within-cluster level mean).
    cluster_means = np.zeros((K, D))
    for k in range(K):
        pts = x_proxy[labels == k]
        cluster_means[k] = pts.mean(axis=0) if pts.shape[0] > 0 else x_proxy.mean(axis=0)

    # ----- stage 2: per-regime dynamics from within-cluster moments
    cluster_vars = np.zeros((K, D))
    cluster_ac1  = np.zeros((K, D))
    for k in range(K):
        pts = x_proxy[labels == k]
        cluster_vars[k] = np.var(pts, axis=0) if pts.shape[0] >= 2 else np.var(x_proxy, axis=0)
        adj = (labels[:-1] == k) & (labels[1:] == k)
        if adj.sum() >= 2:
            x0 = x_proxy[:-1][adj]; x1 = x_proxy[1:][adj]
            mu = 0.5 * (x0.mean(axis=0) + x1.mean(axis=0))
            num = ((x0 - mu) * (x1 - mu)).sum(axis=0)
            den = ((x0 - mu) ** 2).sum(axis=0) + 1e-12
            cluster_ac1[k] = np.clip(num / den, -0.999, 0.999)
    rho = cluster_ac1                                  # (K, D)
    bs  = (1.0 - rho) * cluster_means                  # (K, D)
    for d_idx, mode in enumerate(b_pattern):
        if mode == "zero":
            bs[:, d_idx] = 0.0
    sigmasq = np.maximum(cluster_vars, q_min)          # (K, D)
    As = np.zeros((K, D, D))
    for k in range(K):
        np.fill_diagonal(As[k], np.clip(rho[k], -0.999, 0.999))
    mdl.dynamics.As = As
    mdl.dynamics.bs = bs
    mdl.dynamics.sigmasq = sigmasq
    mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
    # report mu = b/(1-rho) (matches EM-path attribute)
    denom = np.clip(1.0 - rho, 1e-8, None)
    mdl.dynamics_mu_param = bs / denom

    # ----- stage 3: supervised softmax gate (RecurrentOnlyTransitions)
    # P(z_{t+1}=k | x_t) = softmax_k(Rs[k].x_t + r[k]).  Fit Rs, r by
    # multinomial logistic regression of next-step label on x_proxy, on the
    # FIXED k-means labels (no E/M feedback). Ws = 0 (no exogenous inputs).
    Rs = np.zeros((K, D))
    r_vec = np.zeros(K)
    X_gate = x_proxy[:-1]            # (T-1, D)
    y_gate = labels[1:]             # (T-1,)
    present = np.unique(y_gate)
    if present.size >= 2:
        lr = LogisticRegression(solver="lbfgs",
                                C=1e4, max_iter=1000)
        lr.fit(X_gate, y_gate)
        coef = np.asarray(lr.coef_, dtype=float)          # (n_rows, D)
        intr = np.asarray(lr.intercept_, dtype=float)     # (n_rows,)
        classes = [int(c) for c in lr.classes_]
        if coef.shape[0] == 1 and len(classes) == 2:
            # Binary case: sklearn returns ONE coefficient row for the positive
            # class (classes[1]); the negative class (classes[0]) is the
            # implicit reference at 0. The softmax gate is shift-invariant, so
            # set the positive class to (coef, intr) and leave the reference at 0.
            Rs[classes[1]] = coef[0]
            r_vec[classes[1]] = intr[0]
        else:
            # Multiclass: one row per class in lr.classes_.
            for row, cls in enumerate(classes):
                Rs[cls] = coef[row]
                r_vec[cls] = intr[row]
    mdl.transitions.Ws = np.zeros((K, 0))   # M = 0, no inputs
    mdl.transitions.Rs = Rs
    mdl.transitions.r = r_vec

    # ----- outputs (gate-consistent)
    xhat = x_proxy
    zhat = mdl.most_likely_states(xhat, y)

    # data log-likelihood of the Viterbi path (finite scalar; no EM trace).
    x_mask = np.ones_like(xhat, dtype=bool)              # latent-shaped (T, D)
    y_mask = np.ones_like(y, dtype=bool)                 # observation-shaped (T, N)
    log_pi0 = mdl.init_state_distn.log_pi0 - logsumexp(mdl.init_state_distn.log_pi0)
    log_Ps = mdl.transitions.log_transition_matrices(xhat, np.zeros((T, 0)), x_mask, None)  # (T-1, K, K)
    ll_dyn = mdl.dynamics.log_likelihoods(xhat, np.zeros((T, 0)), x_mask, None)             # (T, K)
    ll_emi = mdl.emissions.log_likelihoods(y, np.zeros((T, 0)), y_mask, None, xhat)         # (T, K)
    ll_obs = ll_dyn + ll_emi
    path_ll = float(log_pi0[zhat[0]] + ll_obs[0, zhat[0]])
    for t in range(1, T):
        path_ll += float(log_Ps[t - 1, zhat[t - 1], zhat[t]] + ll_obs[t, zhat[t]])
    elbo = np.array([path_ll], dtype=float)

    class _QShim:
        def __init__(self, datas):
            self.mean_continuous_states = [np.asarray(dd, dtype=float) for dd in datas]
    q = _QShim([xhat])

    return xhat, zhat, elbo, q, mdl
    

def fit_rSLDS_restricted_em(y, params, C=None, d=None, n_iter_em=10, seed=None,
    b_pattern=None, enforce_diag_A=True,
    q_min=1e-6,              # floor on dynamics variance
    alpha_damp=0.5,          # Laplace-EM damping: new=(1-a)*Mstep+a*old; >0 prevents overshoot
    C_mask=None, d_mask=None,
    # ---- anti-collapse guardrails (pre-declared; set on principled grounds,
    #      NOT tuned to test outcomes) ----
    warm_start=True,         # seed EM from the closed-form (k-means) restricted fit
    em_tol=1e-3,             # CONVERGENCE tolerance on the RUNNING-BEST ELBO: converged when
                             # the best ELBO fails to improve by more than em_tol (relative) for
                             # EM_PATIENCE consecutive checks. Best-ELBO (not consecutive-change)
                             # is used because the projected-EM ELBO is non-monotone -- its
                             # consecutive-step relative changes oscillate at the few-percent
                             # level, while the running-best plateaus. cap = n_iter_em iterations;
                             # the highest-ELBO iterate is retained as a drift guard.
    min_occupancy=0.05):     # per-regime usage floor; an EM run that drops the occupied-regime
                             # count below the closed-form's is rejected in favour of that fit

    """
    True rSLDS via ssm (Laplace EM + structured mean field):
    - If C,d provided: emissions are fixed & shared (single_subspace=True).
    - If C,d None: emissions are learned as usual (unrestricted).
    - Robust emissions inversion with SVD+ridge (handles N<D and rank-deficient C).
    - Split-EM outer loop (alpha=0.0):
        * E-only pass  (freeze all m_steps) -> 1 iter
        * M-enabled pass (enable dynamics/transitions/init; emissions fixed iff C,d provided) -> 1 iter
        * Repeat n_iter_em times.
    - After each M pass:
        * enforce_diag_A (zero off-diagonals of A_k); clip |rho_k| <= a_max on diagonal
        * b_pattern handling with µ bookkeeping (µ = b/(1-ρ))
        * Emissions projection if fixed_emissions (respects C_mask, d_mask)
    - Final E-only pass
    - Returns (xhat, zhat, elbo_trace, q_last, mdl)

    NOTE: This is the main EM-based restricted fitting algorithm: data-driven
    init, a per-regime persistence cap (a_max), and a single coupled Laplace-EM
    with damping and per-iteration identifiability constraints.

    ANTI-COLLAPSE GUARDRAILS (pre-declared, principled, frozen before scoring):
      1. warm start  - the closed-form (k-means) restricted fit does the heavy
                       lifting; EM is seeded there and refines.
      2. convergence - coupled Laplace-EM is iterated (resuming the variational
                       posterior) until the RUNNING-BEST ELBO stops improving by
                       more than em_tol (relative) over EM_PATIENCE consecutive
                       checks, capped at n_iter_em iterations. Best-ELBO (not
                       consecutive-step change) is the convergence target because
                       the hard stationarity/projection constraints make the
                       projected-EM ELBO non-monotone: consecutive-step relative
                       changes oscillate at the few-percent level while the
                       running-best plateaus. The highest-ELBO iterate is retained
                       as a drift guard. If the cap is reached first the run is
                       logged as not-converged. Collapse is prevented by the
                       constraints below; a high-ELBO-but-collapsed iterate is
                       caught by the occupancy test.
      3. stationarity- |rho_k| clipped to a_max < 1 every M-step (no unit-root /
                       explosive A); process variance floored at q_min.
      4. stickiness  - StickyRecurrentOnly gate (kappa=_STICKY_KAPPA) biases
                       self-transitions -> enforces dwell, discourages flip-flop.
      5. occupancy   - EM is accepted only if it preserves the closed-form fit's
                       count of regimes above the min_occupancy floor; otherwise
                       the closed-form fit is retained. EM that diverges outright
                       also falls back to the closed-form fit (never worse than
                       not running EM).
    These prevent degenerate optima; they do not and cannot force regimes that
    do not transfer out of sample (that is what T1b/T2/T3 still test).

    Notes on C_mask, d_mask:
        Arrays same shape as Cs, ds (or broadcastable). Entries in {0, 1} (or bool).
        1 = keep the M-step-learned value; 0 = overwrite with C_fix / d_fix.
        Default (None) = all zeros = fully fix emissions to provided C, d.
    """

    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)

    # Shapes & params
    y = np.asarray(y, dtype=float)
    T, N = y.shape
    K = int(params["n_regimes"])
    D = int(params["dim_latent"])
    assert bool(params.get("single_subspace", True)), "Require single_subspace=True."

    # knobs
    if b_pattern is None:
        b_pattern = ["mu_form"] * D
    assert len(b_pattern) == D and all(m in {"free", "zero", "mu_form"} for m in b_pattern)

    # Whether any emission entry is FREE to learn (factor1 / factor1_vix: the factor
    # loadings c_i are learned). Fully-fixed models (fund*, factor2*) have all-zero
    # masks and keep emissions locked. This is what lets the c_i actually be estimated
    # rather than staying pinned at their 0 init.
    has_learnable_emissions = (
        (C_mask is not None and np.any(np.asarray(C_mask) != 0)) or
        (d_mask is not None and np.any(np.asarray(d_mask) != 0)))

    # ----- model
    mdl = ssm.SLDS(
        N, K, D,
        transitions="recurrent_only",       # placeholder; swapped to StickyRecurrentOnly below
        dynamics="diagonal_gaussian",
        emissions="gaussian",
        single_subspace=True,)

    # Swap recurrent_only gate for the sticky variant: Linderman 2017
    # recurrent_only gating + Fox 2011 fixed self-transition stickiness at
    # _STICKY_KAPPA (see StickyRecurrentOnly).
    mdl.transitions = StickyRecurrentOnly(K, D, M=0, kappa=_STICKY_KAPPA)

    # ----- emissions (fixed if C,d provided; else learned)
    fixed_emissions = (C is not None) and (d is not None)
    if fixed_emissions:
        C = np.asarray(C, dtype=float)
        d = np.asarray(d, dtype=float)
        assert C.shape == (N, D),  f"C must be (N,D)=({N},{D}), got {C.shape}"
        assert d.shape == (N,),    f"d must be (N,), got {d.shape}"

        # simple data-driven noise init, then lock
        obs_var = np.var(y, axis=0)
        obs_var = np.clip(np.nan_to_num(obs_var, nan=1.0, posinf=1e6, neginf=1e6), 1e-8, 1e6)
        inv_etas_row = np.log(1.0 / obs_var)[None, :]     # (1, N)
        inv_etas     = np.tile(inv_etas_row, (K, 1))      # (K, N)
        mdl.emissions.Cs = C[None, :, :]
        mdl.emissions.ds = d[None, :]
        mdl.emissions.inv_etas = inv_etas
        mdl.emissions.m_step = (lambda *_, **__: None)  # lock emissions

    # robust inversion for N<D and low-rank C, whether fixed or learned
    def _invert_ridge(self, data, input=None, mask=None, tag=None, ridge=1e-6):
        Y = np.atleast_2d(np.asarray(data, dtype=float)) # (T,N)
        # pull correct C,d (single_subspace)
        Cc = self.Cs[0] if self.Cs.ndim == 3 else self.Cs   # (N, D)
        dc = self.ds[0] if self.ds.ndim == 2 else self.ds   # (N,)
        Yc = Y - dc
        # SVD ridge pseudoinverse: Pinv ≈ (C^T C + λI)^{-1} C^T, done via SVD
        U, s, Vt = np.linalg.svd(Cc, full_matrices=False)   # C = U diag(S) Vt
        s_f = s / (s**2 + ridge)
        Pinv = (Vt.T * s_f) @ U.T                           # (D,N) = C^+
        X = (Pinv @ (Yc.T)).T                               # (T,D)
        return X

    # attach
    mdl.emissions._invert = _invert_ridge.__get__(mdl.emissions, mdl.emissions.__class__)

    # ----- initialisation
    if fixed_emissions and N < D * K:
        # k-means init can't run via ssm (PCA needs N >= D*K for some variants);
        # do our own k-means on y to derive data-informed per-regime dynamics.
        # This is critical: random b init around zero causes EM to converge
        # to b ≈ 0 for all regimes (the local optimum nearest the init).
        from sklearn.cluster import KMeans
        mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
        mdl.dynamics.mu_init     = np.zeros((K, D))
        mdl.dynamics.sigmasq_init = np.ones((K, D))

        # Pseudo-invert C to get per-timestep latent estimate from y.
        # For C = I (fund2 / price clamped to identity), x_t ≈ y_t.
        # For general C, use SVD pseudo-inverse.
        U_svd, s_svd, Vt_svd = np.linalg.svd(C, full_matrices=False)
        s_safe = np.where(s_svd > 1e-8, s_svd, 1.0)
        s_inv  = np.where(s_svd > 1e-8, 1.0 / s_safe, 0.0)
        C_pinv = (Vt_svd.T * s_inv) @ U_svd.T  # (D, N)
        x_proxy = (y - d) @ C_pinv.T  # (T, D)

        # k-means on the proxy latent to find K cluster centres along each dim.
        km = KMeans(n_clusters=K, n_init=10, random_state=seed if seed is not None else 0)
        km.fit(x_proxy)
        # Order clusters by first-dim centre so labelling is deterministic.
        order = np.argsort(km.cluster_centers_[:, 0])
        cluster_means = km.cluster_centers_[order]  # (K, D)

        # Per-cluster moments of x_proxy, ordered to match cluster_means.
        labels = km.labels_
        old_to_new = np.empty(K, dtype=int)
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx

        # Per-cluster variance (regime data scale). A flat tiny sigmasq makes
        # the iteration-1 E-step treat the high-variance regime's points as
        # gross outliers, collapsing responsibility onto one regime.
        cluster_vars = np.zeros((K, D))
        # Per-cluster lag-1 autocorrelation from TIME-ADJACENT within-cluster
        # pairs only (t, t+1 both in the same cluster). The naive concatenation
        # x_proxy[labels==k] would pair non-time-adjacent points (separated by
        # intervening other-regime steps), inflating A toward the unit root.
        # For financial log-returns the true within-regime a is ~0, so this
        # initialises A near zero and lets b carry the regime mean directly
        # (b_k = (1 - a_k) * centroid_k -> ~centroid_k).
        cluster_ac1 = np.zeros((K, D))
        for old_idx in range(K):
            new_idx = old_to_new[old_idx]
            pts = x_proxy[labels == old_idx]
            if pts.shape[0] >= 2:
                cluster_vars[new_idx] = np.var(pts, axis=0)
            else:
                cluster_vars[new_idx] = np.var(x_proxy, axis=0)
            adj = (labels[:-1] == old_idx) & (labels[1:] == old_idx)
            if adj.sum() >= 2:
                x0 = x_proxy[:-1][adj]   # (n_pairs, D)
                x1 = x_proxy[1:][adj]    # (n_pairs, D)
                mu = 0.5 * (x0.mean(axis=0) + x1.mean(axis=0))
                num = ((x0 - mu) * (x1 - mu)).sum(axis=0)
                den = ((x0 - mu) ** 2).sum(axis=0) + 1e-12
                cluster_ac1[new_idx] = np.clip(num / den, -0.999, 0.999)
            else:
                cluster_ac1[new_idx] = 0.0

        # Per-regime persistence from data; b_k = (1 - a_k) * mu_k so the
        # stationary mean of each regime equals its k-means centroid.
        rho_init = cluster_ac1                       # (K, D)
        bs_init = (1.0 - rho_init) * cluster_means
        sigmasq_init = np.maximum(cluster_vars, 1e-6)
        # Diagonal A_k = a_k with small per-regime perturbation for symmetry
        # breaking; clip for stability.
        As_init = np.zeros((K, D, D))
        for k in range(K):
            diag_k = rho_init[k] + 0.01 * npr.randn(D)
            np.fill_diagonal(As_init[k], np.clip(diag_k, -0.999, 0.999))

        mdl.dynamics.As     = As_init
        mdl.dynamics.bs     = bs_init
        mdl.dynamics.sigmasq = sigmasq_init
        mdl.transitions.Rs   = 0.1 * npr.randn(K, D)
        mdl.transitions.r    = np.zeros(K)

        # Data-driven persistence cap for the M-step. ssm's AR(1) dynamics
        # M-step trades drift (b) for persistence (A), walking A toward the
        # unit root and shrinking b toward zero. We cap |A| at the largest
        # within-cluster lag-1 autocorrelation observed at init, plus one
        # standard error (~1/sqrt(n_min) for n_min the smallest cluster size).
        # The cap attenuates the A->unit-root drift during EM. Leak-free: it is
        # read from the IS k-means partition only.
        # Closure local read by _enforce_identifiability_and_mu.
        cluster_sizes = np.array([int((labels == old_idx).sum()) for old_idx in range(K)])
        n_min = max(int(cluster_sizes.min()), 1)
        a_max = float(min(0.99, np.max(np.abs(cluster_ac1)) + 1.0 / np.sqrt(n_min)))

    else:
        # k-means based init via ssm's built-in
        mdl.initialize(y)
        # No data-driven cap available without the k-means partition; fall
        # back to the stability bound so behaviour is unchanged on this path.
        a_max = 0.999

    # If emissions are fixed, re-pin (initialize(y) overwrites Cs/ds/inv_etas
    # via PCA on y, which is wrong when we want C, d locked to user values).
    if fixed_emissions:
        obs_var = np.var(y, axis=0)
        obs_var = np.clip(np.nan_to_num(obs_var, nan=1.0, posinf=1e6, neginf=1e6), 1e-8, 1e6)
        inv_etas_row = np.log(1.0 / obs_var)[None, :]
        inv_etas     = np.tile(inv_etas_row, (K, 1))
        mdl.emissions.Cs       = C[None, :, :]
        mdl.emissions.ds       = d[None, :]
        mdl.emissions.inv_etas = inv_etas

    # ---- warm start: let the closed-form (k-means) restricted fit do the heavy
    #      lifting, then run EM to convergence within the constrained space. The
    #      closed-form solution is stable and non-degenerate; seeding EM there
    #      (vs a cold init) keeps EM in a good basin. base_state also serves as
    #      the fallback if EM diverges or collapses.
    base_state = None        # (xhat, zhat, q, mdl, occupancy) from the closed-form fit
    if warm_start and (C is not None) and (d is not None):
        try:
            x_base, z_base, _eb, q_base, base_mdl = fit_rSLDS_restricted(
                y, params, C=C, d=d, n_iter_em=n_iter_em, seed=seed,
                b_pattern=b_pattern, enforce_diag_A=enforce_diag_A,
                C_mask=C_mask, d_mask=d_mask)
            mdl.dynamics.As      = np.array(base_mdl.dynamics.As, dtype=float)
            mdl.dynamics.bs      = np.array(base_mdl.dynamics.bs, dtype=float)
            mdl.dynamics.sigmasq = np.maximum(np.array(base_mdl.dynamics.sigmasq, dtype=float), q_min)
            mdl.transitions.Rs   = np.array(base_mdl.transitions.Rs, dtype=float)
            mdl.transitions.r    = np.array(base_mdl.transitions.r, dtype=float)
            mdl.emissions.Cs       = np.array(base_mdl.emissions.Cs, dtype=float)
            mdl.emissions.ds       = np.array(base_mdl.emissions.ds, dtype=float)
            mdl.emissions.inv_etas = np.array(base_mdl.emissions.inv_etas, dtype=float)
            z_base = np.asarray(z_base, dtype=int)
            base_occ = np.bincount(z_base, minlength=K) / max(z_base.size, 1)
            base_state = (x_base, z_base, q_base, base_mdl, base_occ)
        except Exception:
            # closed-form warm start unavailable -> fall back to the cold init
            # already set above; run full EM as before.
            base_state = None

    # store µ for reporting (µ = b/(1-ρ))
    mdl.dynamics_mu_param = np.zeros((K, D))

    # ----- helpers: bind m_step, freeze/enable, constraints
    def _bind_mstep(comp):
        return (getattr(comp.__class__, "_m_step", None)
                or getattr(comp.__class__, "m_step")).__get__(comp)

    dyn_mstep_base = _bind_mstep(mdl.dynamics)
    trn_mstep_base = _bind_mstep(mdl.transitions)
    pio_mstep_base = _bind_mstep(mdl.init_state_distn)
    emi_mstep_base = _bind_mstep(mdl.emissions)

    def _freeze_all():
        mdl.dynamics.m_step = (lambda *_, **__: None)
        mdl.transitions.m_step = (lambda *_, **__: None)
        mdl.init_state_distn.m_step = (lambda *_, **__: None)
        mdl.emissions.m_step = (lambda *_, **__: None)

    def dyn_mstep_plain(*args, **kwargs):
        dyn_mstep_base(*args, **kwargs)     # ssm closed-form update
        # floor process variances
        if hasattr(mdl.dynamics, "sigmasq"):
            mdl.dynamics.sigmasq = np.maximum(mdl.dynamics.sigmasq, q_min)
        # Enforce identifiability EVERY M-step (previously applied only once per
        # outer iteration). With the single coupled laplace_em below, folding the
        # A-persistence clip and b zero-pattern in here keeps every internal
        # iteration constrained, preventing the A->unit-root drift / overshoot
        # that produced the non-monotonic ELBO blow-ups.
        A = mdl.dynamics.As
        for k in range(K):
            if enforce_diag_A:
                A[k] = np.diag(np.clip(np.diag(A[k]), -a_max, a_max))
            else:
                di = np.clip(np.diag(A[k]), -a_max, a_max)
                np.fill_diagonal(A[k], di)
        mdl.dynamics.As = A
        B = mdl.dynamics.bs
        for d_idx, mode in enumerate(b_pattern):
            if mode == "zero":
                B[:, d_idx] = 0.0
        mdl.dynamics.bs = B

    def trn_mstep_plain(*args, **kwargs):
        if K == 1:
            return
        trn_mstep_base(*args, **kwargs)     # ssm optimization over recurrent gates

    def _enable_M_pass():
        mdl.dynamics.m_step = dyn_mstep_plain
        mdl.transitions.m_step = trn_mstep_plain
        mdl.init_state_distn.m_step = pio_mstep_base
        # Emissions:
        #   - unrestricted        -> learn full C (ssm base m_step)
        #   - factor1 / _vix      -> learn the C_mask=1 loadings, then clamp fixed cells
        #                            (emi_mstep_masked); this is what estimates c_i
        #   - fund* / factor2*    -> fully fixed (no learnable cells) -> locked
        if not fixed_emissions:
            mdl.emissions.m_step = emi_mstep_base
        elif has_learnable_emissions:
            mdl.emissions.m_step = emi_mstep_masked
        else:
            mdl.emissions.m_step = (lambda *_, **__: None)

    def _enforce_identifiability_and_mu():
        # Honors enforce_diag_A. If False, keep off-diagonals; only
        # clip the diagonal (stability proxy).
        # The clip bound is a_max (data-driven, set at init): it attenuates the
        # AR(1) M-step's drift of A toward the unit root. For returns a_max is
        # small (~0.1-0.2); for persistent channels it relaxes toward 1.
        A = mdl.dynamics.As
        for k in range(K):
            if enforce_diag_A:
                diag = np.clip(np.diag(A[k]), -a_max, a_max)
                A[k] = np.diag(diag)
            else:
                di = np.clip(np.diag(A[k]), -a_max, a_max)
                np.fill_diagonal(A[k], di)
        mdl.dynamics.As = A

        # Record μ = b/(1-ρ) per b_pattern; zero b for "zero" pattern.
        B = mdl.dynamics.bs
        for d_idx, mode in enumerate(b_pattern):
            if mode == "zero":
                B[:, d_idx] = 0.0
                mdl.dynamics_mu_param[:, d_idx] = 0.0
            elif mode == "mu_form":
                rho = A[:, d_idx, d_idx]
                denom = np.clip(1.0 - rho, 1e-8, None)
                mdl.dynamics_mu_param[:, d_idx] = B[:, d_idx] / denom
            # "free": leave b and μ untouched
        mdl.dynamics.bs = B

        # Do NOT re-lock emissions here. _project_emissions_ is the single
        # authority and is called immediately after when fixed_emissions.

    def _project_emissions_(emissions, C_fix, d_fix, C_mask=None, d_mask=None):
        Cs = emissions.Cs  # shape: (K,N,D) or (N,D)
        ds = emissions.ds  # shape: (K,N)   or (N,)

        # default: fully fixed if mask not provided
        if C_mask is None:
            C_mask = np.zeros_like(Cs, dtype=float)
        else:
            C_mask = np.asarray(C_mask, dtype=float)

        if d_mask is None:
            d_mask = np.zeros_like(ds, dtype=float)
        else:
            d_mask = np.asarray(d_mask, dtype=float)

        C_fix = np.asarray(C_fix, dtype=float)
        d_fix = np.asarray(d_fix, dtype=float)

        # -------- broadcast C parts --------
        if Cs.ndim == 3 and C_fix.ndim == 2:
            C_fix = np.broadcast_to(C_fix, Cs.shape)
        elif Cs.ndim == 2 and C_fix.ndim == 3:
            C_fix = C_fix[0]

        if Cs.ndim == 3 and C_mask.ndim == 2:
            C_mask = np.broadcast_to(C_mask, Cs.shape)
        elif Cs.ndim == 2 and C_mask.ndim == 3:
            C_mask = C_mask[0]

        # -------- broadcast d parts --------
        if ds.ndim == 2 and d_fix.ndim == 1:
            d_fix = np.broadcast_to(d_fix, ds.shape)
        elif ds.ndim == 1 and d_fix.ndim == 2:
            d_fix = d_fix[0]

        if ds.ndim == 2 and d_mask.ndim == 1:
            d_mask = np.broadcast_to(d_mask, ds.shape)
        elif ds.ndim == 1 and d_mask.ndim == 2:
            d_mask = d_mask[0]

        # -------- apply projection --------
        emissions.Cs = C_mask * Cs + (1.0 - C_mask) * C_fix
        emissions.ds = d_mask * ds + (1.0 - d_mask) * d_fix

    def emi_mstep_masked(*args, **kwargs):
        # Run ssm's emissions M-step so the FREE loadings (C_mask=1) are estimated,
        # then clamp the FIXED cells (C_mask=0 / d_mask=0) back to C, d. Net effect:
        # factor1/_vix learn c_i while the VIX row [0..0,1] and d=0 stay fixed.
        emi_mstep_base(*args, **kwargs)
        # ssm's single_subspace emissions M-step collapses inv_etas to (1,N); the
        # manual init uses (K,N). Restore (K,N) so the flattened emissions-param
        # length is stable across the M-step (ssm's laplace-EM damping requires it).
        ie = np.asarray(mdl.emissions.inv_etas)
        if ie.ndim == 2 and ie.shape[0] == 1 and K > 1:
            mdl.emissions.inv_etas = np.tile(ie, (K, 1))
        _project_emissions_(mdl.emissions, C_fix=C, d_fix=d, C_mask=C_mask, d_mask=d_mask)


    # ----- coupled Laplace-EM, warm-started, within the constrained space.
    # Run in small chunks (resuming the posterior) until the RUNNING-BEST ELBO
    # plateaus (no >em_tol relative gain over EM_PATIENCE checks = CONVERGENCE),
    # or the n_iter_em cap is hit. Best-ELBO is the convergence target because the
    # projected-EM ELBO is non-monotone under the hard constraints: consecutive-
    # step relative changes oscillate at the few-percent level, while the running-
    # best plateaus -- so a consecutive-change tol measures oscillation, not
    # progress. The highest-ELBO iterate is retained (drift guard). The
    # constrained m_step (A-clip to a_max, b zero-pattern, sigmasq floor) + sticky
    # gate fire every internal iteration, so collapse is prevented by the
    # constraints; a high-ELBO-but-collapsed iterate is still caught by the
    # occupancy test below.
    def _snapshot():
        return dict(
            As=np.array(mdl.dynamics.As), bs=np.array(mdl.dynamics.bs),
            sigmasq=np.array(mdl.dynamics.sigmasq),
            Rs=np.array(mdl.transitions.Rs), r=np.array(mdl.transitions.r),
            Cs=np.array(mdl.emissions.Cs), ds=np.array(mdl.emissions.ds),
            inv_etas=np.array(mdl.emissions.inv_etas))

    def _restore(s):
        mdl.dynamics.As = s["As"]; mdl.dynamics.bs = s["bs"]
        mdl.dynamics.sigmasq = s["sigmasq"]
        mdl.transitions.Rs = s["Rs"]; mdl.transitions.r = s["r"]
        mdl.emissions.Cs = s["Cs"]; mdl.emissions.ds = s["ds"]
        mdl.emissions.inv_etas = s["inv_etas"]

    EM_CHUNK = 2             # iterations between convergence checks
    EM_PATIENCE = 3          # consecutive checks w/o a >em_tol best-ELBO gain -> converged
    elbo_trace = []
    q_last = None
    _enable_M_pass()
    max_attempts = 5
    cap = int(n_iter_em)
    iters_done = 0
    best_elbo = -np.inf
    best_snap = None
    q_best = None
    no_improve = 0
    converged = False
    while iters_done < cap:
        n_this = min(EM_CHUNK, cap - iters_done)
        fit_ok = False
        for attempt in range(max_attempts):
            try:
                vp = "structured_meanfield" if q_last is None else q_last
                elbo, q_last = mdl.fit(
                    y, method="laplace_em", variational_posterior=vp,
                    num_iters=n_this, alpha=alpha_damp, initialize=False, verbose=0)
                fit_ok = True
                break
            except AssertionError:
                if attempt == max_attempts - 1:
                    break
                rng = np.random.RandomState(seed=hash(("EM", iters_done, attempt)) % (2**31 - 1))
                mdl.dynamics.bs = mdl.dynamics.bs + 1e-3 * rng.randn(*mdl.dynamics.bs.shape)
        if not fit_ok:
            break
        elbo_trace.extend(list(elbo))
        iters_done += len(elbo)
        cur = float(elbo[-1])
        # CONVERGENCE on the RUNNING-BEST ELBO. The projected-EM ELBO is non-monotone:
        # consecutive-step relative changes oscillate at the few-percent level, so a
        # consecutive-change tol is the wrong instrument (it measures oscillation, not
        # progress). The running-best ELBO is what plateaus. Converged := the best ELBO
        # fails to improve by more than em_tol (relative) for EM_PATIENCE consecutive
        # checks. best_snap always tracks the argmax iterate (drift guard).
        improved_materially = (np.isfinite(cur) and
                               cur > best_elbo + em_tol * (abs(best_elbo) + 1e-12))
        if np.isfinite(cur) and cur > best_elbo:
            best_elbo = cur
            best_snap = _snapshot()
            q_best = q_last
        if improved_materially:
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= EM_PATIENCE:
                converged = True
                break

    # Restore the highest-ELBO iterate (drift guard) and log convergence status.
    # If EM never produced a usable iterate, fall back to the closed-form fit
    # (divergent EM must never be worse than no EM).
    if best_snap is not None:
        _restore(best_snap)
        q_use = q_best
        status = "converged" if converged else "did NOT converge within n_iter_em cap"
        print(f"[em-status] K={K} D={D} N={N}: {status} in {iters_done} iters "
              f"(best ELBO={best_elbo:.6g}, tol={em_tol:g})", flush=True)
    elif q_last is not None:
        q_use = q_last
    elif base_state is not None:
        x_base, z_base, q_base, base_mdl, _ = base_state
        print(f"[em-fallback] K={K} D={D} N={N}: EM diverged -> closed-form fit retained", flush=True)
        return x_base, z_base, np.asarray([], dtype=float), q_base, base_mdl
    else:
        raise RuntimeError("Laplace-EM failed and no closed-form fallback was available.")

    # final identifiability + emission projection (one authoritative pass)
    _enforce_identifiability_and_mu()
    if fixed_emissions:
        _project_emissions_(mdl.emissions, C_fix=C, d_fix=d, C_mask=C_mask, d_mask=d_mask)
    xhat = q_use.mean_continuous_states[0]
    zhat = mdl.most_likely_states(xhat, y)

    # ---- anti-collapse acceptance test: EM is accepted only if it preserves the
    #      regimes the closed-form fit found. If the highest-ELBO iterate still
    #      collapsed a regime (occupied-regime count below the closed-form's,
    #      vs the min_occupancy floor), revert to the closed-form fit. The
    #      [em-revert] log lets the run report the revert rate.
    if base_state is not None and K > 1:
        x_base, z_base, q_base, base_mdl, base_occ = base_state
        z_em = np.asarray(zhat, dtype=int)
        em_occ = np.bincount(z_em, minlength=K) / max(z_em.size, 1)
        base_used = int((base_occ >= min_occupancy).sum())
        em_used   = int((em_occ   >= min_occupancy).sum())
        if em_used < base_used:
            print(f"[em-revert] K={K} D={D} N={N}: EM occupied {em_used}/{K} regimes "
                  f"vs closed-form {base_used}/{K} (floor={min_occupancy}) -> "
                  f"closed-form fit retained", flush=True)
            return (x_base, z_base, np.asarray(elbo_trace, dtype=float),
                    q_base, base_mdl)

    # EM fit accepted (occupancy guard passed). Logged so the run can report the
    # revert rate = #[em-revert] / (#[em-accept] + #[em-revert]) over K>1 fits.
    if base_state is not None and K > 1:
        print(f"[em-accept] K={K} D={D} N={N}: EM fit accepted (occupancy guard passed)",
              flush=True)
    return xhat, zhat, np.asarray(elbo_trace, dtype=float), q_use, mdl


# ---------------------------------------------------------------
# Fit HMM (ssm) — strict-parity drop-in for fit_rSLDS
# ---------------------------------------------------------------
#
# Parity choices (matched to fit_rSLDS):
#   - observations   : "diagonal_gaussian"  (per-regime diagonal covariance;
#                      matches rSLDS dynamics="diagonal_gaussian" + diagonal
#                      emission noise — no full-Σ information is hidden)
#   - transitions    : "standard"           (stationary K×K transition matrix;
#                      geometric dwell-time, as agreed)
#   - init           : init_method="kmeans" (matches rSLDS k-means init)
#   - fit method     : "em"                 (exact EM for HMM; analogue of
#                      Laplace-EM for rSLDS)
#
# Return signature: (xhat, zhat, elbo, q, mdl) — same tuple as fit_rSLDS.
#
# Compatibility shim (so evaluate_*, and downstream pipeline
# work unchanged):
#   xhat := y  (HMM has no continuous latent)
#   mdl.emissions is synthesized from mdl.observations:
#       Cs       = zeros((K, N, N))        # D := N, Cs·x drops out
#       ds       = observations.mus        # (K, N)
#       inv_etas = log(1/sigmasq)          # (K, N),  == log(1/var_k)
#   most_likely_states(x, y) and expected_states(x, y, mask=None) are shimmed
#   to ignore x and delegate to the HMM's own Viterbi / forward-backward.
#   Causal inference is handled separately by inference_HMM() using native
#   HMM attributes (observations.mus, observations.sigmasq, transitions), not
#   this emissions shim.

def fit_HMM(y, params, n_iter_em=50, seed=None):
    """
    Fit a diagonal-Gaussian HMM via ssm, strict-parity drop-in for fit_rSLDS.

    params: dict(n_regimes, dim_latent, single_subspace)
        dim_latent and single_subspace are accepted for signature parity
        and ignored (HMM has no continuous latent).

    Returns: (xhat, zhat, elbo, q, mdl)
    """

    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape
    K = int(params["n_regimes"])

    # INSTANTIATE
    mdl = ssm.HMM(K, N,
                  observations="diagonal_gaussian",
                  transitions="standard")

    # FIT (ssm.HMM.fit auto-initializes when initialize=True; we pass
    # init_method="kmeans" to match fit_rSLDS's k-means initialization)
    # Patch sklearn KMeans to set n_init=10 explicitly (ssm omits it,
    # triggering a FutureWarning on newer sklearn). Preserve the original
    # __signature__ so sklearn's _get_param_names introspection still works.
    import sklearn.cluster as _skc
    import inspect as _inspect
    _kmeans_init_orig = _skc.KMeans.__init__
    _kmeans_init_sig  = _inspect.signature(_kmeans_init_orig)
    def _kmeans_init_patched(self, *args, **kwargs):
        kwargs.setdefault("n_init", 10)
        return _kmeans_init_orig(self, *args, **kwargs)
    _kmeans_init_patched.__signature__ = _kmeans_init_sig
    _skc.KMeans.__init__ = _kmeans_init_patched
    try:
        lls = mdl.fit(y,
                      method="em",
                      num_iters=n_iter_em,
                      initialize=True,
                      init_method="kmeans",
                      verbose=0)
    finally:
        _skc.KMeans.__init__ = _kmeans_init_orig
    
    elbo = lls  # HMM log-probabilities; named 'elbo' for pipeline symmetry

    # POSTERIORS
    zhat = mdl.most_likely_states(y)                        # (T,)
    xhat = y                                                 # identity

    # --- Synthesize emissions namespace from observations (no info loss) ---
    mus          = np.asarray(mdl.observations.mus,     dtype=float)   # (K, N)
    sigmasq      = np.asarray(mdl.observations.sigmasq, dtype=float)   # (K, N)
    inv_etas     = np.log(1.0 / np.clip(sigmasq, 1e-12, None))         # (K, N) == log(1/var_k)

    class _EmissionsShim: pass
    em = _EmissionsShim()
    em.Cs       = np.zeros((K, N, N))   # D := N, makes Cs·x drop out
    em.ds       = mus                    # (K, N)
    em.inv_etas = inv_etas               # (K, N)
    em.Fs       = []                     # no inputs
    mdl.emissions = em

    # --- Shim most_likely_states(x, y) to match SLDS call signature ---
    _hmm_mls = mdl.most_likely_states
    def _mls(x_or_y, y_=None, *a, **kw):
        data = y_ if y_ is not None else x_or_y
        return _hmm_mls(np.asarray(data))
    mdl.most_likely_states = _mls

    # --- Shim expected_states(x, y, mask=None) to match SLDS call sig ---
    _hmm_es = ssm.HMM.expected_states.__get__(mdl, type(mdl))
    def _es(x_or_y, y_=None, mask=None, *a, **kw):
        data = y_ if y_ is not None else x_or_y
        return _hmm_es(np.asarray(data))   # returns (Ez, Ezzp1, ll)
    mdl.expected_states = _es

    # --- Shim _make_variational_posterior so inference_rSLDS works ---
    class _QShim:
        def __init__(self, datas):
            self.mean_continuous_states = [np.asarray(d, dtype=float) for d in datas]
    def _mvp(variational_posterior=None, datas=None, inputs=None, masks=None,
             tags=None, method=None, **kw):
        return _QShim(datas)
    mdl._make_variational_posterior = _mvp

    q = _QShim([xhat])

    return xhat, zhat, elbo, q, mdl


# ---------------------------------------------------------------
# Fit AR-HMM (ssm) — strict-parity drop-in for fit_rSLDS / fit_HMM
# ---------------------------------------------------------------
#
# Parity choices (matched to fit_HMM / fit_rSLDS):
#   - observations   : "diagonal_ar"        (per-regime VAR(1) with diagonal
#                      innovation covariance: y_t = A_k y_{t-1} + b_k + eps,
#                      eps ~ N(0, diag(sig^2_k)). Diagonal matches rSLDS
#                      dynamics="diagonal_gaussian" and fit_HMM.)
#   - transitions    : "standard"           (stationary K×K; geometric dwell)
#   - init           : init_method="kmeans" (matches fit_rSLDS / fit_HMM)
#   - fit method     : "em"                 (exact EM)
#
# Return signature: (xhat, zhat, elbo, q, mdl) — same tuple as fit_rSLDS.
#
# Compatibility shim (so evaluate_*, and downstream pipeline
# work unchanged):
#   xhat := y  (AR-HMM has no continuous latent separate from y)
#   mdl.emissions is synthesized from mdl.observations so downstream SLDS-style
#   consumers see a uniform emissions-style namespace:
#       Cs       = As                      (K, N, N)  <-- VAR(1) matrix
#       ds       = bs                      (K, N)     <-- per-regime intercept
#       inv_etas = log(1 / sigmasq)        (K, N)     <-- diagonal innovation var
#     NO information loss (diagonal covariance by construction).
#   most_likely_states(x, y), expected_states(x, y, mask=...), and
#   _make_variational_posterior are shimmed to match the rSLDS call signatures.

def fit_AR_HMM(y, params, n_iter_em=50, seed=None):
    """
    Fit a diagonal-Gaussian AR-HMM via ssm, strict-parity drop-in for fit_rSLDS.

    Observation model (per regime k):
        y_t = A_k y_{t-1} + b_k + eps_t,   eps_t ~ N(0, diag(sigmasq_k))

    params: dict(n_regimes, dim_latent, single_subspace)
        dim_latent and single_subspace accepted for signature parity and ignored
        (AR-HMM has no continuous latent separate from y).

    Returns: (xhat, zhat, elbo, q, mdl)
    """

    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape
    K = int(params["n_regimes"])

    # INSTANTIATE
    mdl = ssm.HMM(K, N,
                  observations="diagonal_ar",
                  transitions="standard")

    # FIT
    # Patch sklearn KMeans to set n_init=10 explicitly (ssm omits it,
    # triggering a FutureWarning on newer sklearn). Preserve the original
    # __signature__ so sklearn's _get_param_names introspection still works.
    import sklearn.cluster as _skc
    import inspect as _inspect
    _kmeans_init_orig = _skc.KMeans.__init__
    _kmeans_init_sig  = _inspect.signature(_kmeans_init_orig)
    def _kmeans_init_patched(self, *args, **kwargs):
        kwargs.setdefault("n_init", 10)
        return _kmeans_init_orig(self, *args, **kwargs)
    _kmeans_init_patched.__signature__ = _kmeans_init_sig
    _skc.KMeans.__init__ = _kmeans_init_patched
    try:
        lls = mdl.fit(y,
                      method="em",
                      num_iters=n_iter_em,
                      initialize=True,
                      init_method="kmeans",
                      verbose=0)
    finally:
        _skc.KMeans.__init__ = _kmeans_init_orig
    
    elbo = lls

    # POSTERIORS
    zhat = mdl.most_likely_states(y)                        # (T,)
    xhat = y                                                 # identity

    # --- Synthesize emissions namespace from AR observations ---
    # ssm's AutoRegressiveDiagonalNoiseObservations exposes:
    #   .As       : (K, N, N)   VAR(1) matrices (lag-1)
    #   .bs       : (K, N)      intercepts
    #   .sigmasq  : (K, N)      diagonal innovation variances
    #
    # Pipeline expectation (emissions.Cs · x_{t-1} + emissions.ds with x=y):
    #   μ_k(t) = As[k] · y_{t-1} + bs[k]
    #   R_k    = diag(sigmasq_k)
    As_obs   = np.asarray(mdl.observations.As,      dtype=float)    # (K, N, N)
    bs_obs   = np.asarray(mdl.observations.bs,      dtype=float)    # (K, N)
    sigmasq  = np.asarray(mdl.observations.sigmasq, dtype=float)    # (K, N)
    inv_etas = np.log(1.0 / np.clip(sigmasq, 1e-12, None))          # (K, N)

    class _EmissionsShim: pass
    em = _EmissionsShim()
    em.Cs       = As_obs      # (K, N, N)  — regime-dependent "emission" matrix
    em.ds       = bs_obs      # (K, N)
    em.inv_etas = inv_etas    # (K, N)
    em.Fs       = []
    mdl.emissions = em

    # --- Shim most_likely_states(x, y) to ignore x ---
    _hmm_mls = mdl.most_likely_states
    def _mls(x_or_y, y_=None, *a, **kw):
        data = y_ if y_ is not None else x_or_y
        return _hmm_mls(np.asarray(data))
    mdl.most_likely_states = _mls

    # --- Shim expected_states(x, y, mask=None) ---
    _hmm_es = ssm.HMM.expected_states.__get__(mdl, type(mdl))
    def _es(x_or_y, y_=None, mask=None, *a, **kw):
        data = y_ if y_ is not None else x_or_y
        return _hmm_es(np.asarray(data))
    mdl.expected_states = _es

    # --- Shim _make_variational_posterior ---
    class _QShim:
        def __init__(self, datas):
            self.mean_continuous_states = [np.asarray(d, dtype=float) for d in datas]
    def _mvp(variational_posterior=None, datas=None, inputs=None, masks=None,
             tags=None, method=None, **kw):
        return _QShim(datas)
    mdl._make_variational_posterior = _mvp

    q = _QShim([xhat])

    return xhat, zhat, elbo, q, mdl


# ============================================================================
# fit_SLDS — plain switching LDS (HMM-style gate), unrestricted EM
#
# Identical structure to fit_rSLDS, three substantive changes:
#   1. transitions="standard"           (was "recurrent_only")
#   2. drop the Rs / r / Ws init lines  (StandardTransitions has none)
#   3. init log_Ps from k-means label transitions (Laplace-smoothed) so EM
#      starts from a data-informed gate instead of the ssm uniform default.
# ============================================================================

def fit_SLDS(y, params, n_iter_em=50, seed=None):
    """
    params: dict(n_regimes, dim_latent, single_subspace)
    """
    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)

    N = y.shape[1]
    D = params["dim_latent"]
    K = params["n_regimes"]
    single_subspace = params["single_subspace"]

    def safe_log_inv(var, lo=1e-6, hi=1e6):
        var = np.nan_to_num(var, nan=hi, posinf=hi, neginf=hi)
        var = np.clip(var, lo, hi)
        return np.log(1.0 / var)

    var = np.var(y, 0, keepdims=True)
    log_inv = safe_log_inv(var)

    if K is None:
        K, cluster_stats = fit_kmeans(y, Ks=[2, 3, 4], display=False)
        print(cluster_stats)

    mdl = ssm.SLDS(N, K, D,
                   transitions="standard",
                   dynamics="diagonal_gaussian",
                   emissions="gaussian",
                   single_subspace=single_subspace)

    if N < D * K or D >= N:
        # Same bypass as fit_rSLDS — ssm.SLDS.initialize fails when D*Keff > N.
        # The `D >= N` clause also covers single_subspace K=1 with N == D, where
        # ssm's PCA initializer sets noise_variance_ = 0 -> inv_etas = log(0).
        # See ssm_init_note.md. Identical machinery minus the recurrent-gate
        # (Rs / r / Ws) lines and plus a label-counts log_Ps initialiser.
        mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
        mdl.dynamics.mu_init = np.zeros((K, D))
        mdl.dynamics.sigmasq_init = np.ones((K, D))

        km = KMeans(n_clusters=K, n_init=10,
                    random_state=seed if seed is not None else 0)
        km.fit(y)
        order = np.argsort(km.cluster_centers_[:, 0])
        cluster_means = km.cluster_centers_[order]
        labels = km.labels_
        old_to_new = np.empty(K, dtype=int)
        for new_idx, old_idx in enumerate(order):
            old_to_new[old_idx] = new_idx
        labels_new = np.array([old_to_new[l] for l in labels], dtype=int)

        cluster_vars = np.zeros((K, N))
        cluster_ac1  = np.zeros((K, N))
        for old_idx in range(K):
            new_idx = old_to_new[old_idx]
            pts = y[labels == old_idx]
            if pts.shape[0] >= 2:
                cluster_vars[new_idx] = np.var(pts, axis=0)
            else:
                cluster_vars[new_idx] = np.var(y, axis=0)
            adj = (labels[:-1] == old_idx) & (labels[1:] == old_idx)
            if adj.sum() >= 2:
                y0 = y[:-1][adj]; y1 = y[1:][adj]
                mu = 0.5 * (y0.mean(axis=0) + y1.mean(axis=0))
                num = ((y0 - mu) * (y1 - mu)).sum(axis=0)
                den = ((y0 - mu) ** 2).sum(axis=0) + 1e-12
                cluster_ac1[new_idx] = np.clip(num / den, -0.999, 0.999)
            else:
                cluster_ac1[new_idx] = 0.0

        rho_init = np.zeros((K, D))
        rho_init[:, :N] = cluster_ac1
        cluster_means_D = np.zeros((K, D))
        cluster_means_D[:, :N] = cluster_means
        bs_init = (1.0 - rho_init) * cluster_means_D
        sigmasq_init = np.full((K, D), 1e-4)
        sigmasq_init[:, :N] = np.maximum(cluster_vars, 1e-6)
        As_init = np.zeros((K, D, D))
        for k in range(K):
            diag_k = rho_init[k] + 0.01 * npr.randn(D)
            np.fill_diagonal(As_init[k], np.clip(diag_k, -0.999, 0.999))
        mdl.dynamics.As = As_init
        mdl.dynamics.bs = bs_init
        mdl.dynamics.sigmasq = sigmasq_init

        # log_Ps from Laplace-smoothed k-means transition counts on labels_new.
        T = labels_new.size
        counts = np.zeros((K, K))
        for t in range(T - 1):
            counts[labels_new[t], labels_new[t + 1]] += 1
        counts += 1.0  # Laplace smoothing
        P_init = counts / counts.sum(axis=1, keepdims=True)
        mdl.transitions.log_Ps = np.log(P_init)

        if single_subspace:
            mdl.emissions.Cs = np.eye(N, D)[None, :, :]
            mdl.emissions.ds = np.zeros((1, N))
            mdl.emissions.inv_etas = log_inv
        else:
            mdl.emissions.Cs = np.tile(np.eye(N, D), (K, 1, 1))
            mdl.emissions.ds = np.zeros((K, N))
            mdl.emissions.inv_etas = np.tile(log_inv, (K, 1))
    else:
        mdl.initialize(y)

    # --- emission-variance floor (parity with fit_rSLDS; guards K=1 N==D PCA
    #     noise_variance_=0 -> inv_etas=log(0), and EM driving residual var ~0) ---
    _INV_ETAS_MIN = float(np.log(1e-4))
    mdl.emissions.inv_etas = np.maximum(
        np.nan_to_num(mdl.emissions.inv_etas, neginf=_INV_ETAS_MIN, nan=_INV_ETAS_MIN),
        _INV_ETAS_MIN)
    _emi_mstep_base_uns = mdl.emissions.m_step
    def _emi_mstep_floored(*a, **k):
        _emi_mstep_base_uns(*a, **k)
        mdl.emissions.inv_etas = np.maximum(mdl.emissions.inv_etas, _INV_ETAS_MIN)
    mdl.emissions.m_step = _emi_mstep_floored

    elbo, q = mdl.fit(y,
                      method="laplace_em",
                      variational_posterior="structured_meanfield",
                      num_iters=n_iter_em,
                      alpha=0.0,
                      initialize=False)

    xhat = q.mean_continuous_states[0]
    zhat = mdl.most_likely_states(xhat, y)

    return xhat, zhat, elbo, q, mdl


# ----------------------------------------------------------------------------
# fit_SLDS_restricted — EM-free closed-form, three-stage; standard transitions
#
# Identical to fit_rSLDS_restricted except stage 3:
#   - rSLDS_restricted: multinomial LR on next-step label -> Rs, r
#   - SLDS_restricted : Laplace-smoothed transition counts -> log_Ps
# ----------------------------------------------------------------------------

def fit_SLDS_restricted(y, params, C=None, d=None, n_iter_em=10, seed=None,
        b_pattern=None, enforce_diag_A=True,
        lam_dyn=None, lam_trn=None, delta=None, q_min=1e-6,
        C_mask=None, d_mask=None):
    """
    Restricted SLDS via EM-free closed-form estimation. Standard (HMM-style)
    transitions; gate is a homogeneous (K, K) Markov chain estimated by counts.

    Three-stage fit (training batch only; leak-free):
      1. Assignment : k-means on x_proxy = C^+(y - d) with [level, vol] feats.
      2. Dynamics   : per-regime (A_k, b_k, sigmasq_k) from within-cluster moments.
      3. Gate       : log_Ps = log(Laplace-smoothed transition counts of labels).

    Legacy EM knobs are accepted for signature compatibility and ignored.
    """
    if seed is not None:
        np.random.seed(seed)
        npr.seed(seed)

    y = np.asarray(y, dtype=float)
    T, N = y.shape
    K = int(params["n_regimes"])
    D = int(params["dim_latent"])
    assert bool(params.get("single_subspace", True)), "Require single_subspace=True."
    if b_pattern is None:
        b_pattern = ["mu_form"] * D
    assert len(b_pattern) == D and all(
        m in {"free", "zero", "mu_form"} for m in b_pattern)

    fixed_emissions = (C is not None) and (d is not None)
    assert fixed_emissions, "fit_SLDS_restricted requires fixed emissions C, d."
    C = np.asarray(C, dtype=float)
    d = np.asarray(d, dtype=float)
    assert C.shape == (N, D), f"C must be (N,D)=({N},{D}), got {C.shape}"
    assert d.shape == (N,),   f"d must be (N,), got {d.shape}"

    # ----- model (standard SLDS; HMM gate used at inference)
    mdl = ssm.SLDS(
        N, K, D,
        transitions="standard",
        dynamics="diagonal_gaussian",
        emissions="gaussian",
        single_subspace=True,)

    # learn the free factor loadings closed-form (fund/factor2: mask all-zero
    # -> C returned unchanged, stays locked). Fixes the all-zero return loading.
    C = _estimate_masked_loadings(C, d, y, C_mask)

    obs_var = np.var(y, axis=0)
    obs_var = np.clip(np.nan_to_num(obs_var, nan=1.0, posinf=1e6, neginf=1e6), 1e-8, 1e6)
    inv_etas = np.tile(np.log(1.0 / obs_var)[None, :], (K, 1))
    mdl.emissions.Cs = C[None, :, :]
    mdl.emissions.ds = d[None, :]
    mdl.emissions.inv_etas = inv_etas

    # latent proxy x = C^+(y - d)
    U_svd, s_svd, Vt_svd = np.linalg.svd(C, full_matrices=False)
    s_inv = np.where(s_svd > 1e-8, 1.0 / s_svd, 0.0)
    C_pinv = (Vt_svd.T * s_inv) @ U_svd.T
    x_proxy = (y - d) @ C_pinv.T

    # ----- stage 1: k-means on [level, vol] features (identical to rSLDS_restricted)
    vol_feat = np.zeros((T, D))
    vol_feat[1:] = (x_proxy[1:] - x_proxy[:-1]) ** 2
    vol_feat[0] = vol_feat[1] if T > 1 else 0.0
    feat = np.concatenate([x_proxy, vol_feat], axis=1)
    feat_mean = feat.mean(axis=0, keepdims=True)
    feat_std = feat.std(axis=0, keepdims=True)
    feat_std = np.where(feat_std > 1e-12, feat_std, 1.0)
    feat_z = (feat - feat_mean) / feat_std

    km = KMeans(n_clusters=K, n_init=10,
                random_state=seed if seed is not None else 0)
    km.fit(feat_z)
    level_centroid_z = km.cluster_centers_[:, :D]
    level_centroid = level_centroid_z * feat_std[:, :D] + feat_mean[:, :D]
    order = np.argsort(level_centroid[:, 0])
    old_to_new = np.empty(K, dtype=int)
    for new_idx, old_idx in enumerate(order):
        old_to_new[old_idx] = new_idx
    labels = np.array([old_to_new[l] for l in km.labels_], dtype=int)

    cluster_means = np.zeros((K, D))
    for k in range(K):
        pts = x_proxy[labels == k]
        cluster_means[k] = pts.mean(axis=0) if pts.shape[0] > 0 else x_proxy.mean(axis=0)

    # ----- stage 2: per-regime dynamics (identical to rSLDS_restricted)
    cluster_vars = np.zeros((K, D))
    cluster_ac1  = np.zeros((K, D))
    for k in range(K):
        pts = x_proxy[labels == k]
        cluster_vars[k] = np.var(pts, axis=0) if pts.shape[0] >= 2 else np.var(x_proxy, axis=0)
        adj = (labels[:-1] == k) & (labels[1:] == k)
        if adj.sum() >= 2:
            x0 = x_proxy[:-1][adj]; x1 = x_proxy[1:][adj]
            mu = 0.5 * (x0.mean(axis=0) + x1.mean(axis=0))
            num = ((x0 - mu) * (x1 - mu)).sum(axis=0)
            den = ((x0 - mu) ** 2).sum(axis=0) + 1e-12
            cluster_ac1[k] = np.clip(num / den, -0.999, 0.999)
    rho = cluster_ac1
    bs  = (1.0 - rho) * cluster_means
    for d_idx, mode in enumerate(b_pattern):
        if mode == "zero":
            bs[:, d_idx] = 0.0
    sigmasq = np.maximum(cluster_vars, q_min)
    As = np.zeros((K, D, D))
    for k in range(K):
        np.fill_diagonal(As[k], np.clip(rho[k], -0.999, 0.999))
    mdl.dynamics.As = As
    mdl.dynamics.bs = bs
    mdl.dynamics.sigmasq = sigmasq
    mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
    denom = np.clip(1.0 - rho, 1e-8, None)
    mdl.dynamics_mu_param = bs / denom

    # ----- stage 3: homogeneous Markov chain from label-transition counts.
    # P_hat[j, k] = #(z_{t-1}=j AND z_t=k) / #(z_{t-1}=j). Additive Laplace
    # smoothing (+1) so unseen transitions remain finite under log and short
    # batches with class imbalance don't generate -inf log_Ps cells.
    counts = np.zeros((K, K))
    for t in range(T - 1):
        counts[labels[t], labels[t + 1]] += 1
    counts += 1.0
    P_hat = counts / counts.sum(axis=1, keepdims=True)
    mdl.transitions.log_Ps = np.log(P_hat)

    # ----- outputs (gate-consistent Viterbi)
    xhat = x_proxy
    zhat = mdl.most_likely_states(xhat, y)

    # data log-likelihood of the Viterbi path (finite scalar; no EM trace).
    x_mask = np.ones_like(xhat, dtype=bool)
    y_mask = np.ones_like(y, dtype=bool)
    log_pi0 = mdl.init_state_distn.log_pi0 - logsumexp(mdl.init_state_distn.log_pi0)
    log_Ps = mdl.transitions.log_transition_matrices(
        xhat, np.zeros((T, 0)), x_mask, None)  # standard transitions -> (1, K, K)
    ll_dyn = mdl.dynamics.log_likelihoods(xhat, np.zeros((T, 0)), x_mask, None)
    ll_emi = mdl.emissions.log_likelihoods(y, np.zeros((T, 0)), y_mask, None, xhat)
    ll_obs = ll_dyn + ll_emi
    path_ll = float(log_pi0[zhat[0]] + ll_obs[0, zhat[0]])
    for t in range(1, T):
        ti = t - 1 if log_Ps.shape[0] > 1 else 0
        path_ll += float(log_Ps[ti, zhat[t - 1], zhat[t]] + ll_obs[t, zhat[t]])
    elbo = np.array([path_ll], dtype=float)

    class _QShim:
        def __init__(self, datas):
            self.mean_continuous_states = [np.asarray(dd, dtype=float) for dd in datas]
    q = _QShim([xhat])

    return xhat, zhat, elbo, q, mdl

