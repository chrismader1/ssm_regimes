# fit.py

import numpy as np
import numpy.random as npr
import ssm
from sklearn.cluster import KMeans

from .init import fit_kmeans


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

    if N < D * K:

        # initialize(y) carries out PCA, which fails when trying to extract D components 
        # from N-dim data, but N < D.
        # print(f"N<D*K: initializing manually")
        
        # discrete-state prior π
        # mdl.init_state_dist = np.full(K, 1.0 / K) 
        mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
 
        # continuous-state prior 𝒩(μ_init, diag(σ^2_init))
        mdl.dynamics.mu_init = np.zeros((K, D))
        mdl.dynamics.sigmasq_init = np.ones((K, D))

        # recurrent transition weights R, r
        mdl.transitions.Rs = 0.01 * np.random.randn(K, D)
        mdl.transitions.r = np.zeros(K)

        # linear dynamics A, b, σ^2  (stable ≈ identity)
        mdl.dynamics.As = 0.95 * np.repeat(np.eye(D)[None, :, :], K, axis=0)
        mdl.dynamics.bs = np.zeros((K, D))
        mdl.dynamics.sigmasq = 1e-4 * np.ones((K, D))

        # eps = 1e-6
        # obs_var = np.var(y, 0)
        # padded_var = np.pad(obs_var, (0, D - N), mode='constant', constant_values=obs_var.mean())
        # sigmasq_init = np.maximum(0.1 * padded_var, eps)
        # mdl.dynamics.sigmasq = np.tile(sigmasq_init, (K, 1))

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
    lam_dyn=None,            # dynamics repulsion (around mean); None -> from params or 0
    lam_trn=None,            # transitions repulsion
    delta=None,              # hard min separation radius
    q_min=1e-6,              # floor on dynamics variance
    C_mask=None, d_mask=None):
 
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
        * Optional dynamics repulsion (lam_dyn) and/or hard min separation (delta)
        * Optional enforce_diag_A (zero off-diagonals of A_k); clip |rho_k| < 1 on diagonal
        * Optional b_pattern handling with µ bookkeeping (µ = b/(1-ρ))
        * Emissions projection if fixed_emissions (respects C_mask, d_mask)
    - Final E-only pass
    - Returns (xhat, zhat, elbo_trace, q_last, mdl)
 
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
    if lam_dyn is None:
        lam_dyn = float(params.get("repulsion_strength_dynamics", 0.0))
    if lam_trn is None:
        lam_trn = float(params.get("repulsion_strength_transitions", 0.0))
    if delta is None:
        delta = float(params.get("min_separation", 0.0))
    if b_pattern is None:
        b_pattern = ["mu_form"] * D
    assert len(b_pattern) == D and all(m in {"free", "zero", "mu_form"} for m in b_pattern)
 
    # ----- model
    mdl = ssm.SLDS(
        N, K, D,
        transitions="recurrent_only",       # stick-breaking recurrent logistic gating
        dynamics="diagonal_gaussian",
        emissions="gaussian",
        single_subspace=True,)
 
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

    # ----- initialisation: parity with fit_rSLDS (unrestricted)
    # Original fit_rSLDS_restricted set ALL K regimes to identical dynamics
    # (As = 0.95*I, bs = 0, sigmasq = 1) with only a tiny 0.01-stddev random
    # perturbation on transitions.Rs. With identical per-regime dynamics, EM
    # converges to a symmetric attractor where regimes collapse to copies of
    # each other — explains "same poor results regardless of K" symptom.
    #
    # Fix: use the same init path as fit_rSLDS — call mdl.initialize(y) which
    # runs k-means on y to derive distinct per-regime dynamics, with the same
    # N<D*K fallback to manual init when PCA can't extract D components.
    # After init, re-pin emissions to the fixed (C, d) since initialize(y)
    # would have overwritten them.
    if fixed_emissions and N < D * K:
        # k-means init can't run (PCA needs N >= D); use manual per-regime
        # init that breaks symmetry across K via random perturbations.
        mdl.init_state_distn.log_pi0 = np.log(np.full(K, 1.0 / K))
        mdl.dynamics.mu_init     = np.zeros((K, D))
        mdl.dynamics.sigmasq_init = np.ones((K, D))
        # Per-regime random perturbation of dynamics so regimes are NOT
        # identical at init. Spread diagonals across [0.85, 0.99] then add
        # small random noise.
        diag_seed = np.linspace(0.85, 0.99, K)
        As_init = np.zeros((K, D, D))
        for k in range(K):
            As_init[k] = np.diag(np.full(D, diag_seed[k])) + 0.01 * npr.randn(D, D)
            # keep diagonal-dominant, clip diag for stability
            np.fill_diagonal(As_init[k], np.clip(np.diag(As_init[k]), -0.999, 0.999))
        mdl.dynamics.As     = As_init
        mdl.dynamics.bs      = 0.01 * npr.randn(K, D)        # small random b per regime
        mdl.dynamics.sigmasq = 1e-4 * np.ones((K, D))
        mdl.transitions.Rs   = 0.1 * npr.randn(K, D)         # larger perturbation than 0.01
        mdl.transitions.r    = np.zeros(K)
    else:
        # k-means based init via ssm's built-in
        mdl.initialize(y)

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

    # store µ for reporting (µ = b/(1-ρ))
    mdl.dynamics_mu_param = np.zeros((K, D))
 
    # ----- helpers: bind m_step, freeze/enable, repulsion, min-sep, constraints
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
 
    def _enforce_min_sep_matrix(V, delta_):
        if delta_ <= 0 or K <= 1:
            return V
        for i in range(K):
            for j in range(i+1, K):
                diff = V[i] - V[j]
                nrm = float(np.linalg.norm(diff))
                if nrm < delta_:
                    if nrm < 1e-12:
                        dvec = npr.randn(V.shape[1]); dvec /= (np.linalg.norm(dvec) + 1e-12)
                        V[i] += 0.5 * delta_ * dvec
                        V[j] -= 0.5 * delta_ * dvec
                    else:
                        u = diff / nrm
                        push = 0.5 * (delta_ - nrm) * u
                        V[i] += push
                        V[j] -= push
        return V
 
    def dyn_mstep_with_repulsion(*args, **kwargs):
        dyn_mstep_base(*args, **kwargs)     # ssm closed-form update
        # repulsion around mean + min-sep on [vec(A_k); b_k]
        # FIX (BUG 3): gate lam_dyn and delta independently so min-sep applies
        # even when repulsion is off.
        A = mdl.dynamics.As; B = mdl.dynamics.bs
        if K > 1 and (lam_dyn > 0.0 or delta > 0.0):
            if lam_dyn > 0.0:
                A_bar = A.mean(axis=0, keepdims=True); B_bar = B.mean(axis=0, keepdims=True)
                A = A + lam_dyn * (A - A_bar)
                B = B + lam_dyn * (B - B_bar)
            if delta > 0.0:
                V = np.concatenate([A.reshape(K, -1), B], axis=1)
                V = _enforce_min_sep_matrix(V, delta)
                A = V[:, :D*D].reshape(K, D, D); B = V[:, D*D:]
        mdl.dynamics.As = A; mdl.dynamics.bs = B
        # floor process variances
        if hasattr(mdl.dynamics, "sigmasq"):
            mdl.dynamics.sigmasq = np.maximum(mdl.dynamics.sigmasq, q_min)
 
    def trn_mstep_with_repulsion(*args, **kwargs):
        if K == 1:
            return
        trn_mstep_base(*args, **kwargs)     # ssm optimization over stick-breaking gates
        if lam_trn > 0.0:
            if hasattr(mdl.transitions, "Rs"):
                R = mdl.transitions.Rs
                Rm = R.mean(axis=0, keepdims=True)
                mdl.transitions.Rs = R + lam_trn * (R - Rm)
            if hasattr(mdl.transitions, "r"):
                r = mdl.transitions.r
                rm = r.mean(axis=0, keepdims=True)
                mdl.transitions.r = r + lam_trn * (r - rm)
        # min-sep on [R_k, r_k] (gated on delta, independent of lam_trn)
        if delta > 0.0 and hasattr(mdl.transitions, "Rs") and hasattr(mdl.transitions, "r"):
            VT = np.concatenate([mdl.transitions.Rs, mdl.transitions.r[:, None]], axis=1)
            VT = _enforce_min_sep_matrix(VT, delta)
            mdl.transitions.Rs = VT[:, :D]
            mdl.transitions.r = VT[:, -1]
 
    def _enable_M_pass():
        mdl.dynamics.m_step = dyn_mstep_with_repulsion
        mdl.transitions.m_step = trn_mstep_with_repulsion
        mdl.init_state_distn.m_step = pio_mstep_base
        # emissions: only if unrestricted
        mdl.emissions.m_step = (lambda *_, **__: None) if fixed_emissions else emi_mstep_base
 
    def _enforce_identifiability_and_mu():
        # FIX (BUG 7): honor enforce_diag_A. If False, keep off-diagonals; only
        # clip the diagonal (stability proxy).
        A = mdl.dynamics.As
        for k in range(K):
            if enforce_diag_A:
                diag = np.clip(np.diag(A[k]), -0.999, 0.999)
                A[k] = np.diag(diag)
            else:
                di = np.clip(np.diag(A[k]), -0.999, 0.999)
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
 
        # FIX (BUG 9): do NOT re-lock emissions here. _project_emissions_ is the
        # single authority and will be called immediately after when fixed_emissions.
 
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
 
 
    # ----- outer split-EM (alpha=0.0)
    elbo_trace = []
    q_last = None
 
    for _ in range(int(n_iter_em)):
 
        # E-only — with deterministic-perturbation retry on AssertionError
        # from ssm/lds.py (NaN in expected log-prob during Newton step).
        # Cause: regime collapse to zero responsibility under fixed-C
        # restricted emissions. Fix: perturb dynamics.bs deterministically
        # (seeded from iter index) and re-run the same iteration.
        # Reproducible because perturbation is deterministic in (iter, attempt).
        _freeze_all()
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                elbo_E, q = mdl.fit(
                    y,
                    method="laplace_em",
                    variational_posterior="structured_meanfield",
                    num_iters=1,
                    alpha=0.0,
                    initialize=False)
                break
            except AssertionError:
                if attempt == max_attempts - 1:
                    raise
                # Deterministic perturbation: seeded numpy state from
                # (iter, attempt) so reruns yield identical bs perturbation.
                rng = np.random.RandomState(seed=hash(("E", _, attempt)) % (2**31 - 1))
                mdl.dynamics.bs = mdl.dynamics.bs + 1e-3 * rng.randn(*mdl.dynamics.bs.shape)
        elbo_trace.extend(list(elbo_E))
    
        # M-enabled — same retry pattern.
        _enable_M_pass()
        for attempt in range(max_attempts):
            try:
                elbo_M, q = mdl.fit(
                    y,
                    method="laplace_em",
                    variational_posterior="structured_meanfield",
                    num_iters=1,
                    alpha=0.0,
                    initialize=False)
                break
            except AssertionError:
                if attempt == max_attempts - 1:
                    raise
                rng = np.random.RandomState(seed=hash(("M", _, attempt)) % (2**31 - 1))
                mdl.dynamics.bs = mdl.dynamics.bs + 1e-3 * rng.randn(*mdl.dynamics.bs.shape)
        elbo_trace.extend(list(elbo_M))
        q_last = q

        # identifiability constraints (after M)
        _enforce_identifiability_and_mu()
        # FIX (BUG 1): only project when emissions are restricted; otherwise
        # C_fix/d_fix are None and np.asarray would crash.
        if fixed_emissions:
            _project_emissions_(mdl.emissions, C_fix=C, d_fix=d, C_mask=C_mask, d_mask=d_mask)
 
    # Final E-only (sharpen posterior)
    _freeze_all()
    elbo_F, q_last = mdl.fit(
        y,
        method="laplace_em",
        variational_posterior="structured_meanfield",
        num_iters=1,
        alpha=0.0,
        initialize=False)
    elbo_trace.extend(list(elbo_F))
 
    # outputs
    xhat = q_last.mean_continuous_states[0]
    zhat = mdl.most_likely_states(xhat, y)
    return xhat, zhat, np.asarray(elbo_trace, dtype=float), q_last, mdl


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
# Compatibility shim (so evaluate_*, cusum_overlay, and downstream pipeline
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
# Compatibility shim (so evaluate_*, cusum_overlay, and downstream pipeline
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

