# inference.py

import numpy as np
from scipy.special import logsumexp
from ._utils import _lse, _mvn_logpdf
from .bounds import max_cpll_causal_bound


# ---------------------------------------------------------------
# rSLDS Inference
# ---------------------------------------------------------------

def inference_rSLDS(px, mdl, y, T_train, cadence, dt=1/252, display=False):
    
    """
    OOS inference for a fitted rSLDS in one of two modes.

    inference_mode="online" (default for deployable backtests):
      Strictly causal cadence-stepped expanding-window smoothed inference.
      At each decision point t_d, smooth on y[0:t_d] and broadcast the
      rightmost label across the next `cadence` days. No look-ahead.

    inference_mode="smoothed" (for benchmarking / model comparison):
      Single structured-meanfield smoothed inference on the OOS block
      y[T_train:T]. Each OOS day's label uses past AND future data within
      the OOS block — leaks future information. Use only for comparing model
      classes (rSLDS vs HMM) under identical evaluation protocol; do not
      interpret resulting CAGR as live-trading return.

    Inference pattern (online mode):
      1. Training portion (t = 0 .. T_train - 1):
         single smoothed inference call on y[0:T_train]. Training labels are
         in-sample by design — smoothing here is fine.
      2. OOS portion (t = T_train .. T - 1):
         walk forward in steps of `cadence`. At each decision point t_d, run
         smoothed inference on y[0:t_d] and take the rightmost result
         (xhat[t_d - 1], zhat[t_d - 1], gamma[t_d - 1]) — strictly causal,
         only past data observed. Broadcast that decision across the next
         `cadence` days (or up to T - 1).
      3. CPLL (diagnostic, look-ahead-acceptable):
         single smoothed inference on the full y[0:T] sequence at the end,
         used only for the cpll number (matches old rSLDS.py behaviour).

    Special case: if T_train >= T or T_train + cadence > T (i.e. there is no
    OOS portion to expand), the function performs a single smoothed inference
    on the full y[0:T] and returns those quantities directly. This is the
    clean path for the synthetic pipelines and the post-loop CPLL call in
    gridsearch_actual.

    Parameters
    ----------
    px : pd.Series
        Price series (only its length is used; passed for parity with other
        inference signatures).
    mdl : fitted ssm.SLDS
    y : np.ndarray, shape (T, N) or (T,)
    T_train : int
        Length of the training portion. Indices [0, T_train) are treated as
        in-sample. Indices [T_train, T) are OOS and inferred causally via
        expanding windows.
    cadence : int
        Number of OOS days per expanding-window decision point. cadence=1
        infers each OOS day from its own expanding window. cadence=5 infers
        every 5th OOS day and broadcasts forward.
    dt, display : kept for parity with old signature; unused here.

    Returns
    -------
    {xhat, zhat, gamma, cpll, max_cpll, mdl}
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape
    T_train = int(T_train)
    cadence = int(cadence)
    assert cadence >= 1, f"cadence must be >= 1, got {cadence}"
    assert 0 <= T_train <= T, f"T_train={T_train} out of range for T={T}"

    K = int(mdl.K)
    D = int(mdl.D)

    # --- helper: single smoothed inference on y_window, return rightmost-bar quantities + full gamma ---
    def _smooth(y_window):
        Tw = y_window.shape[0]
        Fs = getattr(mdl.emissions, "Fs", [])
        D_in = Fs[0].shape[1] if len(Fs) else 0
        inputs = np.zeros((Tw, D_in))
        mask = np.ones_like(y_window, dtype=bool)
        q = mdl._make_variational_posterior(
            variational_posterior="structured_meanfield",
            datas=[y_window], inputs=[inputs], masks=[mask], tags=[None],
            method="smf",
        )
        x_smooth = q.mean_continuous_states[0]                  # (Tw, D)
        z_smooth = mdl.most_likely_states(x_smooth, y_window)   # (Tw,)
        g_smooth, *_ = mdl.expected_states(x_smooth, y_window, mask=mask)  # (Tw, K)
        return x_smooth, z_smooth, g_smooth

    # --- short-circuit: no OOS expansion needed (T_train covers everything, or no room for one cadence step) ---
    if T_train >= T or (T_train + cadence) > T:
        x_full, z_full, g_full = _smooth(y)
        max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)
        # true causal one-step predictive log-likelihood.
        cpll, cpll_oos = causal_cpll_rSLDS(mdl, y, T_split=T_train)
        return {"xhat": x_full, "zhat": z_full.astype(int), "gamma": g_full,
                "cpll": float(cpll), "cpll_oos": float(cpll_oos),
                "max_cpll": float(max_cpll), "mdl": mdl}

    # --- training portion: one smoothed call on y[0:T_train] ---
    xhat  = np.zeros((T, D))
    zhat  = np.zeros(T,  dtype=int)
    gamma = np.zeros((T, K))

    if T_train > 0:
        x_tr, z_tr, g_tr = _smooth(y[:T_train])
        xhat[:T_train]  = x_tr
        zhat[:T_train]  = z_tr.astype(int)
        gamma[:T_train] = g_tr

    # --- OOS portion: expanding-window smoothed inference at each decision point ---
    # Decision points: t_d = T_train + cadence, T_train + 2*cadence, ...
    # At t_d we run _smooth(y[0:t_d]) and use the result at index (t_d - 1) to
    # populate days [t_d - cadence, ..., t_d - 1] (the cadence days BEFORE t_d).
    # Then on the final segment we run one extra smoothed call to fill any tail.

    # Walk forward across OOS in cadence-sized blocks.
    # Block i ends at decision point t_d_i = T_train + (i+1)*cadence.
    # The block covers days [T_train + i*cadence, T_train + (i+1)*cadence) — its
    # decision is taken at the start of that block (with knowledge of data up to
    # that start), and the resulting weight is held for `cadence` days.
    #
    # Convention: decision at block start uses _smooth(y[0:block_start])
    # — strictly causal, no future leakage.
    block_start = T_train
    while block_start < T:
        block_end = min(block_start + cadence, T)
        # Decision at block_start: smooth on y[0:block_start], take rightmost result.
        # block_start guaranteed >= 1 here since T_train >= 0 and we guarded T_train >= T above.
        x_d, z_d, g_d = _smooth(y[:block_start])
        # rightmost label at index block_start - 1 of the smoothed window
        x_decision = x_d[-1]
        z_decision = int(z_d[-1])
        g_decision = g_d[-1]
        # broadcast across [block_start, block_end)
        xhat[block_start:block_end]  = x_decision  # broadcast row across rows
        zhat[block_start:block_end]  = z_decision
        gamma[block_start:block_end] = g_decision
        block_start = block_end

    # --- CPLL (true causal one-step predictive log-likelihood) ---
    cpll, cpll_oos = causal_cpll_rSLDS(mdl, y, T_split=T_train)
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)

    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "cpll_oos": float(cpll_oos),
            "max_cpll": float(max_cpll), "mdl": mdl}


def causal_cpll_rSLDS(mdl, y, T_split):
    """
    Causal one-step predictive log-likelihood for the recurrent-only,
    diagonal-Gaussian-dynamics, Gaussian-emission SLDS built in fit.py
    (transitions='recurrent_only', dynamics='diagonal_gaussian',
    emissions='gaussian'). Look-ahead-free GPB1 switching filter: the
    predictive density of y_t uses only data up to t-1.

    Returns (cpll, cpll_oos), summed over t = 1 .. T-1 and t = T_split .. T-1.
    Same units/scale as inference_HMM / inference_ARHMM cpll (observation-space
    one-step predictive log density of y_t).

    -------------------------------------------------------------------------
    NUMERICAL REDESIGN (vs. the original) — what changed and why
    -------------------------------------------------------------------------
    The original blew up to +/-50..320 nats for free-C / identity-C fits while
    staying sane for structured (factor1_vix) C. Root causes and fixes:

    1. INIT.  Original: P0 = pinv(C0) @ R0 @ pinv(C0).T. For an ill-conditioned
       or wide C0 (free / identity emissions) pinv(C0) is huge, so P0 — and
       hence the predictive covariance S = C P C' + R — explodes from t=0.
       For structured C0 it happened to be fine, which is the family split.
       Fix: P0 is a BOUNDED, dynamics-based prior — the per-latent stationary
       variance Qd/(1-rho^2) (clipped for |rho|>=1), taken as the median over
       regimes. This is principled (it is the model's own stationary spread)
       and cannot be inflated by a poorly-conditioned emission. The latent mean
       is still seeded from y_0 by least squares, but with an explicit rcond so
       a near-singular C0 cannot send m0 to infinity.

    2. PREDICTIVE DENSITY.  Robust Cholesky with adaptive jitter and a log-det
       from the Cholesky factor — no explicit inverse, no NaN/Inf from a
       singular S. A genuinely huge S still yields a correctly very-negative
       log density (honest: that fit really did predict badly).

    3. KALMAN UPDATE.  Gain via solve(), not inv(); covariances symmetrized.

    4. OVERFLOW GUARD ONLY.  The collapsed P has its eigenvalues capped at a
       large finite ceiling (P_CEIL). This is NOT a fit rescue and NOT result
       flattering: it engages only once the variance is already astronomically
       large (a divergent fit), and merely keeps the score a large-but-finite
       very-negative number instead of Inf/NaN that would poison the sum. A
       model that diverges still scores as badly as it should.

    The structured (factor1_vix) family — already stable — is unchanged by
    these guards (its P never approaches the ceiling, its S is well-conditioned);
    the guards only stop the free/identity families from returning garbage.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape
    K = int(mdl.K)
    D = int(mdl.D)

    As   = np.asarray(mdl.dynamics.As, dtype=float)        # (K, D, D)
    bs   = np.asarray(mdl.dynamics.bs, dtype=float)        # (K, D)
    Qd   = np.clip(np.asarray(mdl.dynamics.sigmasq, dtype=float), 1e-12, None)  # (K, D)
    Rs   = np.asarray(mdl.transitions.Rs, dtype=float)     # (K, D)
    rvec = np.asarray(mdl.transitions.r, dtype=float)      # (K,)
    Cs   = np.asarray(mdl.emissions.Cs, dtype=float)       # (Kc, N, D)
    ds   = np.asarray(mdl.emissions.ds, dtype=float)       # (Kc, N)
    invE = np.asarray(mdl.emissions.inv_etas, dtype=float) # (Ke, N)

    def _erow(arr, k):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr
        return arr[0] if arr.shape[0] == 1 else arr[k]

    # ---- numerical constants ----
    P_CEIL = 1e10          # overflow guard on latent-covariance eigenvalues
    P_FLOOR = 1e-12        # keep covariances strictly PD
    JIT0   = 1e-9          # base jitter for the predictive covariance
    eyeD = np.eye(D)
    eyeN = np.eye(N)
    LOG2PI = np.log(2.0 * np.pi)

    def _safe_mvn_logpdf(x, mean, cov):
        """log N(x | mean, cov) via jittered Cholesky. Honest large-negative
        for a huge/degenerate cov; never NaN/Inf."""
        S = 0.5 * (cov + cov.T)
        scale = max(1.0, np.trace(S) / max(N, 1))
        jit = JIT0 * scale
        L = None
        for _ in range(8):
            try:
                L = np.linalg.cholesky(S + jit * eyeN)
                break
            except np.linalg.LinAlgError:
                jit *= 10.0
        if L is None:
            return -1e12  # fully degenerate; honest floor, finite
        diff = x - mean
        z = np.linalg.solve(L, diff)              # L z = diff  -> z = L^{-1} diff
        logdet = 2.0 * np.sum(np.log(np.diag(L)))
        return -0.5 * (N * LOG2PI + logdet + float(z @ z))

    def _cap_cov(P):
        """Symmetrize and cap eigenvalues into [P_FLOOR, P_CEIL]."""
        P = 0.5 * (P + P.T)
        ev, V = np.linalg.eigh(P)
        ev = np.clip(ev, P_FLOOR, P_CEIL)
        return (V * ev) @ V.T

    # ---- bounded, dynamics-based prior over x_0 (NOT pinv(C0)-propagated) ----
    rho_diag = np.stack([np.clip(np.abs(np.diag(As[k])), 0.0, 0.999)
                         for k in range(K)])               # (K, D)
    stat_var = Qd / (1.0 - rho_diag ** 2)                  # (K, D) stationary spread
    P0_diag = np.clip(np.median(stat_var, axis=0), P_FLOOR, P_CEIL)
    P = np.diag(P0_diag)

    C0 = Cs[0]
    d0 = ds[0]
    m, *_ = np.linalg.lstsq(C0, y[0] - d0, rcond=1e-6)     # guarded least-squares seed

    cpll = 0.0
    cpll_oos = 0.0
    for t in range(1, T):
        # gate at the filtered mean (point approximation; gate nonlinear in x)
        log_pz = Rs @ m + rvec
        log_pz = log_pz - logsumexp(log_pz)

        comp_ll = np.empty(K)
        m_upd = np.empty((K, D))
        P_upd = np.empty((K, D, D))
        for k in range(K):
            mpk = As[k] @ m + bs[k]
            Ppk = As[k] @ P @ As[k].T + np.diag(Qd[k])
            Ppk = 0.5 * (Ppk + Ppk.T)
            Ck = _erow(Cs, k)
            dk = _erow(ds, k)
            Rk = np.diag(np.exp(-_erow(invE, k)))
            yhat = Ck @ mpk + dk
            S = Ck @ Ppk @ Ck.T + Rk
            comp_ll[k] = _safe_mvn_logpdf(y[t], yhat, S)
            # Kalman gain Kg = Ppk Ck^T S^{-1}, via solve on the symmetrized S
            Ssym = 0.5 * (S + S.T) + 1e-10 * eyeN
            Kg = np.linalg.solve(Ssym, Ck @ Ppk).T          # (D, N)
            m_upd[k] = mpk + Kg @ (y[t] - yhat)
            Pk = (eyeD - Kg @ Ck) @ Ppk
            P_upd[k] = 0.5 * (Pk + Pk.T)

        joint = log_pz + comp_ll
        denom = logsumexp(joint)
        cpll += float(denom)
        if t >= T_split:
            cpll_oos += float(denom)

        w = np.exp(joint - denom)
        m_new = np.sum(w[:, None] * m_upd, axis=0)
        P_new = np.zeros((D, D))
        for k in range(K):
            dmk = (m_upd[k] - m_new)[:, None]
            P_new += w[k] * (P_upd[k] + dmk @ dmk.T)
        P = _cap_cov(P_new)           # symmetrize + overflow guard
        m = m_new

    return cpll, cpll_oos


def inference_HMM(px, mdl, y, T_split, dt=1/252, display=False):
    """
    Causal inference for a fitted diagonal-Gaussian HMM.

    Strictly forward-only filter (alphas normalised per step).
    Uses the learned per-regime means (mus), variances (sigmasq), and
    stationary transition matrix. Initial state distribution pi0 from log_pi0.

    Returns {xhat, zhat, gamma, cpll, max_cpll, mdl} with xhat == y (identity).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape

    # --- learned params ---
    K = int(mdl.K)
    mus    = np.asarray(mdl.observations.mus,     dtype=float)  # (K, N)
    sigma2 = np.clip(np.asarray(mdl.observations.sigmasq, dtype=float), 1e-12, None)  # (K, N)
    P      = np.asarray(mdl.transitions.transition_matrix, dtype=float)  # (K, K)
    log_pi0 = np.asarray(mdl.init_state_distn.log_pi0, dtype=float)
    log_pi0 = log_pi0 - _lse(log_pi0)

    # --- per-timestep emission log-likelihoods log p(y_t | z_t=k) ---
    #     (diagonal Gaussian)
    #     log N(y_t | mus[k], diag(sigma2[k]))
    # shape: (T, K)
    def _log_emit(yt):
        diffs = yt[None, :] - mus                              # (K, N)
        return -0.5 * np.sum(diffs**2 / sigma2 + np.log(2.0 * np.pi * sigma2), axis=1)

    # --- causal forward filter ---
    # gamma[t, k] = p(z_t = k | y[0:t+1])
    # cpll: Σ_{t=1}^{T-1} log p(y_t | y[0:t]) = Σ log Σ_k pred[t, k] * L[t, k]
    gamma = np.zeros((T, K))
    cpll  = 0.0
    cpll_oos = 0.0

    # t = 0
    log_em0 = _log_emit(y[0])
    log_alpha = log_pi0 + log_em0                              # (K,)
    log_Z0 = _lse(log_alpha)
    gamma[0] = np.exp(log_alpha - log_Z0)

    # t = 1 ... T-1
    for t in range(1, T):
        # predictive: pred[k] = Σ_i gamma[t-1, i] * P[i, k]
        pred = gamma[t-1] @ P                                  # (K,)
        log_em = _log_emit(y[t])
        log_w_new = np.log(np.maximum(pred, 1e-300)) + log_em
        log_Z = _lse(log_w_new)
        gamma[t] = np.exp(log_w_new - log_Z)
        cpll += float(log_Z)
        if t >= T_split:
            cpll_oos += float(log_Z)

    zhat = np.argmax(gamma, axis=1).astype(int)
    xhat = y.copy()                                            # identity
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)

    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "cpll_oos": float(cpll_oos),
            "max_cpll": float(max_cpll), "mdl": mdl}


def inference_ARHMM(px, mdl, y, T_split, dt=1/252, display=False):
    """
    Causal inference for a fitted diagonal-Gaussian AR-HMM.

    Strictly forward-only filter. Uses per-regime VAR(1) matrices (As),
    intercepts (bs), diagonal innovation variances (sigmasq), and the
    stationary transition matrix.

    At t = 0 there is no y_{t-1}; emission log-lik uses the intercept alone
    (b_k, sigmasq_k) — matches cusum_overlay's ARHMM init convention.

    Returns {xhat, zhat, gamma, cpll, max_cpll, mdl} with xhat == y (identity).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape

    # --- learned params ---
    K = int(mdl.K)
    As_obs   = np.asarray(mdl.observations.As,      dtype=float)    # (K, N, N)
    bs_obs   = np.asarray(mdl.observations.bs,      dtype=float)    # (K, N)
    sigma2   = np.clip(np.asarray(mdl.observations.sigmasq, dtype=float), 1e-12, None)  # (K, N)
    P        = np.asarray(mdl.transitions.transition_matrix, dtype=float)  # (K, K)
    log_pi0  = np.asarray(mdl.init_state_distn.log_pi0, dtype=float)
    log_pi0  = log_pi0 - _lse(log_pi0)

    # --- t = 0 emission (no y_{t-1}; use intercept b_k) ---
    def _log_emit_t0(y0):
        diffs = y0[None, :] - bs_obs                           # (K, N)
        return -0.5 * np.sum(diffs**2 / sigma2 + np.log(2.0 * np.pi * sigma2), axis=1)

    # --- t >= 1 emission: mu_k(t) = As[k] @ y_{t-1} + bs[k] ---
    def _log_emit(yt, y_prev):
        mu_k = np.einsum('knm,m->kn', As_obs, y_prev) + bs_obs  # (K, N)
        diffs = yt[None, :] - mu_k
        return -0.5 * np.sum(diffs**2 / sigma2 + np.log(2.0 * np.pi * sigma2), axis=1)

    gamma = np.zeros((T, K))
    cpll  = 0.0
    cpll_oos = 0.0

    # t = 0
    log_em0 = _log_emit_t0(y[0])
    log_alpha = log_pi0 + log_em0
    log_Z0 = _lse(log_alpha)
    gamma[0] = np.exp(log_alpha - log_Z0)

    for t in range(1, T):
        pred = gamma[t-1] @ P                                  # (K,)
        log_em = _log_emit(y[t], y[t-1])
        log_w_new = np.log(np.maximum(pred, 1e-300)) + log_em
        log_Z = _lse(log_w_new)
        gamma[t] = np.exp(log_w_new - log_Z)
        cpll += float(log_Z)
        if t >= T_split:
            cpll_oos += float(log_Z)

    zhat = np.argmax(gamma, axis=1).astype(int)
    xhat = y.copy()                                            # identity
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)

    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "cpll_oos": float(cpll_oos),
            "max_cpll": float(max_cpll), "mdl": mdl}


def filter_states_causal(y, mdl, model_type, pi0):
    """
    Causal (filtered) state inference for HMM and AR-HMM models.

    At each time t = 0, ..., T-1, returns argmax_k P(z_t = k | y_{0:t}) — that is,
    the filtered posterior using ONLY observations up to and including t. No
    backward pass, no smoothing.

    This is the look-ahead-free alternative to Viterbi/expected_states on an
    out-of-sample window.

    Parameters
    ----------
    y : array (T, N) or (T,)
        Observation sequence.
    mdl : fitted ssm.HMM (as produced by fit_HMM or fit_AR_HMM).
    model_type : "hmm" or "arhmm".
    pi0 : array (K,)
        Initial discrete prior over states at t=0.

    Returns
    -------
    zhat : array (T,), dtype int
        Causal filtered argmax state at each t.
    """
    from scipy.special import logsumexp

    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape

    P    = np.asarray(mdl.transitions.transition_matrix, dtype=float)   # (K, K)
    logP = np.log(np.clip(P, 1e-300, None))
    K    = P.shape[0]

    if model_type == "hmm":
        mus  = np.asarray(mdl.observations.mus,     dtype=float)        # (K, N)
        sig2 = np.clip(np.asarray(mdl.observations.sigmasq, dtype=float), 1e-12, None)
    elif model_type == "arhmm":
        A_obs = np.asarray(mdl.observations.As,      dtype=float)       # (K, N, N)
        b_obs = np.asarray(mdl.observations.bs,      dtype=float)       # (K, N)
        sig2  = np.clip(np.asarray(mdl.observations.sigmasq, dtype=float), 1e-12, None)
    else:
        raise ValueError(f"filter_states_causal: unsupported model_type {model_type!r}")

    # Log-alpha recursion (filtered log posteriors, unnormalised; normalise per step for numerical stability)
    pi0 = np.asarray(pi0, dtype=float).reshape(-1)
    log_pi0 = np.log(np.clip(pi0, 1e-300, None))
    log_alpha = np.full((T, K), -np.inf)
    zhat = np.zeros(T, dtype=int)

    # t = 0: emission ll depends on model_type
    if model_type == "hmm":
        ll0 = -0.5 * np.sum((y[0] - mus)**2 / sig2 + np.log(2*np.pi*sig2), axis=1)
    else:  # arhmm: at t=0 we have no y_{-1}; use intercept b_obs as prediction
        ll0 = -0.5 * np.sum((y[0] - b_obs)**2 / sig2 + np.log(2*np.pi*sig2), axis=1)
    log_alpha[0] = log_pi0 + ll0
    log_alpha[0] -= logsumexp(log_alpha[0])   # normalise
    zhat[0] = int(np.argmax(log_alpha[0]))

    # Forward recursion
    for t in range(1, T):
        # predict: log p(z_t=j | y_{0:t-1}) = logsumexp_i( log_alpha[t-1,i] + logP[i,j] )
        log_pred = logsumexp(log_alpha[t-1][:, None] + logP, axis=0)   # (K,)

        # emission ll at t
        if model_type == "hmm":
            ll_t = -0.5 * np.sum((y[t] - mus)**2 / sig2 + np.log(2*np.pi*sig2), axis=1)
        else:  # arhmm
            y_prev = y[t-1]
            ll_t = np.empty(K)
            for j in range(K):
                eps = y[t] - (A_obs[j] @ y_prev + b_obs[j])
                ll_t[j] = -0.5 * np.sum(eps**2 / sig2[j] + np.log(2*np.pi*sig2[j]))

        log_alpha[t] = log_pred + ll_t
        log_alpha[t] -= logsumexp(log_alpha[t])   # normalise
        zhat[t] = int(np.argmax(log_alpha[t]))

    return zhat


# ----------------------------------------------------------------------------
# causal_cpll_SLDS — true causal one-step predictive LL for a standard SLDS
#
# Same GPB1 filter as causal_cpll_rSLDS except:
#   - rSLDS gate: log_pz[k] = (Rs @ m + r)[k]          (depends on x_{t-1})
#   - SLDS  gate: log_bz_pred = logsumexp(log_bz + log_Ps, axis=0)
#                                                      (Markov chain on z)
# Tracks a filtered belief b_z over the K regimes; everything else (per-regime
# Kalman predict/update, GPB1 collapse of x) is identical. Comparable on the
# same scale as inference_HMM / inference_ARHMM / causal_cpll_rSLDS.
# ----------------------------------------------------------------------------

def causal_cpll_SLDS(mdl, y, T_split):
    """
    transitions='standard', dynamics='diagonal_gaussian', emissions='gaussian'.
    Returns (cpll, cpll_oos): cpll = sum_{t=1..T-1} log p(y_t | y_{0:t-1});
    cpll_oos = the same sum restricted to t >= T_split.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape
    K = int(mdl.K)
    D = int(mdl.D)

    As     = np.asarray(mdl.dynamics.As, dtype=float)          # (K, D, D)
    bs     = np.asarray(mdl.dynamics.bs, dtype=float)          # (K, D)
    Qd     = np.clip(np.asarray(mdl.dynamics.sigmasq, dtype=float), 1e-12, None)
    log_Ps = np.asarray(mdl.transitions.log_Ps, dtype=float)   # (K, K)
    Cs     = np.asarray(mdl.emissions.Cs, dtype=float)
    ds     = np.asarray(mdl.emissions.ds, dtype=float)
    invE   = np.asarray(mdl.emissions.inv_etas, dtype=float)
    log_pi0 = np.asarray(mdl.init_state_distn.log_pi0, dtype=float)
    log_pi0 = log_pi0 - logsumexp(log_pi0)

    def _erow(arr, k):
        return arr[0] if arr.shape[0] == 1 else arr[k]

    # init belief over x_0 from y_0 via the shared/first emission row
    C0 = Cs[0]; d0 = ds[0]
    R0 = np.diag(np.exp(-invE[0]))
    Cpinv = np.linalg.pinv(C0)
    m = Cpinv @ (y[0] - d0)
    P = Cpinv @ R0 @ Cpinv.T + 1e-6 * np.eye(D)

    log_bz = log_pi0.copy()   # filtered belief over z_0

    eyeD = np.eye(D)
    cpll = 0.0
    cpll_oos = 0.0
    for t in range(1, T):
        # propagate z-belief through the Markov chain (no x-dependence)
        log_bz_pred = logsumexp(log_bz[:, None] + log_Ps, axis=0)
        log_bz_pred = log_bz_pred - logsumexp(log_bz_pred)

        comp_ll = np.empty(K)
        m_upd = np.empty((K, D))
        P_upd = np.empty((K, D, D))
        for k in range(K):
            mpk = As[k] @ m + bs[k]
            Ppk = As[k] @ P @ As[k].T + np.diag(Qd[k])
            Ck = _erow(Cs, k); dk = _erow(ds, k)
            Rk = np.diag(np.exp(-_erow(invE, k)))
            yhat = Ck @ mpk + dk
            S = Ck @ Ppk @ Ck.T + Rk
            comp_ll[k] = _mvn_logpdf(y[t], yhat, S)
            Sinv = np.linalg.inv(0.5 * (S + S.T) + 1e-10 * np.eye(N))
            Kg = Ppk @ Ck.T @ Sinv
            m_upd[k] = mpk + Kg @ (y[t] - yhat)
            P_upd[k] = (eyeD - Kg @ Ck) @ Ppk

        joint = log_bz_pred + comp_ll
        denom = logsumexp(joint)
        cpll += float(denom)
        if t >= T_split:
            cpll_oos += float(denom)
        log_bz = joint - denom   # filtered z-belief at time t

        w = np.exp(log_bz)
        m_new = np.sum(w[:, None] * m_upd, axis=0)
        P_new = np.zeros((D, D))
        for k in range(K):
            dmk = (m_upd[k] - m_new)[:, None]
            P_new += w[k] * (P_upd[k] + dmk @ dmk.T)
        m, P = m_new, P_new

    return cpll, cpll_oos


# ============================================================================
# inference_SLDS — same expanding-window scheme as inference_rSLDS;
# only the cpll call changes (standard transitions instead of recurrent).
# ============================================================================

def inference_SLDS(px, mdl, y, T_train, cadence, dt=1/252, display=False):
    """
    OOS inference for a fitted standard SLDS. Smoothing helper, expanding-
    window walk-forward and GPB1 collapse logic are identical to
    inference_rSLDS — the variational posterior, most_likely_states and
    expected_states all work for standard transitions unchanged. Only the
    causal one-step predictive log-likelihood (causal_cpll_SLDS) differs.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape
    T_train = int(T_train); cadence = int(cadence)
    assert cadence >= 1, f"cadence must be >= 1, got {cadence}"
    assert 0 <= T_train <= T, f"T_train={T_train} out of range for T={T}"

    K = int(mdl.K); D = int(mdl.D)

    def _smooth(y_window):
        Tw = y_window.shape[0]
        Fs = getattr(mdl.emissions, "Fs", [])
        D_in = Fs[0].shape[1] if len(Fs) else 0
        inputs = np.zeros((Tw, D_in))
        mask = np.ones_like(y_window, dtype=bool)
        q = mdl._make_variational_posterior(
            variational_posterior="structured_meanfield",
            datas=[y_window], inputs=[inputs], masks=[mask], tags=[None],
            method="smf",
        )
        x_smooth = q.mean_continuous_states[0]
        z_smooth = mdl.most_likely_states(x_smooth, y_window)
        g_smooth, *_ = mdl.expected_states(x_smooth, y_window, mask=mask)
        return x_smooth, z_smooth, g_smooth

    # short-circuit: no OOS expansion needed
    if T_train >= T or (T_train + cadence) > T:
        x_full, z_full, g_full = _smooth(y)
        max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)
        cpll, cpll_oos = causal_cpll_SLDS(mdl, y, T_split=T_train)
        return {"xhat": x_full, "zhat": z_full.astype(int), "gamma": g_full,
                "cpll": float(cpll), "cpll_oos": float(cpll_oos),
                "max_cpll": float(max_cpll), "mdl": mdl}

    xhat  = np.zeros((T, D))
    zhat  = np.zeros(T,  dtype=int)
    gamma = np.zeros((T, K))

    if T_train > 0:
        x_tr, z_tr, g_tr = _smooth(y[:T_train])
        xhat[:T_train]  = x_tr
        zhat[:T_train]  = z_tr.astype(int)
        gamma[:T_train] = g_tr

    block_start = T_train
    while block_start < T:
        block_end = min(block_start + cadence, T)
        x_d, z_d, g_d = _smooth(y[:block_start])
        xhat[block_start:block_end]  = x_d[-1]
        zhat[block_start:block_end]  = int(z_d[-1])
        gamma[block_start:block_end] = g_d[-1]
        block_start = block_end

    cpll, cpll_oos = causal_cpll_SLDS(mdl, y, T_split=T_train)
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)
    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "cpll_oos": float(cpll_oos),
            "max_cpll": float(max_cpll), "mdl": mdl}



# ============================================================================
# Test 3 (state dependence) — matched-component transition scoring.
#
# The candidate (fitted rSLDS) is compared against a null that shares EVERY
# regime-conditional parameter (dynamics A,b,Q; emissions C,d,S) and differs
# ONLY in the transition mechanism: the recurrent gate softmax(R x + r) is
# replaced by a fixed time-homogeneous K x K matrix, estimated on the same
# training window with all other parameters frozen (best-response null, via
# the same causal filter likelihood used for scoring). The comparison is
# nested with a single degree of freedom, so any predictive difference is
# attributable to the state dependence of the transitions alone, and an
# rSLDS/SLDS fit-collapse asymmetry cannot occur (both models own the same
# regime-conditional densities).
#
# Both are scored with the h-step causal predictive log-likelihood
#     CPLL_h = sum_t log p(y_{t+h} | y_{1:t})
# (GPB1 filter; h-step belief propagation with NO intermediate updates). At
# h = 1 the emission largely re-identifies the regime each step, so the
# transition model barely enters the score; at h > 1 it is the only source
# of regime-forecast information. Numerics are IDENTICAL for both transition
# kinds (bounded dynamics-based P0, guarded lstsq seed, adaptive-jitter
# Cholesky density, eigenvalue cap); the recurrent path at h = 1 reproduces
# causal_cpll_rSLDS exactly. The gate is evaluated at the collapsed predicted
# latent mean (point-gate approximation, as in the one-step scorer).
# ============================================================================

_T3_P_CEIL = 1e10
_T3_P_FLOOR = 1e-12
_T3_JIT0 = 1e-9


def _t3_safe_mvn_logpdf(x, mean, cov, N, eyeN):
    """log N(x | mean, cov) via jittered Cholesky; finite for degenerate cov."""
    S = 0.5 * (cov + cov.T)
    scale = max(1.0, np.trace(S) / max(N, 1))
    jit = _T3_JIT0 * scale
    L = None
    for _ in range(8):
        try:
            L = np.linalg.cholesky(S + jit * eyeN)
            break
        except np.linalg.LinAlgError:
            jit *= 10.0
    if L is None:
        return -1e12
    diff = x - mean
    z = np.linalg.solve(L, diff)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (N * np.log(2.0 * np.pi) + logdet + float(z @ z))


def _t3_cap_cov(P):
    P = 0.5 * (P + P.T)
    ev, V = np.linalg.eigh(P)
    ev = np.clip(ev, _T3_P_FLOOR, _T3_P_CEIL)
    return (V * ev) @ V.T


def causal_cpll_h(mdl, y, T_split, h=4, transition_kind="recurrent",
                  log_Ps=None, log_pi0=None):
    """
    h-step causal predictive log-likelihood, summed over all forecast targets
    t' = h .. T-1 (full) and over t' >= T_split (OOS). Each target conditions
    only on y_{1:t'-h}: the filter state at the origin is propagated h steps
    through the dynamics and transition model with no intermediate updates.

    transition_kind = "recurrent": gate softmax(Rs @ m + r) at the collapsed
        predicted mean (read from mdl.transitions).
    transition_kind = "standard": homogeneous matrix; pass log_Ps (K, K) and
        log_pi0 (K,) explicitly (the matched null), or they are read from
        mdl.transitions / mdl.init_state_distn.

    Returns (cpll_h, cpll_h_oos). For h = 1 the recurrent path reproduces
    causal_cpll_rSLDS (same filter, same numerics).
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape
    K = int(mdl.K)
    D = int(mdl.D)
    h = int(h)
    assert h >= 1, f"h must be >= 1, got {h}"

    As = np.asarray(mdl.dynamics.As, dtype=float)
    bs = np.asarray(mdl.dynamics.bs, dtype=float)
    Qd = np.clip(np.asarray(mdl.dynamics.sigmasq, dtype=float), 1e-12, None)
    Cs = np.asarray(mdl.emissions.Cs, dtype=float)
    ds = np.asarray(mdl.emissions.ds, dtype=float)
    invE = np.asarray(mdl.emissions.inv_etas, dtype=float)

    def _erow(arr, k):
        arr = np.asarray(arr)
        if arr.ndim == 1:
            return arr
        return arr[0] if arr.shape[0] == 1 else arr[k]

    eyeD, eyeN = np.eye(D), np.eye(N)
    recurrent = (transition_kind == "recurrent")
    if recurrent:
        Rs = np.asarray(mdl.transitions.Rs, dtype=float)
        rvec = np.asarray(mdl.transitions.r, dtype=float)
    else:
        if log_Ps is None:
            log_Ps = np.asarray(mdl.transitions.log_Ps, dtype=float)
        log_Ps = log_Ps - logsumexp(log_Ps, axis=1, keepdims=True)
        if log_pi0 is None:
            log_pi0 = np.asarray(mdl.init_state_distn.log_pi0, dtype=float)
        log_pi0 = log_pi0 - logsumexp(log_pi0)

    Rk_all = [np.diag(np.exp(-_erow(invE, k))) for k in range(K)]
    Ck_all = [_erow(Cs, k) for k in range(K)]
    dk_all = [_erow(ds, k) for k in range(K)]

    def _prior(m_cur, log_bz_cur):
        if recurrent:
            lp = Rs @ m_cur + rvec
        else:
            lp = logsumexp(log_bz_cur[:, None] + log_Ps, axis=0)
        return lp - logsumexp(lp)

    def _forecast(m, P, log_bz, y_tgt):
        """h-step density of y_tgt from the filter state at the origin:
        (h-1) GPB1 propagate-collapse steps, then a mixture one-step
        predictive."""
        m_cur, P_cur, lbz = m, P, log_bz
        for _ in range(h - 1):
            lp = _prior(m_cur, lbz)
            w = np.exp(lp)
            mks = np.empty((K, D))
            Pks = np.empty((K, D, D))
            for k in range(K):
                mks[k] = As[k] @ m_cur + bs[k]
                Pk = As[k] @ P_cur @ As[k].T + np.diag(Qd[k])
                Pks[k] = 0.5 * (Pk + Pk.T)
            m_new = np.sum(w[:, None] * mks, axis=0)
            P_new = np.zeros((D, D))
            for k in range(K):
                dmk = (mks[k] - m_new)[:, None]
                P_new += w[k] * (Pks[k] + dmk @ dmk.T)
            m_cur, P_cur, lbz = m_new, _t3_cap_cov(P_new), lp
        lp = _prior(m_cur, lbz)
        comp = np.empty(K)
        for k in range(K):
            mpk = As[k] @ m_cur + bs[k]
            Ppk = As[k] @ P_cur @ As[k].T + np.diag(Qd[k])
            yhat = Ck_all[k] @ mpk + dk_all[k]
            S = Ck_all[k] @ Ppk @ Ck_all[k].T + Rk_all[k]
            comp[k] = _t3_safe_mvn_logpdf(y_tgt, yhat, S, N, eyeN)
        return logsumexp(lp + comp)

    # bounded, dynamics-based prior over x_0; guarded least-squares seed
    rho_diag = np.stack([np.clip(np.abs(np.diag(As[k])), 0.0, 0.999)
                         for k in range(K)])
    stat_var = Qd / (1.0 - rho_diag ** 2)
    P = np.diag(np.clip(np.median(stat_var, axis=0), _T3_P_FLOOR, _T3_P_CEIL))
    m, *_ = np.linalg.lstsq(Cs[0], y[0] - ds[0], rcond=1e-6)
    log_bz = (np.full(K, -np.log(K)) if recurrent else log_pi0.copy())

    cpll_h = 0.0
    cpll_h_oos = 0.0
    # forecast from the origin t = 0 state (no update yet) for target t = h
    if h <= T - 1:
        val = _forecast(m, P, log_bz, y[h])
        cpll_h += val
        if h >= T_split:
            cpll_h_oos += val

    for t in range(1, T):
        # ---- filter update at time t (identical for both kinds) ----
        lp1 = _prior(m, log_bz)
        comp = np.empty(K)
        m_upd = np.empty((K, D))
        P_upd = np.empty((K, D, D))
        for k in range(K):
            mpk = As[k] @ m + bs[k]
            Ppk = As[k] @ P @ As[k].T + np.diag(Qd[k])
            Ppk = 0.5 * (Ppk + Ppk.T)
            yhat = Ck_all[k] @ mpk + dk_all[k]
            S = Ck_all[k] @ Ppk @ Ck_all[k].T + Rk_all[k]
            comp[k] = _t3_safe_mvn_logpdf(y[t], yhat, S, N, eyeN)
            Ssym = 0.5 * (S + S.T) + 1e-10 * eyeN
            Kg = np.linalg.solve(Ssym, Ck_all[k] @ Ppk).T
            m_upd[k] = mpk + Kg @ (y[t] - yhat)
            Pk = (eyeD - Kg @ Ck_all[k]) @ Ppk
            P_upd[k] = 0.5 * (Pk + Pk.T)
        joint = lp1 + comp
        denom = logsumexp(joint)
        w = np.exp(joint - denom)
        m_new = np.sum(w[:, None] * m_upd, axis=0)
        P_new = np.zeros((D, D))
        for k in range(K):
            dmk = (m_upd[k] - m_new)[:, None]
            P_new += w[k] * (P_upd[k] + dmk @ dmk.T)
        m, P = m_new, _t3_cap_cov(P_new)
        log_bz = joint - denom

        # ---- h-step forecast from the state at origin t ----
        tgt = t + h
        if tgt <= T - 1:
            val = _forecast(m, P, log_bz, y[tgt])
            cpll_h += val
            if tgt >= T_split:
                cpll_h_oos += val

    return float(cpll_h), float(cpll_h_oos)


def fit_null_transitions(mdl_r, y_tr, zhat_tr=None):
    """
    Estimate the matched null's (log_Ps, log_pi0) on the training window with
    all regime-conditional parameters frozen at the fitted rSLDS values, by
    maximizing the same causal filter likelihood used for scoring (h = 1).
    Warm-started from Laplace-smoothed transition counts of the decoded
    training path when available; mildly sticky uniform otherwise.
    Nelder-Mead on the row logits (rows renormalized inside the objective);
    falls back to the warm start if optimization does not improve on it.
    """
    from scipy.optimize import minimize

    y_tr = np.asarray(y_tr, dtype=float)
    if y_tr.ndim == 1:
        y_tr = y_tr[:, None]
    K = int(mdl_r.K)

    if zhat_tr is not None and len(np.asarray(zhat_tr)) > 1:
        z = np.asarray(zhat_tr, int)
        cnt = np.ones((K, K))
        for a, b in zip(z[:-1], z[1:]):
            cnt[a, b] += 1.0
        pi_cnt = np.bincount(z, minlength=K).astype(float) + 1.0
    else:
        cnt = np.ones((K, K)) + 9.0 * np.eye(K)
        pi_cnt = np.ones(K)
    L0 = np.log(cnt / cnt.sum(axis=1, keepdims=True))
    log_pi0 = np.log(pi_cnt / pi_cnt.sum())

    def _neg_cpll(theta):
        Lps = theta.reshape(K, K)
        Lps = Lps - logsumexp(Lps, axis=1, keepdims=True)
        c, _ = causal_cpll_h(mdl_r, y_tr, T_split=y_tr.shape[0], h=1,
                             transition_kind="standard",
                             log_Ps=Lps, log_pi0=log_pi0)
        return -c

    f0 = _neg_cpll(L0.ravel())
    res = minimize(_neg_cpll, L0.ravel(), method="Nelder-Mead",
                   options={"maxiter": 200 * K * K, "xatol": 1e-3,
                            "fatol": 1e-4})
    theta = res.x if (np.isfinite(res.fun) and res.fun <= f0) else L0.ravel()
    log_Ps = theta.reshape(K, K)
    log_Ps = log_Ps - logsumexp(log_Ps, axis=1, keepdims=True)
    return log_Ps, log_pi0


def t3_pair_scores(mdl_r, y_tr, y_joint, T_split, h_grid=(1, 4), zhat_tr=None):
    """
    T3 scoring for one (security, batch): fit the matched null on the
    training window, then score candidate and null on the joint [train, test]
    series at each horizon in h_grid. Returns a flat dict of per-batch
    columns:
        cpll{h}_oos        — rSLDS h-step OOS CPLL
        cpll{h}_null_oos   — matched-null h-step OOS CPLL
        t3_gap_h{h}        — their difference (the T3 per-batch gap)
    """
    log_Ps, log_pi0 = fit_null_transitions(mdl_r, y_tr, zhat_tr=zhat_tr)
    out = {}
    for h in h_grid:
        _, oos_r = causal_cpll_h(mdl_r, y_joint, T_split, h=h,
                                 transition_kind="recurrent")
        _, oos_0 = causal_cpll_h(mdl_r, y_joint, T_split, h=h,
                                 transition_kind="standard",
                                 log_Ps=log_Ps, log_pi0=log_pi0)
        out[f"cpll{h}_oos"] = float(oos_r)
        out[f"cpll{h}_null_oos"] = float(oos_0)
        out[f"t3_gap_h{h}"] = float(oos_r - oos_0)
    return out
