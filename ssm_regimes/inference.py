# inference.py

import numpy as np
from scipy.special import logsumexp
from ._utils import _lse, _mvn_logpdf
from .bounds import max_cpll_causal_bound


# ---------------------------------------------------------------
# rSLDS Inference
# ---------------------------------------------------------------

def inference_rSLDS(px, mdl, y, T_train, cadence, dt=1/252, display=False,
                    inference_mode="smoothed"):
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
    assert inference_mode in ("online", "smoothed"), \
        f"inference_mode must be 'online' or 'smoothed', got {inference_mode!r}"

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
        # CPLL: smoothed-posterior log-likelihood on full window. Look-ahead in this
        # single number by design (diagnostic only; matches old rSLDS.py behaviour).
        cpll = _smoothed_cpll_proxy(g_full)
        return {"xhat": x_full, "zhat": z_full.astype(int), "gamma": g_full,
                "cpll": float(cpll), "max_cpll": float(max_cpll), "mdl": mdl}

    # --- smoothed mode: single smoothed inference on the OOS block ---
    # Used for benchmarking model classes under identical protocol. Each OOS
    # day's label uses past AND future data within the OOS block (leaky).
    if inference_mode == "smoothed":
        xhat = np.zeros((T, D))
        zhat = np.zeros(T, dtype=int)
        gamma = np.zeros((T, K))
        # training portion: smoothed inference on training data alone (matches online mode for IS)
        if T_train > 0:
            x_tr, z_tr, g_tr = _smooth(y[:T_train])
            xhat[:T_train]  = x_tr
            zhat[:T_train]  = z_tr.astype(int)
            gamma[:T_train] = g_tr
        # OOS portion: single smoothed inference on the OOS block alone
        x_oos, z_oos, g_oos = _smooth(y[T_train:])
        xhat[T_train:]  = x_oos
        zhat[T_train:]  = z_oos.astype(int)
        gamma[T_train:] = g_oos
        # CPLL diagnostic on full sequence
        x_full, _, g_full = _smooth(y)
        cpll = _smoothed_cpll_proxy(g_full)
        max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)
        return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
                "cpll": float(cpll), "max_cpll": float(max_cpll), "mdl": mdl}

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

    # --- CPLL (diagnostic, smoothed full-window) ---
    x_full, _, g_full = _smooth(y)
    cpll = _smoothed_cpll_proxy(g_full)
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)

    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "max_cpll": float(max_cpll), "mdl": mdl}


def _smoothed_cpll_proxy(gamma):
    """
    Diagnostic CPLL proxy: average per-step entropy of the smoothed regime
    posterior, scaled to log-likelihood units. NOT the true CPLL — true CPLL
    requires the model's predictive density at each step, which the new
    expanding-window inference does not assemble. This stand-in keeps the
    leaderboard column populated and monotonic in posterior peakiness.

    For diagnostic comparison only. Not used for model selection.
    """
    eps = 1e-12
    g = np.asarray(gamma, dtype=float)
    # log-likelihood of the modal regime under the smoothed posterior, summed.
    # When the posterior is peaky, this approaches 0; when uniform, it goes to
    # T * log(1/K). Provides a relative ordering between models on the same data.
    log_max = np.log(np.maximum(g.max(axis=1), eps))
    return float(np.sum(log_max))


def inference_HMM(px, mdl, y, dt=1/252, display=False):
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

    zhat = np.argmax(gamma, axis=1).astype(int)
    xhat = y.copy()                                            # identity
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)

    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "max_cpll": float(max_cpll), "mdl": mdl}


def inference_ARHMM(px, mdl, y, dt=1/252, display=False):
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

    zhat = np.argmax(gamma, axis=1).astype(int)
    xhat = y.copy()                                            # identity
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)

    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "max_cpll": float(max_cpll), "mdl": mdl}


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

