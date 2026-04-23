# inference.py

import numpy as np
from scipy.special import logsumexp
from ._utils import _lse, _mvn_logpdf
from .bounds import max_cpll_causal_bound


# ---------------------------------------------------------------
# rSLDS Inference
# ---------------------------------------------------------------

def inference_rSLDS(px, mdl, y, dt=1/252, display=False, x0_var=1.0):
    """
    Causal inference for a fitted rSLDS.

    Strictly forward-only IMM (Interacting Multiple Model) filter:
      at every t, xhat[t], zhat[t], gamma[t] depend only on y[0:t+1].
    Uses the learned dynamics (A, b, Q), recurrent transitions (Rs, r, log_Ps),
    emissions (C, d, diag(exp(inv_etas))), and initial state distribution (pi0).
    Initial continuous state prior: N(0, x0_var * I) per regime.

    Returns {xhat, zhat, gamma, cpll, max_cpll, mdl} where all arrays are the
    causal filtered quantities over the full sequence y.
    """
    y = np.asarray(y, dtype=float)
    if y.ndim == 1:
        y = y[:, None]
    T, N = y.shape

    # --- extract learned params (all frozen) ---
    K  = int(mdl.K)
    D  = int(mdl.D)
    A  = np.asarray(mdl.dynamics.As, dtype=float)           # (K, D, D)
    b  = np.asarray(mdl.dynamics.bs, dtype=float)           # (K, D)
    # dynamics noise covariance (diagonal-Gaussian case exposes sigmasq)
    Q_diag = np.asarray(mdl.dynamics.sigmasq, dtype=float)  # (K, D)
    Q  = np.stack([np.diag(q) for q in Q_diag], axis=0)     # (K, D, D)
    # emissions (single_subspace=True assumed: shape (1, N, D))
    C  = np.asarray(mdl.emissions.Cs,       dtype=float)[0] # (N, D)
    d_ = np.asarray(mdl.emissions.ds,       dtype=float)[0] # (N,)
    R_diag = np.exp(np.asarray(mdl.emissions.inv_etas, dtype=float)[0])  # (N,)
    R  = np.diag(R_diag)                                    # (N, N)
    # recurrent transitions
    log_Ps_base = np.asarray(mdl.transitions.log_Ps, dtype=float)  # (K, K)
    log_Ps_base = log_Ps_base - _lse(log_Ps_base, axis=1, keepdims=True)
    Rs = np.asarray(mdl.transitions.Rs, dtype=float)        # (K, D)
    # initial state distribution
    log_pi0 = np.asarray(mdl.init_state_distn.log_pi0, dtype=float)  # (K,)
    pi0 = np.exp(log_pi0 - _lse(log_pi0))

    # --- IMM filter state ---
    # per-regime (μ, Σ), per-regime weight w. Init at t = 0 before observing y[0]:
    #   x_0 | z_0 = k  ~ N(0, x0_var * I)  (diffuse, matches ssm's implicit prior)
    mu  = np.zeros((K, D))
    Sig = np.tile(float(x0_var) * np.eye(D), (K, 1, 1))     # (K, D, D)
    w   = pi0.copy()                                        # (K,)

    # outputs
    xhat  = np.zeros((T, D))
    gamma = np.zeros((T, K))
    cpll  = 0.0

    for t in range(T):
        y_t = y[t]

        # ---- predictive mixture given y[0:t] (used for cpll at t >= 1) ----
        if t == 0:
            # no "prediction from t-1" at t=0; update directly from prior
            # predictive per-regime is (mu, Sig) themselves (the prior)
            mu_pred, Sig_pred = mu, Sig
            w_mix = w.copy()
        else:
            # --- IMM interaction step: mix (mu, Sig, w) into each target regime j ---
            # Per transitions.py RecurrentTransitions.log_transition_matrices,
            # log_Ps[t, k, j] = log_Ps_base[k, j] + (Rs[j] . x_t),
            # i.e. the recurrent effect is additive on column j and shared across
            # rows k. In IMM each source regime k carries its own mean mu[k],
            # so we build a K x K transition matrix using mu[k] as the source state.
            log_Ps_full = np.zeros((K, K))
            for k in range(K):
                logits = log_Ps_base[k, :] + (Rs @ mu[k])   # (K,)
                log_Ps_full[k, :] = logits - _lse(logits)
            P = np.exp(log_Ps_full)                         # (K, K) row-stochastic

            # mixture weight for each target j: w_mix[j] = Σ_k w[k] * P[k, j]
            w_mix = w @ P                                    # (K,)
            w_mix = np.maximum(w_mix, 1e-300)

            # conditional source-given-target: p_k_given_j[k, j] = w[k] P[k,j] / w_mix[j]
            p_kj = (w[:, None] * P) / w_mix[None, :]         # (K, K)

            # collapse per target j
            mu_tilde  = p_kj.T @ mu                          # (K, D) target j x D
            Sig_tilde = np.zeros((K, D, D))
            for j in range(K):
                for k in range(K):
                    dd = mu[k] - mu_tilde[j]
                    Sig_tilde[j] += p_kj[k, j] * (Sig[k] + np.outer(dd, dd))

            # predict per target j
            mu_pred  = np.zeros((K, D))
            Sig_pred = np.zeros((K, D, D))
            for j in range(K):
                mu_pred[j]  = A[j] @ mu_tilde[j] + b[j]
                Sig_pred[j] = A[j] @ Sig_tilde[j] @ A[j].T + Q[j]

        # ---- update on y_t per regime (Kalman) ----
        mu_post  = np.zeros((K, D))
        Sig_post = np.zeros((K, D, D))
        L = np.zeros(K)           # per-regime predictive likelihood p(y_t | z_t=j, y[0:t])
        for j in range(K):
            S_j = C @ Sig_pred[j] @ C.T + R                  # (N, N)
            y_pred = C @ mu_pred[j] + d_
            # log predictive likelihood
            L[j] = _mvn_logpdf(y_t, y_pred, S_j)
            # Kalman gain
            S_inv_C = np.linalg.solve(S_j, C)                # (N, D)
            Kg = Sig_pred[j] @ S_inv_C.T                     # (D, N)
            nu = y_t - y_pred
            mu_post[j]  = mu_pred[j] + Kg @ nu
            Sig_post[j] = Sig_pred[j] - Kg @ C @ Sig_pred[j]

        # ---- reweight and renormalise ----
        # log w_new[j] = log w_mix[j] + L[j]
        log_w_new = np.log(np.maximum(w_mix, 1e-300)) + L
        log_Z = _lse(log_w_new)
        w = np.exp(log_w_new - log_Z)

        # ---- accumulate causal cpll (skip t=0: no prediction from prior data) ----
        if t >= 1:
            cpll += float(log_Z)

        # ---- commit state ----
        mu  = mu_post
        Sig = Sig_post

        # ---- emit outputs ----
        xhat[t]  = (w[:, None] * mu).sum(axis=0)             # posterior mean
        gamma[t] = w

    zhat = np.argmax(gamma, axis=1).astype(int)
    max_cpll = max_cpll_causal_bound(y, reg_scale=1e-8)

    return {"xhat": xhat, "zhat": zhat, "gamma": gamma,
            "cpll": float(cpll), "max_cpll": float(max_cpll), "mdl": mdl}


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

