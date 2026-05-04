# cusum.py

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp


# ---------------------------------------------------------------
# CUSUM Overlay
# ---------------------------------------------------------------

def cusum_overlay(prices, y, xhat, mdl, h_z, model_type, verbose=False):
    
    """
    Multi-hypothesis CUSUM (parallel per-alt streams) on latent-dynamics + transition log-likelihoods.
    Works for K >= 2.

    For each t and current regime k:
        score_j = log P(x_t | x_{t-1}, j) + log P(j | x_{t-1})
        s_{j|k} = score_j - score_k
        S_t^{(j)} = max(0, S_{t-1}^{(j)} + s_{j|k}), j != k
    If max_j S_t^{(j)} > h_z, switch to argmax_j S_t^{(j)} and reset all S^{(·)}.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.special import logsumexp

    # short series guard
    if xhat.shape[0] <= 10:
        return None

    # shapes
    y = np.atleast_2d(y)
    xhat = np.atleast_2d(xhat)
    if y.shape[0] < y.shape[1]: y = y.T
    if xhat.shape[0] < xhat.shape[1]: xhat = xhat.T
    T, D = xhat.shape

    # model params
    if model_type == "hmm":
        # HMM: no dynamics; diag-Gaussian emissions on y (xhat == y).
        mus_e  = np.asarray(mdl.observations.mus,     dtype=float)  # (K, D)
        sig2_e = np.clip(np.asarray(mdl.observations.sigmasq, dtype=float), 1e-12, None)
        P      = np.asarray(mdl.transitions.transition_matrix, dtype=float)  # (K, K)
        logP   = np.log(np.clip(P, 1e-300, None))
        K      = mus_e.shape[0]
    elif model_type == "arhmm":
        # AR-HMM: y_t | z_t=j ~ N(A_j y_{t-1} + b_j, diag(sig2_j)); xhat == y.
        A_obs   = np.asarray(mdl.observations.As,      dtype=float)  # (K, D, D)
        b_obs   = np.asarray(mdl.observations.bs,      dtype=float)  # (K, D)
        sig2_e  = np.clip(np.asarray(mdl.observations.sigmasq, dtype=float), 1e-12, None)
        P       = np.asarray(mdl.transitions.transition_matrix, dtype=float)
        logP    = np.log(np.clip(P, 1e-300, None))
        K       = A_obs.shape[0]
    else:
        A    = mdl.dynamics.As            # (K, D, D)
        b    = mdl.dynamics.bs            # (K, D)
        sig2 = mdl.dynamics.sigmasq       # (K, D)
        Rw   = mdl.transitions.Rs         # (K, D)
        r    = mdl.transitions.r          # (K,)
        K    = A.shape[0]

    # storage
    z_hat = np.zeros(T, dtype=int)
    S_star_series = []
    p_star_series = []

    # init regime
    if model_type == "hmm":
        y0 = xhat[0]
        ll0 = -0.5 * np.sum((y0 - mus_e)**2 / sig2_e + np.log(2*np.pi*sig2_e), axis=1)
        z0 = int(np.nanargmax(ll0))
    elif model_type == "arhmm":
        # No y_{t-1} available at t=0; init by emission ll against intercepts b_obs only.
        y0 = xhat[0]
        ll0 = -0.5 * np.sum((y0 - b_obs)**2 / sig2_e + np.log(2*np.pi*sig2_e), axis=1)
        z0 = int(np.nanargmax(ll0))
    else:
        xp0 = xhat[0]
        logits0 = Rw @ xp0 + r
        logits0 -= logsumexp(logits0)
        z0 = int(np.nanargmax(logits0))
    z_hat[0] = z0

    # per-alt CUSUM buffer
    S_vec = np.zeros(K)   # will ignore the current index each step

    for t in range(1, T):
        k_curr = z_hat[t-1]
        xp = xhat[t-1]
        xn = xhat[t]

        if model_type == "hmm":
            scores = np.empty(K)
            for j in range(K):
                eps = xn - mus_e[j]
                ll  = -0.5 * np.sum(eps**2 / sig2_e[j] + np.log(2*np.pi*sig2_e[j]))
                scores[j] = ll + logP[k_curr, j]
        elif model_type == "arhmm":
            scores = np.empty(K)
            for j in range(K):
                eps = xn - (A_obs[j] @ xp + b_obs[j])
                ll  = -0.5 * np.sum(eps**2 / sig2_e[j] + np.log(2*np.pi*sig2_e[j]))
                scores[j] = ll + logP[k_curr, j]
        else:
            # rSLDS: gating logits + latent dynamics
            logits = Rw @ xp + r
            logits -= logsumexp(logits)   # log softmax

            # regime scores: dynamics ll + transition log-prob
            # ll_j = -1/2 * sum_d [ (eps_jd^2 / sig2_jd) + log(2π sig2_jd) ]
            # where eps_j = xn - (A_j xp + b_j)
            scores = np.empty(K)
            for j in range(K):
                eps = xn - (A[j] @ xp + b[j])
                ll  = -0.5 * np.sum(eps**2 / sig2[j] + np.log(2*np.pi*sig2[j]))
                scores[j] = ll + logits[j]
        
        score_k = scores[k_curr]
        s_jk = scores - score_k
        s_jk[k_curr] = -np.inf  # exclude current from updates/argmax

        # update per-alt CUSUMs
        for j in range(K):
            if j == k_curr:
                continue
            S_vec[j] = max(0.0, S_vec[j] + s_jk[j])

        # select most suspicious alternative
        j_star = int(np.argmax(S_vec))
        S_star = S_vec[j_star]
        S_star_series.append(S_star)

        # optional posterior proxy of the challenger (for plotting)
        # p_star ≈ softmax(scores)[j_star]
        # compute safely by subtracting max
        m = np.max(scores)
        p_star = np.exp(scores[j_star] - m) / np.sum(np.exp(scores - m))
        p_star_series.append(p_star)

        # decision
        if S_star > h_z:
            z_hat[t] = j_star
            S_vec[:] = 0.0           # reset all streams after a change
        else:
            z_hat[t] = k_curr

    # --- Δn estimate (using max-CUSUM track) ---
    S_arr = np.asarray(S_star_series)
    pos_inc = np.maximum(np.diff(np.concatenate(([0.0], S_arr))), 0.0)
    avg_inc = np.convolve(pos_inc, np.ones(10)/10, mode="same")
    numer = np.maximum(h_z - S_arr, 0.0)
    denom = avg_inc
    Delta_n = np.full_like(S_arr, np.inf, dtype=float)
    np.divide(numer, denom, out=Delta_n, where=(denom > 0))
    pad = len(prices) - len(Delta_n) - 1
    if pad > 0:
        Delta_n = np.concatenate([Delta_n, np.full(pad, np.nan)])

    # plots
    if verbose:
        t_idx = prices.index[1:]
        fig, ax = plt.subplots(5, 1, figsize=(9, 6), sharex=True)

        ax[0].plot(t_idx, p_star_series, label="P*(challenger)")
        ax[0].set_ylabel("Posterior"); ax[0].legend(); ax[0].grid(True)

        ax[1].plot(t_idx, S_arr, label="max_j CUSUM S*_t")
        ax[1].axhline(h_z, color="red", linestyle="--", label="Threshold h_z")
        ax[1].legend(); ax[1].grid(True)

        ax[2].step(prices.index, z_hat, where='post', label="ẑ_t")
        ax[2].legend(); ax[2].grid(True)

        ax[3].plot(prices.index, prices.values, label="Price")
        ax[3].legend(); ax[3].grid(True)

        ax[4].plot(t_idx, Delta_n, label="Δn estimate")
        ax[4].set_yscale("log"); ax[4].legend(); ax[4].grid(True)

        plt.tight_layout()

    return z_hat


def cusum_overlay_basic(prices, y, xhat, mdl, h_z, verbose=False):
    
    """
    Adaptive-scale CUSUM on latent-dynamics + transition log-likelihoods.

    Definitions:
        log P(xₜ | xₜ₋₁, regime): measures how well the latent dynamics explain the observed state.
        log P(regime | xₜ₋₁): measures how likely the regime is given the past state.
        The sum is the joint log-probability; a complete measure of model fit at time t

    sₜ is the log-ratio of how well the alternative regime explains the current state vs the current regime.
    
    At each time t:
        sₜ = log-likelihood ratio between alternative and current regime:
            sₜ = [log P(xₜ | xₜ₋₁, alt regime) + log P(alt regime | xₜ₋₁)]
                − [log P(xₜ | xₜ₋₁, curr regime) + log P(curr regime | xₜ₋₁)]
    
        zₜ = (sₜ − μₜ) / σₜ        # z-score of sₜ
        Sₜ = max(0, Sₜ₋₁ + zₜ)     # cumulative sum of z-scores

    If Sₜ > h_z, a regime change is triggered.

    Reasonable ranges for h_z. High sensitivity: 2-3. Low sensitivity: 5-7
    
    """
    from scipy.special import logsumexp
    import numpy as np
    import matplotlib.pyplot as plt

    if xhat.shape[0] <= 10: 
        # print(f'skipping CUSUM: time series too short {xhat.shape[0]}')
        return None  # Skip CUSUM if time series too short
        
    # Ensure shapes
    y = np.atleast_2d(y)
    xhat = np.atleast_2d(xhat)
    if y.shape[0] < y.shape[1]: y = y.T
    if xhat.shape[0] < xhat.shape[1]: xhat = xhat.T
    T, D = xhat.shape

    # Model parameters
    A = mdl.dynamics.As
    b = mdl.dynamics.bs
    sig2 = mdl.dynamics.sigmasq
    Rw  = mdl.transitions.Rs
    r   = mdl.transitions.r

    # Storage
    z_hat = np.zeros(T, dtype=int)
    z_hat[0] = 1  # start in regime 1
    S_arr, z_arr = [], []
    p_new_series = []

    # Online stats
    mu = 0.0
    m2 = 0.0
    S = 0.0

    for t in range(1, T):
        k_curr = z_hat[t - 1]
        k_alt = 1 - k_curr
        xp = xhat[t - 1]
        xn = xhat[t]

        eps_c = xn - (A[k_curr] @ xp + b[k_curr])
        eps_a = xn - (A[k_alt]  @ xp + b[k_alt])

        ll_c = -0.5 * np.sum(eps_c**2 / sig2[k_curr] + np.log(2*np.pi*sig2[k_curr]))
        ll_a = -0.5 * np.sum(eps_a**2 / sig2[k_alt ] + np.log(2*np.pi*sig2[k_alt ]))

        logits = Rw @ xp + r
        logits -= logsumexp(logits)

        s = (ll_a + logits[k_alt]) - (ll_c + logits[k_curr])

        # Posterior for plotting
        p_new = 1.0 / (1.0 + np.exp(ll_c - ll_a))
        p_new_series.append(p_new)
        
        # Cumulative sum of s-values (no z-score)
        S = max(0.0, S + s)
        
        S_arr.append(S)
        
        if S > h_z:
            z_hat[t] = k_alt
            S = 0.0  # Reset cumulative sum after regime change
        else:
            z_hat[t] = k_curr

    # Δn estimate
    S_arr = np.asarray(S_arr)
    pos_inc = np.maximum(np.diff(np.concatenate(([0.0], S_arr))), 0.0)
    avg_inc = np.convolve(pos_inc, np.ones(10) / 10, mode="same")
    numer = np.maximum(h_z - S_arr, 0.0)
    denom = avg_inc
    Delta_n = np.full_like(S_arr, np.inf, dtype=float)
    np.divide(numer, denom, out=Delta_n, where=(denom > 0))
    pad = len(prices) - len(Delta_n) - 1
    if pad > 0:
        Delta_n = np.concatenate([Delta_n, np.full(pad, np.nan)])

    # -------- plots --------
    if verbose:
        t_idx = prices.index[1:]
        fig, ax = plt.subplots(6, 1, figsize=(9, 6), sharex=True)

        ax[0].plot(t_idx, p_new_series, label="P(new regime)")
        ax[0].set_ylabel("Posterior"); ax[0].legend(); ax[0].grid(True)

        ax[1].plot(t_idx, S_arr, label="CUSUM $S_t$")
        ax[1].axhline(h_z, color="red", linestyle="--", label="Threshold $h_z$")
        ax[1].legend(); ax[1].grid(True)

        ax[2].step(prices.index, z_hat, where='post', label=r"$\hat{z}_t$")
        ax[2].legend(); ax[2].grid(True)

        ax[3].plot(prices.index, prices.values, label="Price")
        ax[3].legend(); ax[3].grid(True)

        ax[4].plot(t_idx, Delta_n, label="Δn estimate")
        ax[4].set_yscale("log"); ax[4].legend(); ax[4].grid(True)

        ax[5].plot(prices.index, y[:, 0] if y.ndim == 2 else y, label="Observed $y_t$")
        ax[5].legend(); ax[5].grid(True)

        plt.tight_layout()

    return z_hat


def cusum_overlay_zscore(prices, y, xhat, mdl, h_z, verbose=False):
    
    """
    Adaptive-scale CUSUM on latent-dynamics + transition log-likelihoods.

    Definitions:
        log P(xₜ | xₜ₋₁, regime): measures how well the latent dynamics explain the observed state.
        log P(regime | xₜ₋₁): measures how likely the regime is given the past state.
        The sum is the joint log-probability; a complete measure of model fit at time t

    sₜ is the log-ratio of how well the alternative regime explains the current state vs the current regime.
    
    At each time t:
        sₜ = log-likelihood ratio between alternative and current regime:
            sₜ = [log P(xₜ | xₜ₋₁, alt regime) + log P(alt regime | xₜ₋₁)]
                − [log P(xₜ | xₜ₋₁, curr regime) + log P(curr regime | xₜ₋₁)]
    
        zₜ = (sₜ − μₜ) / σₜ        # z-score of sₜ
        Sₜ = max(0, Sₜ₋₁ + zₜ)     # cumulative sum of z-scores

    If Sₜ > h_z, a regime change is triggered.

    Reasonable ranges for h_z. High sensitivity: 2-3. Low sensitivity: 5-7
    
    """
    from scipy.special import logsumexp
    import numpy as np
    import matplotlib.pyplot as plt

    if xhat.shape[0] <= 10: 
        # print(f'skipping CUSUM: time series too short {xhat.shape[0]}')
        return None  # Skip CUSUM if time series too short
        
    # Ensure shapes
    y = np.atleast_2d(y)
    xhat = np.atleast_2d(xhat)
    if y.shape[0] < y.shape[1]: y = y.T
    if xhat.shape[0] < xhat.shape[1]: xhat = xhat.T
    T, D = xhat.shape

    # Model parameters
    A = mdl.dynamics.As
    b = mdl.dynamics.bs
    sig2 = mdl.dynamics.sigmasq
    Rw  = mdl.transitions.Rs
    r   = mdl.transitions.r

    # Storage
    z_hat = np.zeros(T, dtype=int)
    z_hat[0] = 1  # start in regime 1
    S_arr, z_arr = [], []
    p_new_series = []

    # Online stats
    mu = 0.0
    m2 = 0.0
    S = 0.0

    for t in range(1, T):
        k_curr = z_hat[t - 1]
        k_alt = 1 - k_curr
        xp = xhat[t - 1]
        xn = xhat[t]

        eps_c = xn - (A[k_curr] @ xp + b[k_curr])
        eps_a = xn - (A[k_alt]  @ xp + b[k_alt])

        ll_c = -0.5 * np.sum(eps_c**2 / sig2[k_curr] + np.log(2*np.pi*sig2[k_curr]))
        ll_a = -0.5 * np.sum(eps_a**2 / sig2[k_alt ] + np.log(2*np.pi*sig2[k_alt ]))

        logits = Rw @ xp + r
        logits -= logsumexp(logits)

        s = (ll_a + logits[k_alt]) - (ll_c + logits[k_curr])

        # Posterior for plotting
        p_new = 1.0 / (1.0 + np.exp(ll_c - ll_a))
        p_new_series.append(p_new)

        # Online z-score
        δ = s - mu
        mu += δ / t
        m2 += δ * (s - mu)
        var = m2 / max(t - 1, 1)
        std = np.sqrt(var + 1e-12)

        z_score = (s - mu) / std
        S = max(0.0, S + z_score)

        S_arr.append(S)
        z_arr.append(z_score)

        if S > h_z:
            z_hat[t] = k_alt
            S = 0.0
            mu = 0.0
            m2 = 0.0
        else:
            z_hat[t] = k_curr

    # Δn estimate
    S_arr = np.asarray(S_arr)
    pos_inc = np.maximum(np.diff(np.concatenate(([0.0], S_arr))), 0.0)
    avg_inc = np.convolve(pos_inc, np.ones(10) / 10, mode="same")
    numer = np.maximum(h_z - S_arr, 0.0)
    denom = avg_inc
    Delta_n = np.full_like(S_arr, np.inf, dtype=float)
    np.divide(numer, denom, out=Delta_n, where=(denom > 0))
    pad = len(prices) - len(Delta_n) - 1
    if pad > 0:
        Delta_n = np.concatenate([Delta_n, np.full(pad, np.nan)])

    # -------- plots --------
    if verbose:
        t_idx = prices.index[1:]
        fig, ax = plt.subplots(6, 1, figsize=(9, 6), sharex=True)

        ax[0].plot(t_idx, p_new_series, label="P(new regime)")
        ax[0].set_ylabel("Posterior"); ax[0].legend(); ax[0].grid(True)

        ax[1].plot(t_idx, S_arr, label="CUSUM $S_t$")
        ax[1].axhline(h_z, color="red", linestyle="--", label="Threshold $h_z$")
        ax[1].legend(); ax[1].grid(True)

        ax[2].step(prices.index, z_hat, where='post', label=r"$\hat{z}_t$")
        ax[2].legend(); ax[2].grid(True)

        ax[3].plot(prices.index, prices.values, label="Price")
        ax[3].legend(); ax[3].grid(True)

        ax[4].plot(t_idx, Delta_n, label="Δn estimate")
        ax[4].set_yscale("log"); ax[4].legend(); ax[4].grid(True)

        ax[5].plot(prices.index, y[:, 0] if y.ndim == 2 else y, label="Observed $y_t$")
        ax[5].legend(); ax[5].grid(True)

        plt.tight_layout()

    return z_hat
    
