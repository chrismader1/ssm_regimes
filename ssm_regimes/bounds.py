# bounds.py 

import numpy as np


def max_cpll_causal_bound(y, reg_scale=1e-8, debug=False):
    """
    Zero-information upper bound on causal CPLL:
        max_cpll = Σ_{t=1}^{T-1} log N(y_t | ȳ, s²)
    where ȳ and s² are the sample mean and sample variance of y (per dim,
    treated as independent diagonal Gaussian).

    This is the best predictive log-likelihood achievable by a constant
    forecaster that knows only the sample's marginal distribution.
    Strictly a ceiling — uses y's own mean and variance in-sample, which is
    legitimate for a normalisation bound (not a live forecaster).

    Matches the summation range of causal CPLL (t = 1 ... T-1: no predictive
    contribution at t = 0 since there is no prior data to predict from).
    """
    Y = np.asarray(y, dtype=float)
    if Y.ndim == 1:
        Y = Y[:, None]
    T, N = Y.shape
    if T < 2:
        return 0.0

    mu_y = Y.mean(axis=0)                                       # (N,)
    var_y = Y.var(axis=0, ddof=0)                               # (N,)
    reg = float(reg_scale) * (np.mean(np.abs(var_y)) + 1.0)
    var_y = np.clip(var_y, reg, None)

    # per-t log N(y_t | mu_y, diag(var_y)) summed over t = 1 ... T-1
    diffs = Y[1:] - mu_y[None, :]                               # (T-1, N)
    lls = -0.5 * np.sum(diffs**2 / var_y + np.log(2.0 * np.pi * var_y), axis=1)

    if debug:
        print(f"[max_cpll_causal_bound] T={T}, N={N}, mu_y={mu_y}, var_y={var_y}, sum_lls={lls.sum():.3e}")
    return float(lls.sum())


def compute_stability_margin(mdl, model_type):
    """
    Compute per-regime stability margins and apply decision rule.
    
    Check that all regimes are dynamically stable by ensuring 
    spectral radius of each A_k is < 1.

    Decision: 'accept' if all margins > 0, else 'reject'
    For HMM there are no latent dynamics; returns ([], 'n/a').
    """
    if model_type in ("hmm", "arhmm"):
        return np.array([]), 'n/a'
    A_matrices = mdl.dynamics.As  # shape (K, D_latent, D_latent)
    margins = np.array([
        1.0 - max(abs(np.linalg.eigvals(A_k)))
        for A_k in A_matrices
    ])
    decision = 'accept' if np.all(margins > 0) else 'reject'
    return margins, decision
    
    