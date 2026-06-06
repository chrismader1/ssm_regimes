# _utils.py

import numpy as np


# ---------------------------------------------------------------
# Internal: logsumexp and multivariate Gaussian log-pdf
# ---------------------------------------------------------------

def _lse(a, axis=None, keepdims=False):
    a = np.asarray(a, dtype=float)
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    if not keepdims:
        if axis is not None:
            out = np.squeeze(out, axis=axis)
        else:
            out = float(np.squeeze(out))
    return out


def _mvn_logpdf(y, mu, Sigma):
    """
    Multivariate Gaussian log pdf, with jitter on Sigma for numerical stability.
    y, mu: (N,); Sigma: (N, N).
    """
    N = y.shape[0]
    # symmetrize + jitter
    S = 0.5 * (Sigma + Sigma.T) + 1e-10 * np.eye(N)
    L = np.linalg.cholesky(S)
    diff = y - mu
    z = np.linalg.solve(L, diff)
    return -0.5 * N * np.log(2.0 * np.pi) - np.sum(np.log(np.diag(L))) - 0.5 * np.sum(z**2)
