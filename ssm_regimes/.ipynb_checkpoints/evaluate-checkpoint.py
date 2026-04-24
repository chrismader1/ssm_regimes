"""
ssm_regimes.evaluate
====================

Generic regime-detection evaluation. No trading logic, no plotting.

Two entrypoints:

- evaluate_regimes_synthetic(...) -> (summary, extras)
    For synthetic experiments where z_true is known.
    Returns classification metrics, changepoint detection metrics, stability,
    ELBO diagnostics, mode usage, CPLL. Label-invariant matching.
    `extras` carries zhat_adj, gamma, flipped flag, true_cp, pred_cp, matched,
    unmatched_true, unmatched_pred, lag_list — everything a trading / plotting
    overlay needs without recomputation.

- evaluate_regimes_actual(...) -> summary
    For real-data runs where no ground-truth regimes are available.
    Returns mode usage, ELBO (aggregated across runs), CPLL, stability.

Both functions are pure computation. Downstream wrappers (e.g. in
ssm_regimes_trading) add trading metrics and plots.
"""

import numpy as np
from collections import Counter
from itertools import groupby
from sklearn.metrics import (
    confusion_matrix, precision_score, recall_score, adjusted_rand_score
)

from .bounds import compute_stability_margin


def evaluate_regimes_synthetic(y, xhat, zhat, z_true, elbo, mdl, cpll, max_cpll,
                               label_invariant=True):
    """
    Generic synthetic-data regime evaluation. Pure computation, no plotting.

    Parameters
    ----------
    y : ndarray (T, N)
        Observed data.
    xhat : ndarray (T, D)
        Inferred continuous latents (for rSLDS). For HMM/ARHMM, xhat == y.
    zhat : ndarray (T,)
        Inferred discrete regimes.
    z_true : ndarray (T,)
        Ground-truth discrete regimes.
    elbo : list or None
        ELBO trace from the fit (None if not available).
    mdl : fitted ssm model
        Used for `mdl.expected_states` and `compute_stability_margin`.
    cpll, max_cpll : float
        Causal predictive log-likelihood and zero-info upper bound.
    label_invariant : bool
        If True, flip zhat to maximise accuracy against z_true.

    Returns
    -------
    summary : dict
        Metrics dict (keys match the original evaluate_rSLDS_synthetic output
        minus trading-specific entries, which are added by the trading wrapper).
    extras : dict
        Intermediate quantities the trading/plotting overlay needs:
          zhat_adj, gamma, flipped, true_cp, pred_cp, matched,
          unmatched_true, unmatched_pred, lag_list, min_len, smoothing
    """
    T = len(zhat)

    # ---- Label-invariant matching ----
    flipped = False
    if label_invariant:
        loss1 = np.mean(zhat != z_true)
        loss2 = np.mean((1 - zhat) != z_true)
        loss = min(loss1, loss2)

        acc1 = np.mean(zhat == z_true)
        acc2 = np.mean((1 - zhat) == z_true)
        if acc2 > acc1:
            zhat_adj = 1 - zhat
            accuracy = acc2
            flipped = True
        else:
            zhat_adj = zhat
            accuracy = acc1
    else:
        loss = np.mean(zhat != z_true)
        accuracy = np.mean(zhat == z_true)
        zhat_adj = zhat

    ari = adjusted_rand_score(z_true, zhat_adj)
    conf_mat = confusion_matrix(z_true, zhat_adj, labels=[0, 1])
    prec = precision_score(z_true, zhat_adj, average='macro', zero_division=0)
    rec  = recall_score(z_true, zhat_adj, average='macro', zero_division=0)
    denom = prec + rec
    f1_score = 2 * prec * rec / denom if denom > 1e-8 else 0.0

    # ---- Changepoint detection ----
    true_cp = np.where(np.diff(z_true) != 0)[0]
    pred_cp = np.where(np.diff(zhat_adj) != 0)[0]

    matched = []
    unmatched_true = []
    unmatched_pred = list(pred_cp)
    for cp in true_cp:
        found = False
        for pcp in pred_cp:
            if abs(pcp - cp) <= 5:
                matched.append(abs(pcp - cp))
                if pcp in unmatched_pred:
                    unmatched_pred.remove(pcp)
                found = True
                break
        if not found:
            unmatched_true.append(cp)
    cp_error = np.mean(matched) if matched else None

    # ---- Regime-length smoothing ----
    smoothing = np.mean([len(list(g)) for _, g in groupby(zhat_adj)])

    # ---- Detection lag ----
    lag_list = []
    for t in range(1, T):
        if z_true[t] != z_true[t - 1]:
            true_regime = z_true[t]
            for lag in range(0, T - t):
                if zhat_adj[t + lag] == true_regime:
                    lag_list.append(lag)
                    break
    detection_lag_mean = np.mean(lag_list) if lag_list else np.nan

    # ---- Stability margin ----
    stability_margins, stability_decision = compute_stability_margin(mdl)

    # ---- ELBO diagnostics ----
    if elbo is not None:
        elbo_start = elbo[0]
        elbo_end   = elbo[-1]
        elbo_delta = elbo_end - elbo_start
    else:
        elbo_start = elbo_end = elbo_delta = np.nan

    # ---- Mode usage ----
    mode_usage = dict(Counter(zhat_adj))

    # ---- Align arrays (last prediction can overshoot) ----
    min_len = min(len(y), len(zhat))
    y_c        = y[:min_len]
    zhat_c     = zhat[:min_len]
    zhat_adj_c = zhat_adj[:min_len]
    z_true_c   = z_true[:min_len]
    xhat_c     = xhat[:min_len]

    # ---- Posterior marginals (gamma) ----
    mask = np.ones_like(y_c, dtype=bool)
    gamma, *_ = mdl.expected_states(xhat_c, y_c, mask=mask)
    if label_invariant and flipped:
        gamma = gamma[:, ::-1]

    summary = {
        "loss": loss,
        "accuracy": accuracy,
        "ari": ari,
        "precision": prec,
        "recall": rec,
        "f1_score": f1_score,
        "changepoint_error": cp_error,
        "avg_inferred_regime_length": smoothing,
        "detection_lag_mean": detection_lag_mean,
        "detection_lag_all": lag_list,
        "elbo_start": elbo_start,
        "elbo_end": elbo_end,
        "elbo_delta": elbo_delta,
        "mode_usage": mode_usage,
        "confusion_matrix": conf_mat,
        "n_matched_changepoints": len(matched),
        "unmatched_true_changepoints": unmatched_true,
        "unmatched_pred_changepoints": unmatched_pred,
        "stability_margins": stability_margins,
        "stability_decision": stability_decision,
        "cpll": cpll,
        "max_cpll": max_cpll,
    }

    extras = {
        "zhat_adj": zhat_adj_c,
        "gamma": gamma,
        "flipped": flipped,
        "true_cp": true_cp,
        "pred_cp": pred_cp,
        "matched": matched,
        "unmatched_true": unmatched_true,
        "unmatched_pred": unmatched_pred,
        "lag_list": lag_list,
        "min_len": min_len,
        "y_aligned": y_c,
        "xhat_aligned": xhat_c,
        "zhat_aligned": zhat_c,
        "z_true_aligned": z_true_c,
    }

    return summary, extras


def evaluate_regimes_actual(y, xhat, zhat, elbo, mdl, cpll, max_cpll, model_type):
    """
    Generic actual-data regime evaluation. No z_true, no label matching, no plotting.

    Parameters
    ----------
    y : ndarray (T, N)
    xhat : ndarray (T, D)
    zhat : ndarray (T,)
    elbo : list-of-lists or None
        One list per EM run; aggregated as min(start)/max(end) across runs.
    mdl : fitted ssm model
    cpll, max_cpll : scalar or array-like
        If array-like, best run selected by argmax(cpll).
    model_type : str
        One of "rslds", "hmm", "arhmm". Used by compute_stability_margin.

    Returns
    -------
    summary : dict
    """
    mode_usage = dict(Counter(zhat))
    smoothing = np.mean([len(list(g)) for _, g in groupby(zhat)])

    # ---- ELBO across runs ----
    if elbo is not None and len(elbo):
        elbo_start = float(np.nanmin([run[0]  for run in elbo]))
        elbo_end   = float(np.nanmax([run[-1] for run in elbo]))
        elbo_delta = float(elbo_end - elbo_start)
    else:
        elbo_start = elbo_end = elbo_delta = np.nan

    # ---- CPLL across runs (scalar or list) ----
    cpll_all     = np.atleast_1d(cpll).astype(float)
    max_cpll_all = np.atleast_1d(max_cpll).astype(float)
    L = min(len(cpll_all), len(max_cpll_all))
    if L == 0:
        cpll_best, max_cpll_paired = np.nan, np.nan
    else:
        cpll_all, max_cpll_all = cpll_all[:L], max_cpll_all[:L]
        i_star = int(np.nanargmax(cpll_all))
        cpll_best = float(cpll_all[i_star])
        max_cpll_paired = float(max_cpll_all[i_star])

    stability_margins, stability_decision = compute_stability_margin(mdl, model_type)

    summary = {
        "avg_inferred_regime_length": smoothing,
        "elbo_start (min all runs)": elbo_start,
        "elbo_end (max all runs)": elbo_end,
        "elbo_delta (max all runs)": elbo_delta,
        "mode_usage": mode_usage,
        "stability_margins": stability_margins,
        "stability_decision": stability_decision,
        "cpll (max all runs)": cpll_best,
        "max cpll (proxy bound, paired)": max_cpll_paired,
    }
    return summary
