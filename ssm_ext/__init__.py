"""
ssm_ext — Regime-model extensions to ssm.

Public API:
  fit:        fit_rSLDS, fit_rSLDS_restricted, fit_SLDS, fit_SLDS_restricted,
              fit_HMM, fit_AR_HMM
  inference:  inference_rSLDS, inference_SLDS, causal_cpll_SLDS,
              inference_HMM, inference_ARHMM, filter_states_causal
  evaluate:   evaluate_regimes_synthetic, evaluate_regimes_actual
  bounds:     max_cpll_causal_bound, compute_stability_margin
  params:     print_rSLDS_matrices, get_rSLDS_params, get_rSLDS_args,
              compute_required_w, compute_R_r_from_latents,
              compute_R_r_from_soft_probs, infer_params_from_model
  init:       fit_kmeans
"""

from .fit import (
    fit_rSLDS, fit_rSLDS_restricted, fit_rSLDS_restricted_em,
    fit_SLDS, fit_SLDS_restricted,
    fit_HMM, fit_AR_HMM,
)
from .inference import (
    inference_rSLDS, inference_SLDS, causal_cpll_SLDS,
    inference_HMM, inference_ARHMM, filter_states_causal,
)
from .evaluate import (
    evaluate_regimes_synthetic, evaluate_regimes_actual,
    evaluate_regimes_synthetic_kreg,
)
from .bounds import max_cpll_causal_bound, compute_stability_margin
from .params import (
    print_rSLDS_matrices, get_rSLDS_params, get_rSLDS_args,
    compute_required_w, compute_R_r_from_latents, compute_R_r_from_soft_probs,
    infer_params_from_model,
)
from .params import extract_params_records
from .init import fit_kmeans

__all__ = [
    "fit_rSLDS", "fit_rSLDS_restricted",
    "fit_SLDS", "fit_SLDS_restricted",
    "fit_HMM", "fit_AR_HMM",
    "inference_rSLDS", "inference_SLDS", "causal_cpll_SLDS",
    "inference_HMM", "inference_ARHMM", "filter_states_causal",
    "evaluate_regimes_synthetic", "evaluate_regimes_actual",
    "evaluate_regimes_synthetic_kreg",
    "max_cpll_causal_bound", "compute_stability_margin",
    "print_rSLDS_matrices", "get_rSLDS_params", "get_rSLDS_args",
    "compute_required_w", "compute_R_r_from_latents", "compute_R_r_from_soft_probs",
    "infer_params_from_model",
    "extract_params_records",
    "fit_kmeans",
]
