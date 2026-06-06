# params.py

import inspect
import numpy as np
import pandas as pd
import ssm
from scipy.optimize import minimize
import json
from scipy.special import logsumexp


def print_rSLDS_matrices(model, restricted=False, dt=None, latent_names=None):

    print("\n========== FITTED MODEL ==========\n")

    def pmat(arr, label, semantic_key):
        # Shape strings chosen to MATCH the OLD function exactly:
        semantic_shape_str = {
            "Cs": "(K, N, D)",
            "ds": "(K, N)",        # IMPORTANT: match OLD (was (N, K) in the newer draft)
            "inv_etas": "(K, N)",
            "As": "(K, D, D)",
            "bs": "(K, D)",
            "sigmasq": "(K, D)",
            "Rs": "(K, D)",
            "r": "(K,)",
        }[semantic_key]
        # IMPORTANT: two spaces after label to match OLD
        print(f"{label}  {semantic_shape_str} = {arr.shape}")
        K = arr.shape[0]
        if arr.ndim == 3:
            rows = arr.shape[1]
            cols = arr.shape[2]
            for r in range(rows):
                line = ""
                for k in range(K):
                    row_str = " ".join(f"{arr[k, r, c]: .4f}" for c in range(cols))
                    # IMPORTANT: four spaces between blocks to match OLD
                    line += f"[{row_str}]    "
                print(line)
        elif arr.ndim == 2:
            for k in range(K):
                row_str = " ".join(f"{arr[k, c]: .4f}" for c in range(arr.shape[1]))
                # IMPORTANT: four spaces and no trailing newline until after loop, to match OLD
                print(f"[{row_str}]", end="    ")
            print()
        elif arr.ndim == 1:
            row_str = " ".join(f"{arr[k]: .4f}" for k in range(K))
            print(f"[{row_str}]")
        print()

    # Matrix dumps
    pmat(model.emissions.Cs, "Cs", "Cs")
    pmat(model.emissions.ds, "ds", "ds")
    pmat(model.emissions.inv_etas, "inv_etas", "inv_etas")
    pmat(model.dynamics.As, "As", "As")
    pmat(model.dynamics.bs, "bs", "bs")
    pmat(model.dynamics.sigmasq, "sigmasq", "sigmasq")
    pmat(model.transitions.Rs, "Rs", "Rs")
    pmat(model.transitions.r, "r", "r")

    # Regime-type classification
    print("Dynamics: xₜ = a⋅xₜ₋₁ + b + ε")
    print()
    print("Inferred Regime Types:")
    K = model.dynamics.As.shape[0]
    D = model.dynamics.As.shape[1]

    for k in range(K):
        a_vals = np.array([model.dynamics.As[k, d, d] for d in range(D)])
        b_vals = np.array([model.dynamics.bs[k, d] for d in range(D)])

        a_zero = np.allclose(a_vals, 0, atol=1e-2)
        a_one = np.allclose(a_vals, 1.0, atol=1e-2)
        a_ar1 = np.all(np.abs(a_vals) < 1.0)
        b_zero = np.allclose(b_vals, 0.0, atol=1e-4)

        if a_zero and b_zero:
            eqn = "xₜ = ε        (white_noise)"
        elif a_zero and not b_zero:
            eqn = "xₜ = b + ε    (iid_drift)"
        elif a_ar1 and b_zero:
            eqn = "xₜ = a xₜ₋₁ + ε    (ar1)"
        elif a_ar1 and not b_zero:
            eqn = "xₜ = a xₜ₋₁ + b + ε    (ar1_drift)"
        elif a_one and b_zero:
            eqn = "xₜ = xₜ₋₁ + ε    (rw)"
        elif a_one and not b_zero:
            eqn = "xₜ = xₜ₋₁ + b + ε    (rw_drift)"
        else:
            eqn = "Unclassified"

        print(f"Regime {k}:  {eqn}")
    print("\n")

    # If restricted==False, we exactly matched OLD; stop here.
    if not restricted:
        return

    # Otherwise, add interpretable AR(1) parameters for restricted models.
    As = model.dynamics.As  # (K, D, D), assumed diagonal
    bs = model.dynamics.bs  # (K, D)
    sig2 = model.dynamics.sigmasq  # (K, D)
    K, D, _ = As.shape
    eps = 1e-8

    print()
    print("-------------------------------------------------")
    print("Interpretable AR(1) parameters (restricted=True) ")
    print("-------------------------------------------------")
    if latent_names is None:
        latent_names = [f"latent{d}" for d in range(D)]

    for k in range(K):
        print(f"Regime {k}:")
        for d in range(D):
            rho = As[k, d, d]
            b = bs[k, d]
            mu = (b / (1.0 - rho)) if abs(1.0 - rho) > eps else float("nan")
            s2 = sig2[k, d]
            if rho > 0 and abs(rho) < 1:
                hl_steps = float(np.log(2.0) / abs(np.log(rho)))
            else:
                hl_steps = float("inf")
            if (dt is not None and np.isfinite(hl_steps)):
                hl_years = (hl_steps * dt)
            else:
                hl_years = float("nan")
            var_stat = (s2 / (1.0 - rho**2)) if abs(rho) < 1 else float("nan")
            name = latent_names[d] if d < len(latent_names) else f"latent{d}"
            print(
                f"  {name:>10s}: rho={rho: .4f}, mu={mu: .6f}, sigma2={s2: .6e}, "
                f"half_life_steps={hl_steps: .2f}, half_life_years={hl_years: .4f}, "
                f"var_stat={var_stat: .6e}"
            )
        print()


def get_rSLDS_params(model, include_values=False):

    # Example: params_dict = get_model_params(mdl, include_values=False)

    seen = set()
    out = {}

    def visit(obj, prefix=""):
        obj_id = id(obj)
        if obj_id in seen:
            return
        seen.add(obj_id)

        if isinstance(obj, (int, float, str, bool, type(None))):
            if include_values:
                print(f"{prefix}: scalar\n{obj}\n")
            else:
                print(f"{prefix}: scalar")
            out[prefix] = obj

        elif isinstance(obj, np.ndarray):
            if include_values:
                print(f"{prefix}: {obj.shape}\n{obj}\n")
            else:
                print(f"{prefix}: {obj.shape}")
            out[prefix] = obj

        elif isinstance(obj, (list, tuple)):
            print(f"{prefix}: list[{len(obj)}]")
            for i, item in enumerate(obj):
                visit(item, f"{prefix}[{i}]")

        elif isinstance(obj, dict):
            print(f"{prefix}: dict[{len(obj)}]")
            for k, v in obj.items():
                visit(v, f"{prefix}.{k}")

        else:
            for attr in dir(obj):
                if attr.startswith("_") or attr == "params":
                    continue
                try:
                    val = getattr(obj, attr)
                    visit(val, f"{prefix}.{attr}" if prefix else attr)
                except Exception:
                    continue

    def collect_params(component, label):
        if hasattr(component, "parameters"):
            for k, v in component.parameters.items():
                key = f"{label}.{k}"
                if include_values:
                    print(f"{key}: {v.shape}\n{v}\n")
                else:
                    print(f"{key}: {v.shape}")
                out[key] = v

    # Traverse full model
    visit(model)

    # Add labeled parameter blocks
    collect_params(model.transitions, "transitions")
    collect_params(model.dynamics, "dynamics")
    collect_params(model.emissions, "emissions")
    collect_params(model.init_state_distn, "init_state_distn")

    return out


def get_rSLDS_args():
    
    import inspect
    
    sig = inspect.signature(ssm.SLDS.__init__)
    print("SLDS.__init__ parameters:\n")
    for name, param in sig.parameters.items():
        print(f"{name} : {param}")
    
    def print_constructor_params(obj, name, depth=0):
        indent = "  " * depth
        cls = obj.__class__
        print(f"\n{indent}{name} ({cls.__name__}) constructor:")
        sig = inspect.signature(cls.__init__)
        for pname, param in sig.parameters.items():
            if pname != "self":
                print(f"{indent}  {pname}: {param}")
    
            # Check if param is a class itself (e.g. a submodule), recurse
            try:
                attr = getattr(obj, pname)
                if hasattr(attr, '__class__') and not isinstance(attr, (int, float, str, list, tuple, dict, type(None))):
                    print_constructor_params(attr, f"{name}.{pname}", depth + 1)
            except Exception:
                continue  # skip inaccessible attributes
    
    
    # Instantiate a model with valid components
    mdl = ssm.SLDS(N=2, K=2, D=2,
                   transitions="recurrent_only",
                   dynamics="diagonal_gaussian",
                   emissions="gaussian",
                   single_subspace=True)
    
    # Recurse into submodules
    print_constructor_params(mdl, "SLDS")
    print_constructor_params(mdl.transitions, "Transitions")
    print_constructor_params(mdl.dynamics, "Dynamics")
    print_constructor_params(mdl.emissions, "Emissions")


def compute_required_w(sigma, dt, confidence=0.95):

    # Note: compute_required_w is analytical, compute_R_r_from_latents is data-driven.
    
    """
    Compute w such that the softmax transition assigns the correct regime 
    with given probability for a 1-σ move (i.e., signal magnitude = σ√dt).

    Parameters:
    - sigma: std of the signal (e.g., Δlog(price) or Δlog(EPS))
    - dt: time increment
    - confidence: desired classification confidence (e.g., 0.95)

    Returns:why 
    - required value of w
    """
    logit_margin = np.log(confidence / (1.0 - confidence))
    typical = sigma * np.sqrt(dt)
    w = logit_margin / (2.0 * typical)
    return w


def compute_R_r_from_latents(x, z, K=2):
    """
    Fit softmax transition parameters (R, r) from latent data and regime labels.
    """

    from sklearn.linear_model import LogisticRegression

    # Guard against trivial case: all z same
    if len(np.unique(z)) < 2:
        # Force balanced fake labels for logistic fit
        z[:len(z)//2] = 0
        z[len(z)//2:] = 1

    clf = LogisticRegression(fit_intercept=True)
    clf.fit(x, z)
    w = clf.coef_[0]
    b = clf.intercept_[0]

    Rs = np.stack([w, -w])       # shape (K=2, D)
    r = np.array([b, -b])        # shape (K=2,)

    return Rs, r

def compute_R_r_from_soft_probs(x, Ez, K):
    """
    x  : (T-1, D) latent states x_{t-1}
    Ez : (T-1, K) posterior P(z_t = k), we use Ez[:,1] as soft label
    K  : number of regimes (assume 2)
    """
    assert K == 2, "Only supports binary case"
    x = np.asarray(x)
    y_soft = Ez[:, 1]  # target: P(z_t = 1)

    T, D = x.shape
    X_aug = np.hstack([x, np.ones((T, 1))])  # shape (T, D+1)

    def loss(w_aug):
        logits = X_aug @ w_aug       # shape (T,)
        probs = 1 / (1 + np.exp(-logits))
        eps = 1e-8
        ce = - y_soft * np.log(probs + eps) - (1 - y_soft) * np.log(1 - probs + eps)
        return np.mean(ce)

    w0 = np.zeros(D + 1)
    res = minimize(loss, w0, method='L-BFGS-B')
    w_aug = res.x

    w, b = w_aug[:-1], w_aug[-1]
    Rs = np.stack([w, -w])
    r = np.array([b, -b])
    return Rs, r


def infer_params_from_model(model, mu_true, sigma_true, sigma_diff_true, dt=1/252):
    
    """
    Return a DataFrame with estimated and true drift/volatility (continuous time)
    for each regime of a 1-D two-regime rSLDS.

    Parameters
    ----------
    model : rSLDS object (fit with K=2, D=N=1)
    mu_true : float                      # absolute drift used in generator
    sigma_true : float                   # centre volatility in generator
    sigma_diff_true : float              # sigma_up − sigma_down in generator
    dt : float                           # data time step (default 1/252)
    """
    # estimated continuous-time drift & vol -------------------------------
    A  = model.dynamics.As[:, 0, 0]
    b  = model.dynamics.bs[:, 0]
    s2 = model.dynamics.sigmasq[:, 0]

    mu_c_est  = b / (1 - A) / dt
    sig_c_est = np.sqrt(s2 / (1 - A**2)) / np.sqrt(dt)

    # true values as used in generate_synthetic_data ----------------------
    drift_actual = np.array([-mu_true,  mu_true])              # regime 0/1
    vol_actual   = np.array([sigma_true + sigma_diff_true/2,   # regime 0
                             sigma_true - sigma_diff_true/2])  # regime 1

    # pack and return -----------------------------------------------------
    df = pd.DataFrame({
        "drift_est"   : mu_c_est,
        "vol_est"     : sig_c_est,
        "drift_actual": drift_actual,
        "vol_actual"  : vol_actual,
    })
    df = df.set_index(pd.Index([0, 1], name="regime"))
    print(df)


# --- moved from ssm_trading.gridsearch: single source for model param extraction ---
def extract_params_records(model, model_type, K, D, N, batch_id):
    """
    Extract per-regime fitted parameters from a fitted model into a list of
    dicts (one per regime). Identifier columns (security/config/dates/etc) are
    added by the caller; this function only emits the parameter columns.

    Returns: list of K dicts, each with keys
        regime, model_type,
        rho, b, mu, sigmasq, Rs, r, Cs, ds,    (rSLDS-applicable; HMM/AR-HMM -> NaN)
        A_obs, b_obs,                          (AR-HMM-applicable; rSLDS/HMM -> NaN)
        transition_self, log_pi0
    Variable-length values are JSON-encoded strings.
    """
    out = []
    log_pi0_full = np.asarray(model.init_state_distn.log_pi0, dtype=float)
    log_pi0_full = log_pi0_full - logsumexp(log_pi0_full)  # normalised

    if model_type in ("rslds", "rslds_restricted"):
        As       = np.asarray(model.dynamics.As,      dtype=float)   # (K, D, D)
        bs       = np.asarray(model.dynamics.bs,      dtype=float)   # (K, D)
        sigmasq  = np.asarray(model.dynamics.sigmasq, dtype=float)   # (K, D)
        Rs       = np.asarray(model.transitions.Rs,   dtype=float)   # (K, D)
        r_vec    = np.asarray(model.transitions.r,    dtype=float)   # (K,)
        Cs       = np.asarray(model.emissions.Cs,     dtype=float)   # (1,N,D) or (K,N,D)
        ds       = np.asarray(model.emissions.ds,     dtype=float)   # (1,N) or (K,N)
        invE     = np.asarray(model.emissions.inv_etas, dtype=float) # (1,N) or (K,N)
        shared   = (Cs.shape[0] == 1)        # one shared subspace vs K per-regime
        eps = 1e-8
        for k in range(K):
            rho_k = np.diag(As[k])                                   # (D,)
            b_k   = bs[k]                                            # (D,)
            denom = np.where(np.abs(1.0 - rho_k) > eps, 1.0 - rho_k, np.nan)
            mu_k  = b_k / denom                                      # (D,)
            sig_k = sigmasq[k]                                       # (D,)
            Rs_k  = Rs[k]                                            # (D,)
            C_k    = Cs[0]   if shared else Cs[k]
            d_k    = ds[0]   if shared else ds[k]
            invE_k = invE[0] if shared else invE[k]
            out.append({
                "regime":          int(k),
                "model_type":      model_type,
                "rho":             json.dumps(rho_k.tolist()),
                "b":               json.dumps(b_k.tolist()),
                "mu":              json.dumps(mu_k.tolist()),
                "sigmasq":         json.dumps(sig_k.tolist()),
                "Rs":              json.dumps(Rs_k.tolist()),
                "r":               float(r_vec[k]),
                "Cs":              json.dumps(C_k.tolist()),
                "ds":              json.dumps(d_k.tolist()),
                "inv_etas":        json.dumps(invE_k.tolist()),
                "A_obs":           None,
                "b_obs":           None,
                "transition_self": float("nan"),
                "log_pi0":         float(log_pi0_full[k]),
                "log_Ps":          None,
            })
        return out

    if model_type in ("slds", "slds_restricted"):
        As       = np.asarray(model.dynamics.As,      dtype=float)   # (K, D, D)
        bs       = np.asarray(model.dynamics.bs,      dtype=float)   # (K, D)
        sigmasq  = np.asarray(model.dynamics.sigmasq, dtype=float)   # (K, D)
        log_Ps   = np.asarray(model.transitions.log_Ps, dtype=float) # (K, K)
        log_Ps   = log_Ps - logsumexp(log_Ps, axis=1, keepdims=True)
        Cs       = np.asarray(model.emissions.Cs,     dtype=float)
        ds       = np.asarray(model.emissions.ds,     dtype=float)
        invE     = np.asarray(model.emissions.inv_etas, dtype=float)
        shared   = (Cs.shape[0] == 1)        # one shared subspace vs K per-regime
        eps = 1e-8
        for k in range(K):
            rho_k = np.diag(As[k])
            b_k   = bs[k]
            denom = np.where(np.abs(1.0 - rho_k) > eps, 1.0 - rho_k, np.nan)
            mu_k  = b_k / denom
            sig_k = sigmasq[k]
            C_k    = Cs[0]   if shared else Cs[k]
            d_k    = ds[0]   if shared else ds[k]
            invE_k = invE[0] if shared else invE[k]
            out.append({
                "regime":          int(k),
                "model_type":      model_type,
                "rho":             json.dumps(rho_k.tolist()),
                "b":               json.dumps(b_k.tolist()),
                "mu":              json.dumps(mu_k.tolist()),
                "sigmasq":         json.dumps(sig_k.tolist()),
                "Rs":              None,
                "r":               float("nan"),
                "Cs":              json.dumps(C_k.tolist()),
                "ds":              json.dumps(d_k.tolist()),
                "inv_etas":        json.dumps(invE_k.tolist()),
                "A_obs":           None,
                "b_obs":           None,
                "transition_self": float(np.exp(log_Ps[k, k])),
                "log_pi0":         float(log_pi0_full[k]),
                "log_Ps":          json.dumps(log_Ps[k].tolist()),
            })
        return out

    if model_type == "hmm":
        mus      = np.asarray(model.observations.mus,     dtype=float)   # (K, N)
        sigmasq  = np.asarray(model.observations.sigmasq, dtype=float)   # (K, N)
        P        = np.asarray(model.transitions.transition_matrix, dtype=float)  # (K, K)
        for k in range(K):
            out.append({
                "regime":          int(k),
                "model_type":      model_type,
                "rho":             None,
                "b":               None,
                "mu":              json.dumps(mus[k].tolist()),
                "sigmasq":         json.dumps(sigmasq[k].tolist()),
                "Rs":              None,
                "r":               float("nan"),
                "Cs":              None,
                "ds":              None,
                "inv_etas":        None,
                "A_obs":           None,
                "b_obs":           None,
                "transition_self": float(P[k, k]),
                "log_pi0":         float(log_pi0_full[k]),
                "log_Ps":          None,
            })
        return out

    if model_type == "arhmm":
        A_obs    = np.asarray(model.observations.As,      dtype=float)   # (K, N, N)
        b_obs    = np.asarray(model.observations.bs,      dtype=float)   # (K, N)
        sigmasq  = np.asarray(model.observations.sigmasq, dtype=float)   # (K, N)
        P        = np.asarray(model.transitions.transition_matrix, dtype=float)  # (K, K)
        for k in range(K):
            out.append({
                "regime":          int(k),
                "model_type":      model_type,
                "rho":             None,
                "b":               None,
                "mu":              None,
                "sigmasq":         json.dumps(sigmasq[k].tolist()),
                "Rs":              None,
                "r":               float("nan"),
                "Cs":              None,
                "ds":              None,
                "inv_etas":        None,
                "A_obs":           json.dumps(A_obs[k].tolist()),
                "b_obs":           json.dumps(b_obs[k].tolist()),
                "transition_self": float(P[k, k]),
                "log_pi0":         float(log_pi0_full[k]),
                "log_Ps":          None,
            })
        return out

    raise ValueError(f"_extract_params_records: unsupported model_type {model_type!r}")
