"""
Step 1: Generate dataset and predict scores using HHtobbyy preprocessing.

Calls HHtobbyy's add_vars_resolved(), then applies the ORIGINAL selection,
standardization, and model to produce X.npy / y.npy / rel_w.npy.

Usage:
    python step1_prepare_predict.py                # all MC + Data from config
    python step1_prepare_predict.py --sample GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00
    python step1_prepare_predict.py --data-only
"""
import sys, os, importlib.util, json, argparse, pickle
import numpy as np
import awkward as ak
import pyarrow.parquet as pq

# Add HHtobbyy src to path
_HH_SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _HH_SRC not in sys.path:
    sys.path.insert(0, _HH_SRC)

# ── Load HHtobbyy modules directly (bypass broken __init__.py) ──────────────
def _load_mod(name, relpath):
    path = os.path.join(_HH_SRC, relpath)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_retrieval = _load_mod("HHtobbyy.workspace_utils.retrieval_utils",
                       "HHtobbyy/workspace_utils/retrieval_utils.py")
_preproc_u = _load_mod("HHtobbyy.preprocessing.preprocessing_utils",
                       "HHtobbyy/preprocessing/preprocessing_utils.py")

# Patch sys.modules so that `from HHtobbyy.xxx import yyy` works inside add_vars_resolved
import types
for _pkg in ["HHtobbyy", "HHtobbyy.workspace_utils", "HHtobbyy.preprocessing"]:
    if _pkg not in sys.modules:
        sys.modules[_pkg] = types.ModuleType(_pkg)
sys.modules["HHtobbyy.workspace_utils"].retrieval_utils = _retrieval
sys.modules["HHtobbyy.preprocessing"].preprocessing_utils = _preproc_u

_resolved = _load_mod("HHtobbyy.preprocessing.resolved_preprocessing",
                      "HHtobbyy/preprocessing/resolved_preprocessing.py")
add_vars_resolved = _resolved.add_vars_resolved

# ── Import our config ───────────────────────────────────────────────────────
from config import *  # noqa: also imports VH_COMPONENT_XSEC

# ── Original framework imports ──────────────────────────────────────────────
_FW = "/eos/user/y/yucao/program/HHbbgg_conditional_classifiers"
sys.path.insert(0, os.path.join(_FW, "models"))
sys.path.insert(0, os.path.join(_FW, "data"))
import torch, torch.nn as nn
from mlp import MLP


# ── Helpers ─────────────────────────────────────────────────────────────────
def load_sample(filepath, era, sample_name, is_data=False, xsec=None):
    """Load one parquet file, run HHtobbyy preprocessing, return (X, rel_w)."""
    full = f"{PARQUET_BASE}/{filepath}"
    pf = pq.ParquetFile(full)
    X_list, w_list = [], []

    for batch in pf.iter_batches(batch_size=500000):
        ak_batch = ak.from_arrow(batch)

        # Call HHtobbyy add_vars_resolved (= the core framework preprocessing)
        era_tag = f"Run3_{era}/sim/{os.path.basename(os.path.dirname(os.path.dirname(filepath)))}"
        add_vars_resolved(ak_batch, era_tag)

        # Apply ORIGINAL selection (not HHtobbyy's BDT mask)
        mask = ((ak_batch["mass"] > 100) & (ak_batch["mass"] < 180) &
                (ak_batch["lead_mvaID"] > -0.7) & (ak_batch["sublead_mvaID"] > -0.7))
        ak_batch = ak_batch[mask]
        if len(ak_batch) == 0:
            continue

        # Compute event weight (matching original PrepareInputs)
        if is_data:
            w = np.ones(len(ak_batch), dtype=np.float64)
        else:
            _xs = xsec if xsec is not None else XSEC.get(sample_name, 1.0)
            w = ak.to_numpy(ak.fill_none(ak_batch["weight"], np.nan)) * _xs * LUMI

        # Extract 63 model variables (map HHtobbyy names → original names)
        X_cols = []
        for ov in INPUT_VARS:
            hh_name = orig_var_to_hh(ov)
            val = ak_batch[hh_name] if hh_name in ak_batch.fields else ak_batch[ov]
            X_cols.append(np.asarray(ak.fill_none(val, -999.0), dtype=np.float32))
        X = np.column_stack(X_cols).astype(np.float32)

        X_list.append(X)
        w_list.append(w)

    if not X_list:
        return None, None
    return np.concatenate(X_list), np.concatenate(w_list)


def load_data_sample(filepath, era):
    """Load one Data parquet file."""
    return load_sample(filepath, era, "Data", is_data=True)


def standardize_and_predict(X):
    """Standardize with original mean/std, run model, return (X_std, y)."""
    mask = (X < -998); X[mask] = np.nan
    X_std = (X - ORIG_MEAN) / ORIG_STD
    X_std = np.nan_to_num(X_std, nan=-9)

    with open(f"{MODEL_DIR}/params.json") as f:
        bp = json.load(f)
    act_fn = getattr(nn, bp["act_fn_name"])
    model = MLP(63, bp["num_layers"], bp["num_nodes"], 4, act_fn, bp["dropout_prob"])
    state = torch.load(f"{MODEL_DIR}/mlp.pth", weights_only=False, map_location="cpu")
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    X_t = torch.tensor(X_std, dtype=torch.float32)
    preds = []
    for i in range(0, len(X_t), 1024):
        with torch.no_grad():
            yb = nn.functional.softmax(model(X_t[i:i+1024]), dim=1)
            preds.append(yb.cpu().numpy())
    return X_std, np.concatenate(preds)


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", nargs="+", default=list(MC_SAMPLES.keys()))
    p.add_argument("--data-only", action="store_true")
    p.add_argument("--mc-only", action="store_true")
    args = p.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── MC ──────────────────────────────────────────────────────────────
    if not args.data_only:
        for sample in args.sample:
            pq_rel = MC_SAMPLES[sample]
            paths = pq_rel if isinstance(pq_rel, list) else [pq_rel]

            X_parts, w_parts = [], []
            for p_rel in paths:
                # For VH merged sample, use per-component xsec
                xs = None
                if sample == "VHtoGG_M_125":
                    for comp_name, comp_xs in VH_COMPONENT_XSEC.items():
                        if comp_name in p_rel:
                            xs = comp_xs; break
                X_i, w_i = load_sample(p_rel, ERA, sample, xsec=xs)
                if X_i is not None:
                    X_parts.append(X_i); w_parts.append(w_i)

            if not X_parts:
                print(f"SKIP {sample}: no events")
                continue

            X = np.concatenate(X_parts); rel_w = np.concatenate(w_parts)
            X_std, y = standardize_and_predict(X)

            out_dir = f"{OUTPUT_DIR}/individual_samples/{ERA}/{sample}"
            os.makedirs(out_dir, exist_ok=True)
            np.save(f"{out_dir}/X.npy", X_std)
            np.save(f"{out_dir}/y.npy", y)
            np.save(f"{out_dir}/rel_w.npy", rel_w)
            print(f"MC  {ERA}/{sample}: X={X_std.shape} y={y.shape} rel_w={rel_w.sum():.4f}")

    # ── Data ────────────────────────────────────────────────────────────
    if not args.mc_only:
        out_dir = f"{OUTPUT_DIR}/individual_samples_data/{ERA}"
        os.makedirs(out_dir, exist_ok=True)

        X_parts, w_parts = [], []
        for p_rel in DATA_PATHS:
            X_i, w_i = load_data_sample(p_rel, ERA)
            if X_i is not None:
                X_parts.append(X_i); w_parts.append(w_i)

        if X_parts:
            X = np.concatenate(X_parts); rel_w = np.concatenate(w_parts)
            X_std, y = standardize_and_predict(X)
            np.save(f"{out_dir}/X.npy", X_std)
            np.save(f"{out_dir}/y.npy", y)
            np.save(f"{out_dir}/rel_w.npy", rel_w)
            print(f"DATA {ERA}: X={X_std.shape} y={y.shape} rel_w={rel_w.sum():.4f}")
        else:
            print("DATA: no events")

    print(f"\nDone. Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
