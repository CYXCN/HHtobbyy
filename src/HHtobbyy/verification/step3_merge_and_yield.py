"""
Step 3: Merge SnT verification output into scored parquet, then compute yields.

Since step1 only saves X.npy / y.npy / rel_w.npy (no events.parquet),
we read mass and dijet_mass from the ORIGINAL events.parquet (identical events, same order).

Usage:
    python step3_merge_and_yield.py
"""
import sys, os, numpy as np, pandas as pd
import pyarrow as pa, pyarrow.parquet as pq, pyarrow.compute as pc

from config import *

# Sample name mapping for scored parquet (matches reference naming)
FF_MAP = {
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": "GluGluToHH_kl-1p00_kt-1p00_c2-0p00",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000": "VBFHH_CV-1p000_C2V-1p000_C3-1p000",
    "GluGluHToGG_M_125": "GluGluHToGG",
    "VBFHToGG_M_125": "VBFHToGG",
    "VHtoGG_M_125": "VHToGG",
    "ttHtoGG_M_125": "ttHToGG",
}


def main():
    out_dir = f"{OUTPUT_DIR}/scored_samples/merged"
    os.makedirs(out_dir, exist_ok=True)
    out_path = f"{out_dir}/merged_scored_events.parquet"

    tables = []
    total = 0

    # ── MC ──────────────────────────────────────────────────────────────
    for sample in MC_SAMPLES:
        snt_d = f"{OUTPUT_DIR}/individual_samples/{ERA}/{sample}"
        orig_d = f"{ORIG_BASE}/individual_samples/{ERA}/{sample}"

        if not os.path.exists(f"{snt_d}/y.npy"):
            continue

        y = np.load(f"{snt_d}/y.npy")
        rel_w = np.load(f"{snt_d}/rel_w.npy")
        orig_events = pq.read_table(f"{orig_d}/events.parquet",
                                    columns=["mass", "nonResReg_vbfpair_dijet_mass_DNNreg"])

        display = FF_MAP.get(sample, sample)
        df = pd.DataFrame({
            "sample": np.full(len(y), display),
            "year": np.full(len(y), 2024, dtype=np.int64),
            "mass": orig_events.column("mass").to_numpy(),
            "nonResReg_vbfpair_dijet_mass_DNNreg":
                orig_events.column("nonResReg_vbfpair_dijet_mass_DNNreg").to_numpy(),
            "rel_xsec_weight": rel_w.astype(np.float64),
            "weight_tot": rel_w.astype(np.float64),
            "is_nonRes_bkg_score": y[:, 0].astype(np.float64),
            "is_Res_bkg_score": y[:, 1].astype(np.float64),
            "is_ggHH_sig_score": y[:, 2].astype(np.float64),
            "is_VBFHH_sig_score": y[:, 3].astype(np.float64),
        })
        tables.append(pa.Table.from_pandas(df))
        total += len(df)
        print(f"  MC  {sample}: {len(df)} rows")

    # ── Data ────────────────────────────────────────────────────────────
    data_snt = f"{OUTPUT_DIR}/individual_samples_data/{ERA}"
    data_orig = f"{ORIG_BASE}/individual_samples_data/{ERA}"
    if os.path.exists(f"{data_snt}/y.npy"):
        y = np.load(f"{data_snt}/y.npy")
        orig_events = pq.read_table(f"{data_orig}/events.parquet",
                                    columns=["mass", "nonResReg_vbfpair_dijet_mass_DNNreg"])

        df = pd.DataFrame({
            "sample": np.full(len(y), "Data"),
            "year": np.full(len(y), 2024, dtype=np.int64),
            "mass": orig_events.column("mass").to_numpy(),
            "nonResReg_vbfpair_dijet_mass_DNNreg":
                orig_events.column("nonResReg_vbfpair_dijet_mass_DNNreg").to_numpy(),
            "rel_xsec_weight": np.ones(len(y), dtype=np.float64),
            "weight_tot": np.ones(len(y), dtype=np.float64),
            "is_nonRes_bkg_score": y[:, 0].astype(np.float64),
            "is_Res_bkg_score": y[:, 1].astype(np.float64),
            "is_ggHH_sig_score": y[:, 2].astype(np.float64),
            "is_VBFHH_sig_score": y[:, 3].astype(np.float64),
        })
        tables.append(pa.Table.from_pandas(df))
        total += len(df)
        print(f"  DATA {ERA}: {len(df)} rows")

    merged = pa.concat_tables(tables, promote_options="permissive")
    pq.write_table(merged, out_path)
    print(f"\nMerged {total} rows -> {out_path}")
    print(f"\nNow run:")
    print(f"  python check_yield_noboost.py {out_path}")


if __name__ == "__main__":
    main()
