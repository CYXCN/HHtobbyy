"""
Shared configuration for HHtobbyy verification.
"""
import os, json, pickle
import numpy as np

VERIF_DIR = os.path.dirname(os.path.abspath(__file__))

# ——— Paths ——————————————————————————————————————————————————————
PARQUET_BASE = "/eos/cms/store/group/phys_b2g/HHbbgg/HiggsDNA_parquet"
ORIG_BASE = "/eos/cms/store/group/phys_higgs/nonresonant_HH/PrivateProd/Yuxiang/manos_bbgg_verify"
REF_BASE   = "/eos/cms/store/group/phys_higgs/nonresonant_HH/bbgg/HHbbgg_Run3/Run3HHbbgg_multiclassDNN/storeForFilesFromCorrespondingUAFDirs/HHbbgg_conditional_classifiers_afterPAS/Version_20260222_bTagWPs_revisedNewBaseline_ResNonResggHHVBFHH_xsecWeightInSignal_1D_16-25_stopAt140Epochs_forPreApp"
MODEL_DIR  = f"{REF_BASE}/after_random_search_best1"
OUTPUT_DIR = f"/eos/cms/store/group/phys_higgs/nonresonant_HH/PrivateProd/Yuxiang/manos_bbgg_verify/Thomas_framework"

# ——— Model inputs ————————————————————————————————————————————————
with open(f"{ORIG_BASE}/input_vars.txt") as f:
    INPUT_VARS = json.load(f)
with open(f"{REF_BASE}/mean_std_dict.pkl", "rb") as f:
    _ms = pickle.load(f)
    ORIG_MEAN = _ms["mean"]
    ORIG_STD  = _ms["std_dev"]

# ——— Era & samples ———————————————————————————————————————————————
ERA = "2024"

MC_SAMPLES = {
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00":
        "v6/Run3_2024/sim/GluGlutoHH_kl-1p00_kt-1p00_c2-0p00/nominal/NOTAG_merged.parquet",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000":
        "v6/Run3_2024/sim/VBFHH_CV-1p000_C2V-1p000_C3-1p000/nominal/NOTAG_merged.parquet",
    "GluGluHToGG_M_125":
        "v6/Run3_2024/sim/GluGluHtoGG/nominal/NOTAG_merged.parquet",
    "VBFHToGG_M_125":
        "v6/Run3_2024/sim/VBFHtoGG/nominal/NOTAG_merged.parquet",
    "VHtoGG_M_125": [
        "v6/Run3_2024/sim/WmHtoGG/nominal/NOTAG_merged.parquet",
        "v6/Run3_2024/sim/WpHtoGG/nominal/NOTAG_merged.parquet",
        "v6/Run3_2024/sim/ZHtoGG/nominal/NOTAG_merged.parquet",
    ],
    "ttHtoGG_M_125":
        "v6/Run3_2024/sim/ttHtoGG/nominal/NOTAG_merged.parquet",
}

DATA_PATHS = [
    "v6/Run3_2024/data/DataCEG0_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataCEG1_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataDEG0_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataDEG1_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataEEG0_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataEEG1_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataFEG0_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataFEG1_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataGEG0_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataGEG1_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataHEG0_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/DataHEG1_2024_NOTAG_merged.parquet",
    "v6/Run3_2024/data/Data_Run2024Iv1_EG0_NOTAG_merged.parquet",
    "v6/Run3_2024/data/Data_Run2024Iv1_EG1_NOTAG_merged.parquet",
    "v6/Run3_2024/data/Data_Run2024Iv2_EG0_NOTAG_merged.parquet",
    "v6/Run3_2024/data/Data_Run2024Iv2_EG1_NOTAG_merged.parquet",
]

# ——— Physics constants (matching original PrepareInputs) ————————————————
LUMI = 109.95

XSEC = {
    "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00": 0.034170e3 * 0.00227 * 0.576 * 2,
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000":          0.0019292e3 * 0.00227 * 0.576 * 2,
    "GluGluHToGG_M_125": 52.23e3 * 0.00227,
    "VBFHToGG_M_125":     4.078e3 * 0.00227,
    "VHtoGG_M_125":       None,  # merged: use per-component xsec below
    "ttHtoGG_M_125":      0.5700e3 * 0.00227,
}

# Per-component xsec for merged VH (matches original vh_component_names)
VH_COMPONENT_XSEC = {
    "WmHtoGG": 0.562032e3 * 0.00227,
    "WpHtoGG": 0.880114e3 * 0.00227,
    "ZHtoGG":  0.9361e3 * 0.00227,
}

# Variable name mapping: HHtobbyy → original (only bTagWP differs)
def hh_var_to_orig(hh_name):
    return hh_name.replace("bTagWP", "btag_WP_")


def orig_var_to_hh(orig_name):
    return orig_name.replace("btag_WP_", "bTagWP")
