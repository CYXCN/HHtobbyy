## Issues Fixed

| Item | HHtobbyy original | Fixed (matching original PrepareInputs) |
|------|-------------------|----------------------------------------|
| Lumi (2024) | 109.08 fb⁻¹ | 109.95 fb⁻¹ |
| ggHH xsec (Run3) | 34.43 × 0.0026 | 0.034170e3 × 0.00227 × 0.576 × 2 |
| Selection | `resolved_BDT_mask` (extra b-jet/geometry/trigger cuts) | mass 100–180 + mvaID > −0.7 |
| Standardization | `snt_standardization_fold0.json` (old data mean/std) | Original training `mean_std_dict.pkl` |
| Variable naming | `bTagWPL` → `bTagWPM` ... | `btag_WP_L` → `btag_WP_M` ... (naming only) |

> bTagWP thresholds are identical for 2024 (both use `btagUParTAK4B` with same values).

## Files

```
verification/
├── config.py                    # paths, lumi, xsec, variable name mapping
├── step1_prepare_predict.py     # calls HHtobbyy add_vars_resolved → original selection/std/model → output
├── step2_compare.py             # compares SnT output vs original output
├── step3_merge_and_yield.py     # merge scored parquet + run yield check
├── check_yield_noboost.py       # yield checker (no is_boosted)
└── README.md
```

## Usage

### Step 1: Generate datasets and predict

```bash
conda activate b_hive
cd HHtobbyy/src/HHtobbyy/verification

# All MC + Data
python step1_prepare_predict.py

# MC only
python step1_prepare_predict.py --mc-only

# Single sample
python step1_prepare_predict.py --sample GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00
```

Output goes to `OUTPUT_DIR` defined in `config.py`.

### Step 2: Compare with original

```bash
python step2_compare.py
```

### Step 3: Yield check

```bash
# Merge SnT output into scored parquet
python step3_merge_and_yield.py

# Run yield check
python check_yield_noboost.py \
  /eos/cms/store/group/phys_higgs/nonresonant_HH/PrivateProd/Yuxiang/manos_bbgg_verify/Thomas_framework/scored_samples/merged/merged_scored_events.parquet _thomas

python3 check_yield_noboost.py /eos/cms/store/group/phys_higgs/nonresonant_HH/PrivateProd/Yuxiang/manos_bbgg_verify/scored_samples/merged/merged_scored_events.parquet _snt
```

## Approach

Step 1 calls HHtobbyy's core preprocessing function `add_vars_resolved()`, then applies the original selection, standardization, and model weights. Step 2 compares the output to confirm numerical identity.
