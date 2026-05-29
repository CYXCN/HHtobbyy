import pyarrow.parquet as pq
import pyarrow.compute as pc
import pyarrow as pa
import pandas as pd
import numpy as np

# --- mappings ---
SAMPLE_NAME_MAP = {
    "GluGluToHH_kl-1p00_kt-1p00_c2-0p00": "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    "VBFHH_CV-1p000_C2V-1p000_C3-1p000": "VBFHH_CV_1p000_C2V_1p000_C3_1p000",
    "GluGluHToGG": "GluGluHtoGG_M_125",
    "VBFHToGG": "VBHFHtoGG_M_125",
    "VHToGG": "VHtoGG_M_125",
    "ttHToGG": "ttHtoGG_M_125",
    "GGJets": "GGJets",
    "TTGG": "TTGG",
    "DDQCDGJets": "DDQCDGJET",
    "Data": "Data",
}

RESOLVED_SAMPLE_ORDER = [
    "VBHFHtoGG_M_125", "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125",
    "GluGluHtoGG_M_125", "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000", "TTGG", "GGJets", "DDQCDGJET",
]

BOOSTED_SAMPLE_MAP = {
    "DDQCDGJets": "Data-driven QCD", "GGJets": "γγ + jets",
    "GluGluHToGG": "ggH", "TTGG": "tt + γγ",
    "VBFHToGG": "VBF H", "VHToGG": "VH", "ttHToGG": "ttH",
    "Data": "Data",
    "GluGluToHH_kl-1p00_kt-1p00_c2-0p00": "ggHH (signal)",
    "VBFHH_CV-1p000_C2V-1p000_C3-1p000": "VBF HH (signal)",
}

BOOSTED_SAMPLE_ORDER = [
    "Data-driven QCD", "γγ + jets", "ggH", "tt + γγ",
    "VBF H", "VH", "ttH", "Data", "ggHH (signal)", "VBF HH (signal)",
]

# Ref column -> internal category (ref skips cat0=boosted, shifts by 1)
REF_COL_TO_CAT = {
    "vbfhh_cat1": "cat1",  # VBF-like
    "cat1":       "cat2",  # ggHH tight
    "cat2":       "cat3",  # ggHH medium
    "cat3":       "cat4",  # ggHH loose
}
REF_COLS = ["vbfhh_cat1", "cat1", "cat2", "cat3"]

GGHH_SIG_REF = "GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00"
VBFHH_SIG_REF = "VBFHH_CV_1p000_C2V_1p000_C3_1p000"
SINGLE_HIGGS_REF = [
    "GluGluHtoGG_M_125", "VBHFHtoGG_M_125",
    "VHtoGG_M_125", "ttHtoGG_M_125", "BBHto2G_M_125",
]

SR_mass_range = (120, 130)

def load_data(file_path):
    columns = [
        "sample", "year", "mass", "weight_tot",
        "is_VBFHH_sig_score", "is_ggHH_sig_score",
        "is_nonRes_bkg_score", "is_Res_bkg_score",
        "nonResReg_vbfpair_dijet_mass_DNNreg",
    ]
    table = pq.read_table(file_path, columns=columns)
    rw = table.column("weight_tot")
    rw = pc.if_else(pc.is_null(rw), 1.0, rw)
    table = table.set_column(
        table.schema.get_field_index("weight_tot"), "weight_tot", rw)
    print(f"  Loaded {len(table):,} rows")
    return table


def define_regions(table):
    mass = table.column("mass")
    sr = pc.and_(pc.greater_equal(mass, SR_mass_range[0]), pc.less_equal(mass, SR_mass_range[1]))
    sb_low = pc.and_(pc.greater_equal(mass, 100.0), pc.less(mass, 115.0))
    sb_high = pc.and_(pc.greater(mass, 135.0), pc.less_equal(mass, 180.0))
    return {
        "sr": sr, "sb": pc.or_(sb_low, sb_high),
        "full": pc.and_(pc.greater_equal(mass, 100.0), pc.less_equal(mass, 180.0)),
    }

v6_cut = {
  "cat1" : "is_VBFHH_sig_score >= 0.774 & dijet_mass > 80 & dijet_mass < 190",
  "cat2" : "(is_ggHH_sig_score > 0.99292177413867067 & is_nonRes_bkg_score < 0.42667485697897478 & is_Res_bkg_score < 0.90612735576941672)",
  "cat3" : "(is_ggHH_sig_score > 0.68115907008322718 & is_nonRes_bkg_score < 0.00064209200655934 & is_Res_bkg_score < 0.96341029269140221)",
  "cat4" : "(is_ggHH_sig_score > 0.87036774207656309 & is_nonRes_bkg_score < 0.00221775294750575 & is_Res_bkg_score < 0.84544492824676509)",
}

def parse_cut(v6_cut):
    import re
    VBF_cut = None
    ggHH_cuts = {}
    for cat_name, cut_str in v6_cut.items():
        if "is_boosted == 1" in cut_str:
            continue
        
        if "is_VBFHH_sig_score" in cut_str:
            match = re.search(r'is_VBFHH_sig_score >= ([\d.]+)', cut_str)
            if match:
                VBF_cut = float(match.group(1))
            continue
        scores = []
        for prefix in ["is_ggHH_sig_score > ", "is_nonRes_bkg_score < ", "is_Res_bkg_score < "]:
            match = re.search(prefix + r'([\d.]+)', cut_str)
            if match:
                scores.append(float(match.group(1)))
        if len(scores) == 3:
            ggHH_cuts[cat_name] = scores
    
    return VBF_cut, ggHH_cuts


def define_categories(table):
    from collections import OrderedDict as od
    dij_mass = table.column("nonResReg_vbfpair_dijet_mass_DNNreg")
    vbf_score = table.column("is_VBFHH_sig_score")
    gg_score = table.column("is_ggHH_sig_score")
    nonres_score = table.column("is_nonRes_bkg_score")
    res_score = table.column("is_Res_bkg_score")

    VBFHH_cut, ggHH_cuts = parse_cut(v6_cut)

    dijet_mass_low, dijet_mass_high = 80.0, 190.0

    cat0 = pc.equal(dij_mass, -999.0)  # always False, no boosted

    cat1 = pc.and_(pc.greater_equal(vbf_score, VBFHH_cut),
                   pc.and_(pc.greater(dij_mass, dijet_mass_low),
                           pc.less(dij_mass, dijet_mass_high)))

    ggHH_base = pc.and_(pc.and_(pc.greater(dij_mass, dijet_mass_low),
                                pc.less(dij_mass, dijet_mass_high)),
                        pc.less(vbf_score, VBFHH_cut))
    
    ggHH_cat_conds = od()
    for cat, (gg_cut, nonres_cut, res_cut) in ggHH_cuts.items():
        ggHH_cat_conds[cat] = pc.and_(pc.and_(pc.greater(gg_score, gg_cut),
                                              pc.less(nonres_score, nonres_cut)),
                                      pc.less(res_score, res_cut))

    ggHH_cat_defs = od()
    used_cat_conds = []
    for cat, cond in ggHH_cat_conds.items():
        temp = pc.and_(ggHH_base, cond)
        for used_cond in used_cat_conds:
            temp = pc.and_(temp, pc.invert(used_cond))
        ggHH_cat_defs[cat] = temp
        used_cat_conds.append(cond)
    
    final_dict = {"cat0": cat0, "cat1": cat1}
    final_dict.update(ggHH_cat_defs)

    return final_dict


def aggregate(table, categories, region_mask, col_map, by_year=False):
    """Sum weight_tot grouped by sample (and optionally year)."""
    rw = table.column("weight_tot")
    zero = pc.multiply(rw, 0.0)
    agg_data = {"sample": table.column("sample")}
    if by_year:
        agg_data["year"] = table.column("year")
    for col_name, cat_key in col_map.items():
        mask = pc.and_(region_mask, categories[cat_key])
        agg_data[col_name] = pc.if_else(mask, rw, zero)
    cols = list(col_map.keys())
    group_keys = ["sample", "year"] if by_year else ["sample"]
    grouped = pa.table(agg_data).group_by(group_keys).aggregate([(c, "sum") for c in cols])
    result = grouped.to_pandas().set_index(group_keys)
    result.columns = [c.replace("_sum", "") for c in result.columns]
    return result


def build_resolved_table(yields_sr, yields_sb):
    df = yields_sr.copy()
    df.index = df.index.map(lambda x: SAMPLE_NAME_MAP.get(x, x))
    for name in RESOLVED_SAMPLE_ORDER:
        if name not in df.index:
            df.loc[name] = 0.0
    order = [n for n in RESOLVED_SAMPLE_ORDER if n in df.index]
    # Keep Data row (not in SAMPLE_ORDER, add at end before summary)
    if "Data" in df.index:
        order.append("Data")
    df = df.reindex(order)

    # Signal total: vbfhh_cat1 = VBFHH, ggHH cats = GluGlutoHH
    sig_row = {}
    for col in REF_COLS:
        s = VBFHH_SIG_REF if col == "vbfhh_cat1" else GGHH_SIG_REF
        sig_row[col] = df.loc[s, col] if s in df.index else 0.0
    df.loc["Signal total"] = pd.Series(sig_row)

    # Interpolated background = GGJets + DDQCDGJET (data-driven MC estimate)
    interp_row = {}
    for col in REF_COLS:
        interp_row[col] = sum(df.loc[s, col] for s in ["GGJets", "DDQCDGJET"] if s in df.index)
    df.loc["Interpolated background total"] = pd.Series(interp_row)

    # Other background: vbfhh_cat1 = ggHH + singleH; ggHH cats = VBFHH + singleH
    other_row = {}
    for col in REF_COLS:
        sig = GGHH_SIG_REF if col == "vbfhh_cat1" else VBFHH_SIG_REF
        samples = [sig] + SINGLE_HIGGS_REF
        other_row[col] = df.loc[[s for s in samples if s in df.index], col].sum()
    df.loc["Other background total"] = pd.Series(other_row)

    df.loc["Background total"] = df.loc["Interpolated background total"] + df.loc["Other background total"]
    df.loc["Total yield"] = df.loc["Signal total"] + df.loc["Background total"]

    # Blind data in SR
    if "Data" in df.index:
        df.loc["Data"] = np.nan
    if "Data" in yields_sb.index:
        df.loc["N_data_SB"] = yields_sb.loc["Data"]
    return df


REF_RESOLVED = pd.DataFrame({
    "vbfhh_cat1": [0.040,0.003,0.001,0.001,0.004,0.002,0.028,0.006,4.303,2.849,np.nan,0.028,7.158,0.050,7.208,7.236,None,6.000],
    "cat1":       [0.036,0.174,0.462,0.010,0.335,1.817,0.012,0.067,4.337,3.522,np.nan,1.817,6.693,1.029,7.722,9.539,None,9.000],
    "cat2":       [0.116,0.803,0.515,0.047,1.050,2.474,0.018,0.200,35.493,19.800,np.nan,2.474,55.493,2.548,58.041,60.515,None,141.000],
    "cat3":       [0.236,1.599,4.999,0.065,2.142,1.445,0.037,1.357,41.346,11.983,np.nan,1.445,54.686,9.077,63.763,65.208,None,128.000],
}, index=[
    "VBHFHtoGG_M_125","VHtoGG_M_125","ttHtoGG_M_125","BBHto2G_M_125",
    "GluGluHtoGG_M_125","GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00",
    "VBFHH_CV_1p000_C2V_1p000_C3_1p000","TTGG","GGJets","DDQCDGJET",
    "Data",
    "Signal total","Interpolated background total","Other background total",
    "Background total","Total yield","Z_asimov","N_data_SB",
]).round(3)

def print_our_table(df, title="OUR RESOLVED YIELDS"):
    print(f"\n=== {title} (SR: m_γγ in {SR_mass_range} GeV) ===")
    print(df.to_string(float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "  N/A"))


def print_era_summary_table(df):
    print("\n=== PER-ERA SUMMARY TABLE ===")
    with pd.option_context("display.max_rows", 200, "display.max_columns", None, "display.width", 220):
        print(df.to_string(index=False, float_format=lambda x: f"{x:.6f}" if pd.notna(x) else "  N/A"))

def print_comparison(ours, ref, title):
    """Print side-by-side comparison: our yields | reference yields."""
    # Build multi-level columns: (category, ours/ref)
    cols = ours.columns
    data = {}
    for c in cols:
        data[(c, "ours")] = ours[c]
        if c in ref.columns:
            data[(c, "ref")] = [ref.loc[i, c] if i in ref.index else None for i in ours.index]
    comp = pd.DataFrame(data, index=ours.index)
    comp.columns = pd.MultiIndex.from_tuples(comp.columns)

    print(f"\n=== {title} ===")
    print("  (ours | ref)")
    pd.set_option('display.max_rows', 50)
    pd.set_option('display.width', 200)
    print(comp.to_string(float_format=lambda x: f"{x:.3f}" if pd.notna(x) else "  N/A"))
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')


def build_boosted_table(boosted_yields):
    df = boosted_yields.copy()
    df.index = df.index.map(lambda x: BOOSTED_SAMPLE_MAP.get(x, x))
    for name in BOOSTED_SAMPLE_ORDER:
        if name not in df.index:
            df.loc[name] = 0.0
    df = df.reindex([n for n in BOOSTED_SAMPLE_ORDER if n in df.index])
    # Blind data in SR
    if "Data" in df.index:
        df.loc["Data", "sr"] = np.nan
    sig_names = ["ggHH (signal)", "VBF HH (signal)"]
    bkg_names = [s for s in df.index if s not in sig_names and s != "Data"]
    df.loc["Sum of Backgrounds"] = df.loc[[s for s in bkg_names if s in df.index]].sum()
    return df


def asimov_significance(signal, background):
    if pd.isna(signal) or pd.isna(background) or signal <= 0 or background <= 0:
        return 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.sqrt(2.0 * ((signal + background) * np.log1p(signal / background) - signal)))


def build_era_summary_table(yields_sr_by_year, yields_sb_by_year):
    rows = []
    years = sorted(yields_sr_by_year.index.get_level_values("year").unique())
    for year in years:
        sr_year = yields_sr_by_year.xs(year, level="year").copy()
        sb_year = yields_sb_by_year.xs(year, level="year").copy()

        sr_year.index = sr_year.index.map(lambda x: SAMPLE_NAME_MAP.get(x, x))
        sb_year.index = sb_year.index.map(lambda x: SAMPLE_NAME_MAP.get(x, x))

        for name in RESOLVED_SAMPLE_ORDER:
            if name not in sr_year.index:
                sr_year.loc[name] = 0.0
            if name not in sb_year.index:
                sb_year.loc[name] = 0.0

        for category in REF_COLS:
            signal_sample = VBFHH_SIG_REF if category == "vbfhh_cat1" else GGHH_SIG_REF
            other_signal_sample = GGHH_SIG_REF if category == "vbfhh_cat1" else VBFHH_SIG_REF

            interp_samples = ["GGJets", "DDQCDGJET"]
            other_samples = [other_signal_sample] + SINGLE_HIGGS_REF

            s_sr = sr_year.loc[signal_sample, category] if signal_sample in sr_year.index else 0.0
            b_interp_sr = sr_year.loc[[s for s in interp_samples if s in sr_year.index], category].sum()
            b_other_sr = sr_year.loc[[s for s in other_samples if s in sr_year.index], category].sum()
            b_total_sr = b_interp_sr + b_other_sr

            b_interp_sb = sb_year.loc[[s for s in interp_samples if s in sb_year.index], category].sum()
            b_other_sb = sb_year.loc[[s for s in other_samples if s in sb_year.index], category].sum()
            b_total_sb = b_interp_sb + b_other_sb
            n_data_sb = sb_year.loc["Data", category] if "Data" in sb_year.index else np.nan

            rows.append({
                "Era": str(year),
                "Category": category,
                "S_SR": s_sr,
                "B_interp_SR": b_interp_sr,
                "B_other_SR": b_other_sr,
                "B_total": b_total_sr,
                "B_total_SB": b_total_sb,
                "N_data_SB": n_data_sb,
                "Z_asimov": asimov_significance(s_sr, b_total_sr),
            })

    return pd.DataFrame(rows, columns=[
        "Era", "Category", "S_SR", "B_interp_SR", "B_other_SR",
        "B_total", "B_total_SB", "N_data_SB", "Z_asimov",
    ])


REF_BOOSTED = pd.DataFrame({
    # "full":  [48.5, 102, 0.5, 8.8, 0.008, 0.9, 4.2, 188, 0.4, 0.06, 164.885],
    "sr":    [np.nan, 0.49, 0.04, 0.006, 0.002, 0.05, 0.07, np.nan, 0.28, 0.003, 0.658],
    "sb":    [0, 0.7, 0.002, 0.02, 0, 0, 0, 1, 0.003, 0, 0.722],
}, index=BOOSTED_SAMPLE_ORDER + ["Sum of Backgrounds"]).round(3)


def main(file_path, ext=""):
    print("Loading data...")
    table = load_data(file_path)
    # set 'weight_tot' value to 1.0 for data only
    if "Data" in table.column("sample").unique():
        data_mask = pc.equal(table.column("sample"), "Data")
        table = table.set_column(
            table.schema.get_field_index("weight_tot"), "weight_tot",
            pc.if_else(data_mask, 1.0, table.column("weight_tot"))
        )
    regions = define_regions(table)
    categories = define_categories(table)

    print("Aggregating resolved yields...")
    yields_sr = aggregate(table, categories, regions["sr"], REF_COL_TO_CAT)
    yields_sb = aggregate(table, categories, regions["sb"], REF_COL_TO_CAT)

    # --- per-year yields (CSV only) ---
    yields_sr_by_year = aggregate(table, categories, regions["sr"], REF_COL_TO_CAT, by_year=True)
    yields_sb_by_year = aggregate(table, categories, regions["sb"], REF_COL_TO_CAT, by_year=True)

    # boosted
    boosted_sr = aggregate(table, categories, regions["sr"], {"sr": "cat0"})
    boosted_sb = aggregate(table, categories, regions["sb"], {"sb": "cat0"})
    boosted_yields = pd.concat([boosted_sr, boosted_sb], axis=1)

    boosted_sr_by_year = aggregate(table, categories, regions["sr"], {"sr": "cat0"}, by_year=True)
    boosted_sb_by_year = aggregate(table, categories, regions["sb"], {"sb": "cat0"}, by_year=True)
    boosted_by_year = pd.concat([boosted_sr_by_year, boosted_sb_by_year], axis=1)

    resolved_ours = build_resolved_table(yields_sr, yields_sb)

    # try to get extract 2024 results for comparison, if available
    if '2024' in yields_sr_by_year.index.get_level_values("year"):
        print("\nFound 2024 data, attempting to extract 2024 yields for comparison...")
        resolved_2024 = build_resolved_table(
            yields_sr_by_year.xs(2024, level="year"), yields_sb_by_year.xs(2024, level="year"))
        print_our_table(resolved_2024, "OUR RESOLVED YIELDS (2024)")
        resolved_2024.to_csv(f"yield_table_resolved{ext}_2024.csv")
    boosted_ours = build_boosted_table(boosted_yields)
    era_summary = build_era_summary_table(yields_sr_by_year, yields_sb_by_year)

    print_our_table(resolved_ours, "OUR RESOLVED YIELDS")
    print_our_table(boosted_ours, "OUR BOOSTED YIELDS")
    print_era_summary_table(era_summary)
    # print_comparison(resolved_ours, REF_RESOLVED,
    #                  "RESOLVED YIELDS (SR: m_γγ in [115, 135] GeV)")
    # print_comparison(boosted_ours, REF_BOOSTED,
    #                  "BOOSTED YIELDS (full/SR/SB, cat0)")

    resolved_ours.to_csv(f"yield_table_resolved{ext}.csv")
    # boosted_ours.to_csv("yield_table_boosted.csv")
    era_summary.to_csv(f"yield_table_era_summary{ext}.csv", index=False)

    # per-year: pivot to wide format
    def pivot_by_year(df, name_map, sample_order):
        df = df.copy().round(3)
        df.index = df.index.map(lambda x: (name_map.get(x[0], x[0]), x[1]))
        df.index.names = ["sample", "year"]
        pv = df.unstack("year").sort_index(axis=1)
        order = [s for s in sample_order if s in pv.index]
        if "Data" in pv.index and "Data" not in order:
            order.append("Data")
        return pv.reindex(order)

    pivot_by_year(yields_sr_by_year, SAMPLE_NAME_MAP,
                  RESOLVED_SAMPLE_ORDER).to_csv(f"yield_table_resolved_by_year{ext}.csv")
    bst_by_year = pivot_by_year(boosted_by_year, BOOSTED_SAMPLE_MAP, BOOSTED_SAMPLE_ORDER)
    # Blind Data SR
    sr_cols = [c for c in bst_by_year.columns if c[0] == "sr"]
    if "Data" in bst_by_year.index:
        bst_by_year.loc["Data", sr_cols] = np.nan
    # bst_by_year.to_csv(f"yield_table_boosted_by_year{ext}.csv")
    # print("\nSaved: yield_table_resolved.csv, yield_table_boosted.csv")
    print("Per-year (wide): yield_table_resolved_by_year.csv, yield_table_boosted_by_year.csv")
    print("Era summary: yield_table_era_summary.csv")
    return resolved_ours, boosted_ours, era_summary


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        path = sys.argv[1]
        ext = sys.argv[2] if len(sys.argv) > 2 else ""
    else:
        path = '/data/yxcao/HHbbgg_data/manos_data/v6/merged_scored_events.parquet'
        ext = ""
    main(path, ext)
