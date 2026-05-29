"""
Step 2: Compare SnT verification output with original framework output.

Compares X.npy (standardized inputs), rel_w.npy (weights), and y.npy (scores).
For Data, compares event count instead of rel_w (original Data has no weight file).

Usage:
    python step2_compare.py
    python step2_compare.py --sample GluGlutoHHto2B2G_kl_1p00_kt_1p00_c2_0p00
"""
import sys, os, argparse, numpy as np
from config import *

TOL = 1e-5


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample", nargs="+", default=list(MC_SAMPLES.keys()))
    p.add_argument("--snt-dir", default=OUTPUT_DIR)
    p.add_argument("--orig-dir", default=f"{ORIG_BASE}/individual_samples")
    p.add_argument("--orig-data-dir", default=f"{ORIG_BASE}/individual_samples_data")
    return p.parse_args()


def compare_npy(name, snt_path, orig_path):
    """Compare two .npy files. Returns (status, message)."""
    if not os.path.exists(snt_path):
        return "SKIP", f"{name}: SnT file missing"
    if not os.path.exists(orig_path):
        return "SKIP", f"{name}: orig file missing"

    a = np.load(snt_path)
    b = np.load(orig_path)

    if a.shape != b.shape:
        return "FAIL", f"{name}: shape {a.shape} vs {b.shape}"

    mx = np.abs(a - b).max()
    if mx == 0:
        return "PASS", f"{name}: identical ({a.shape})"
    elif mx < TOL:
        return "PASS", f"{name}: max diff {mx:.2e} < {TOL}"
    else:
        n = (np.abs(a - b) > TOL).sum()
        return "FAIL", f"{name}: max diff {mx:.2e}, {n} > tol"


def compare_count(name, snt_path, orig_path):
    """Compare event counts (for Data where orig has no rel_w)."""
    if not os.path.exists(snt_path):
        return "SKIP", f"{name}: SnT file missing"
    if not os.path.exists(orig_path):
        return "SKIP", f"{name}: orig file missing"

    n_snt = len(np.load(snt_path))
    n_orig = len(np.load(orig_path))

    if n_snt == n_orig:
        return "PASS", f"{name}: count match ({n_snt})"
    else:
        return "FAIL", f"{name}: count {n_snt} vs {n_orig}"


def main():
    args = get_args()
    results = []

    # ── MC samples ────────────────────────────────────────────────────
    for sample in args.sample:
        snt_d = f"{args.snt_dir}/individual_samples/{ERA}/{sample}"
        orig_d = f"{args.orig_dir}/{ERA}/{sample}"

        print(f"\n{'='*60}")
        print(f"{ERA}/{sample}")
        print(f"  SnT:  {snt_d}")
        print(f"  Orig: {orig_d}")
        print(f"{'='*60}")

        for fname in ["X.npy", "rel_w.npy", "y.npy"]:
            st, msg = compare_npy(fname, f"{snt_d}/{fname}", f"{orig_d}/{fname}")
            results.append((f"{sample}/{fname}", st, msg))
            if st != "PASS":
                print(f"  [{st}] {msg}")

    # ── Data ───────────────────────────────────────────────────────────
    snt_d = f"{args.snt_dir}/individual_samples_data/{ERA}"
    orig_d = f"{args.orig_data_dir}/{ERA}"
    if os.path.exists(f"{snt_d}/X.npy"):
        print(f"\n{'='*60}")
        print(f"Data {ERA}")
        print(f"{'='*60}")

        st, msg = compare_npy("X.npy", f"{snt_d}/X.npy", f"{orig_d}/X.npy")
        results.append(("Data/X.npy", st, msg))
        if st != "PASS":
            print(f"  [{st}] {msg}")

        st, msg = compare_npy("y.npy", f"{snt_d}/y.npy", f"{orig_d}/y.npy")
        results.append(("Data/y.npy", st, msg))
        if st != "PASS":
            print(f"  [{st}] {msg}")

        # Data: compare event count instead of rel_w (original has no rel_w for Data)
        st, msg = compare_count("rel_w", f"{snt_d}/y.npy", f"{orig_d}/y.npy")
        results.append(("Data/rel_w (event count)", st, msg))
        if st != "PASS":
            print(f"  [{st}] {msg}")

    # ── Summary ────────────────────────────────────────────────────────
    passed = sum(1 for _, s, _ in results if s == "PASS")
    failed = sum(1 for _, s, _ in results if s == "FAIL")
    skipped = sum(1 for _, s, _ in results if s == "SKIP")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for name, st, msg in results:
        if st != "PASS":
            print(f"  [{st}] {name}: {msg}")
    print(f"\n  PASS: {passed}  FAIL: {failed}  SKIP: {skipped}  TOTAL: {len(results)}")
    if failed == 0:
        print("  ALL CHECKS PASSED")
    else:
        print("  SOME CHECKS FAILED")


if __name__ == "__main__":
    main()
