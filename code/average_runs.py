#!/usr/bin/env python3
"""Averages the per-locus tables of real_adh_migrate_scan.py over training
runs: <table>.csv (run 67) and <table>-seed<N>.csv (runs 68-76, written with
--net-seeds). The network columns (theta, s and their 95% interval bounds,
PGINN_pct_s0) are the means over runs; the other columns are taken from run
67, where they are identical. Adds, per network, the range over runs of the
median s_hat (<net>_s_min, <net>_s_max) and the number of runs whose 95%
interval excludes 0 (<net>_excl0_runs; lower end >= 0.5, as in
adh_neutral_cal_summary.py). Optionally writes the tab:real-data body.
  python average_runs.py --table data/real-featlog/real_adh_migrate_scan_featlog.csv \\
      --out-table ../figures/real_adh_migrate_scan_featlog_runs.csv --out-tex ../figures/tab_real_data_featlog.tex"""
import argparse
import csv
import re

import numpy as np

NETS = ["MSE", "PGINN"]
NET_COLS = [f"{n}_{c}" for n in NETS for c in ["theta", "theta_q025", "theta_q975", "s", "s_q025", "s_q975"]] + ["PGINN_pct_s0"]


def run_tables(table, seeds):
    return [table if s == 67 else re.sub(r"(\.\w+)$", f"-seed{s}" + r"\1", table) for s in seeds]


def average(tables):
    """Rows of the run-averaged table (list of dicts, integer columns as int, others as float)."""
    runs = [list(csv.DictReader(open(t))) for t in tables]
    out = []
    for i, base in enumerate(runs[0]):
        assert all(r[i]["locus"] == base["locus"] for r in runs), f"locus order differs at row {i}"
        row = {k: (int(v) if v.lstrip("-").isdigit() else float(v)) for k, v in base.items()}
        for c in NET_COLS:
            if c in base:
                row[c] = float(np.mean([float(r[i][c]) for r in runs]))
        for n in NETS:
            med = np.array([float(r[i][f"{n}_s"]) for r in runs])
            row[f"{n}_s_min"], row[f"{n}_s_max"] = float(med.min()), float(med.max())
            row[f"{n}_excl0_runs"] = int(sum(float(r[i][f"{n}_s_q025"]) >= 0.5 for r in runs))
        out.append(row)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--table", required=True, help="run-67 table; the -seed<N> tables sit next to it")
    ap.add_argument("--net-seeds", type=int, nargs="+", default=list(range(67, 77)))
    ap.add_argument("--out-table", required=True)
    ap.add_argument("--out-tex", default="")
    args = ap.parse_args()
    rows = average(run_tables(args.table, args.net_seeds))
    with open(args.out_table, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out_table} ({len(rows)} loci, {len(args.net_seeds)} training runs)")
    for r in rows:
        print(f"  locus {r['locus']:2d}: " + "; ".join(
            f"{n} theta {r[n + '_theta']:.4f}, s {r[n + '_s']:.2f} [{r[n + '_s_min']:.2f}-{r[n + '_s_max']:.2f}] "
            f"(95% {r[n + '_s_q025']:.2f}-{r[n + '_s_q975']:.2f}), interval excludes 0 in {r[n + '_excl0_runs']} runs"
            for n in NETS))
    if args.out_tex:
        from real_adh_migrate_scan import write_tex
        write_tex(rows, args.out_tex)


if __name__ == "__main__":
    main()
