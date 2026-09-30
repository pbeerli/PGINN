#!/usr/bin/env python3
"""
Real-data selection scan on the DGRP Adh locus (Kreitman's classic locus),
analogous to real_lct_scan.py but adapted to what the four-gamete test
(four_gamete_test.py) actually showed about this data: R_m = 11 minimum
recombination events spread across the 3.3 kb region (breakpoints roughly
every ~300 bp), so a single non-recombining tree for the whole locus is
not defensible. Non-overlapping ~300 bp windows are used instead, sized to
roughly match the empirical recombination-block scale.

Per window, tree building follows (per discussion, not real_lct_scan.py's
UPGMA):
  1. NJ tree from the window alignment (starting topology).
  2. Fix that topology; estimate a GTR+Gamma substitution model and
     re-optimize branch lengths under ML (PAUP*'s lscores with
     userbrlens=no).
  3. Full heuristic search (TBR) starting from that tree, using the
     ML-estimated model, to see whether a better topology exists.
  4. Compare the heuristic-search tree to the NJ tree (log-likelihood
     and topology).
Each window's resulting tree (heuristic-search winner) is then run through
the existing tree.py/Mm.py feature pipeline and the trained PGINN model,
exactly as msprime_scan_singlechrom.py treats single unaveraged trees.

This script only builds and compares trees (stage 1). Feature extraction
and PGINN inference are wired in once tree building is verified against
real PAUP* output.
"""
import argparse
import os
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_FASTA = HERE / "inputs" / "adh" / "dgrp_adh_205lines.fasta"
OUT_DIR = HERE / "data" / "real-adh-cache" / "windows"
PAUP_BIN = os.environ.get("PAUP_BIN", "paup")
ADH_START = 14_615_552
WINDOW_SIZE = 300


def read_fasta(path):
    names, seqs = [], []
    name, chunks = None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name is not None:
                seqs.append("".join(chunks))
            name = line[1:].strip()
            names.append(name)
            chunks = []
        else:
            chunks.append(line.strip())
    if name is not None:
        seqs.append("".join(chunks))
    return names, seqs


def make_windows(seq_len, window_size):
    bounds = []
    start = 0
    while start < seq_len:
        end = min(start + window_size, seq_len)
        bounds.append((start, end))
        start = end
    return bounds


def window_alignment(names, seqs, start, end):
    """Slice [start, end), drop lines fully-N within this window."""
    kept_names, kept_seqs = [], []
    for n, s in zip(names, seqs):
        sub = s[start:end]
        if sub.count("N") < len(sub):
            kept_names.append(n)
            kept_seqs.append(sub)
    return kept_names, kept_seqs


def write_nexus_paup(names, seqs, out_nex, nj_tre, ml_tre, log_file):
    ntax = len(names)
    nchar = len(seqs[0])
    lines = []
    lines.append("#NEXUS")
    lines.append("begin data;")
    lines.append(f" dimensions ntax={ntax} nchar={nchar};")
    lines.append(" format datatype=dna missing=N gap=- interleave=no;")
    lines.append(" matrix")
    for n, s in zip(names, seqs):
        lines.append(f"  '{n}'  {s}")
    lines.append(" ;")
    lines.append("end;")
    lines.append("")
    lines.append("begin paup;")
    lines.append(f" log file={log_file} replace=yes;")
    lines.append(" set autoclose=yes warnreset=no warntree=no increase=auto criterion=likelihood;")
    lines.append(" nj treefile=none showtree=no brlens=yes;")
    lines.append(" lset nst=6 rmatrix=estimate basefreq=empirical rates=gamma shape=estimate pinvar=0 clock=no;")
    lines.append(" lscores 1 / userbrlens=no;")
    lines.append(f" savetrees file={nj_tre} from=1 to=1 brlens=yes format=newick replace=yes;")
    lines.append(" hsearch start=1 swap=tbr;")
    lines.append(" lscores all / userbrlens=no;")
    lines.append(f" savetrees file={ml_tre} brlens=yes format=newick replace=yes;")
    lines.append(" log stop;")
    lines.append(" quit;")
    lines.append("end;")
    Path(out_nex).write_text("\n".join(lines) + "\n")


def run_paup(nex_path):
    result = subprocess.run([PAUP_BIN, "-n", str(nex_path)], capture_output=True, text=True, timeout=1800)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", default=str(DEFAULT_FASTA))
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--only-window", type=int, default=None,
                     help="0-based window index; process only this one (for testing)")
    args = ap.parse_args()

    names, seqs = read_fasta(args.fasta)
    seqs = [s for s in seqs]
    L = len(seqs[0])
    bounds = make_windows(L, args.window_size)
    print(f"{len(bounds)} windows of size {args.window_size} across {L} bp")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for wi, (start, end) in enumerate(bounds):
        if args.only_window is not None and wi != args.only_window:
            continue
        wnames, wseqs = window_alignment(names, seqs, start, end)
        gstart, gend = ADH_START + start, ADH_START + end - 1
        print(f"window {wi}: chr2L:{gstart}-{gend}  ({end-start} bp), {len(wnames)} usable lines")
        if len(wnames) < 4:
            print(f"  skipping window {wi}: too few usable lines ({len(wnames)})")
            continue

        nex_path = OUT_DIR / f"window_{wi:02d}.nex"
        nj_tre = OUT_DIR / f"window_{wi:02d}_nj.tre"
        ml_tre = OUT_DIR / f"window_{wi:02d}_ml.tre"
        log_file = OUT_DIR / f"window_{wi:02d}.log"
        write_nexus_paup(wnames, wseqs, nex_path, nj_tre, ml_tre, log_file)

        result = run_paup(nex_path)
        if result.returncode != 0:
            print(f"  PAUP* FAILED (exit {result.returncode}) on window {wi}")
            print(result.stdout[-3000:])
            print(result.stderr[-2000:])
        else:
            print(f"  PAUP* OK -> {nj_tre.name}, {ml_tre.name}")


if __name__ == "__main__":
    main()
