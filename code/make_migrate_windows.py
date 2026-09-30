#!/usr/bin/env python3
"""
Generate a single migrate-n infile treating the 12 Adh windows as 12
unlinked loci in one dataset (Raleigh/DGRP is one real, unstructured
population -- this is not a migration-rate estimation; migrate is used
here purely for its per-locus Bayesian genealogy sampler, giving a
posterior sample of trees per window instead of a single point-estimate
ML tree). One MPI run handles all 12 loci in parallel internally --
no need for 12 separate infiles/directories.

Each window becomes one locus. Since all 205 DGRP lines that are not
fully-N over the *whole* 3,351 bp locus (176 of them, matching
four_gamete_test.py's filter) are used, each individual's per-locus
concatenated sequence is exactly their original, unsliced Adh sequence --
the window boundaries only need to be declared on the site-count line,
not applied to the sequence data itself.

migrate-n infile format:
  line 1: "<npop> <nloci> <title>"
  line 2: "(s<nsites>)" per locus, concatenated with spaces
  line 3 (per population): "<ninds>   <popname>"
  then one line per individual: 10-char name field (space-padded)
  immediately followed by the full concatenated sequence, no separator.
"""
import argparse
from pathlib import Path

import four_gamete_test as fgt

HERE = Path(__file__).resolve().parent
DEFAULT_FASTA = HERE / "inputs" / "adh" / "dgrp_adh_205lines.fasta"
ADH_START = 14_615_552
WINDOW_SIZE = 300
OUT_PATH = HERE / "data" / "real-adh-cache" / "migrate" / "infile"
NAME_FIELD_WIDTH = 10


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


def drop_fully_missing(names, seqs):
    keep = [i for i, s in enumerate(seqs) if s.count("N") < len(s)]
    return [names[i] for i in keep], [seqs[i] for i in keep]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fasta", default=str(DEFAULT_FASTA))
    ap.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    ap.add_argument("--out", default=str(OUT_PATH))
    ap.add_argument("--four-gamete", action="store_true",
                    help="variable-length loci cut at the four-gamete breakpoints "
                         "(four_gamete_test.compatible_segments) instead of a fixed grid")
    args = ap.parse_args()

    names, seqs = read_fasta(args.fasta)
    names, seqs = drop_fully_missing(names, seqs)
    L = len(seqs[0])
    if args.four_gamete:
        _, positions, columns = fgt.biallelic_segregating_sites(names, seqs)
        incompatible = [(a, b) for a in range(len(positions)) for b in range(a + 1, len(positions))
                        if fgt.four_gamete_incompatible(columns[a], columns[b])]
        bounds, _ = fgt.compatible_segments(positions, incompatible, L)
    else:
        bounds = make_windows(L, args.window_size)
    nloci = len(bounds)
    ninds = len(names)
    print(f"{nloci} loci across {L} bp, {ninds} individuals")

    site_tags = " ".join(f"(s{end - start})" for start, end in bounds)

    title = f"dgrp_adh_{nloci}loci_chr2L_{ADH_START}_{ADH_START + L - 1}"
    lines = [f"1 {nloci} {title}", site_tags, f"{ninds}   RAL_Adh"]
    for n, s in zip(names, seqs):
        if len(n) > NAME_FIELD_WIDTH:
            raise ValueError(f"name {n!r} exceeds the {NAME_FIELD_WIDTH}-char migrate name field")
        lines.append(f"{n:<{NAME_FIELD_WIDTH}}{s}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(lines) + "\n")

    for wi, (start, end) in enumerate(bounds):
        gstart, gend = ADH_START + start, ADH_START + end - 1
        print(f"  locus {wi+1:2d} (window {wi:02d}): chr2L:{gstart}-{gend} ({end-start} bp)")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
