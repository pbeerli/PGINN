#!/usr/bin/env python3
"""Cut a region out of the DGRP sequences of the Drosophila Genome Nexus
(Lack et al. 2015, Genetics 199:1229-1241), as for inputs/adh/
dgrp_adh_205lines.fasta (see docs/adh-data.md).

Input: one chromosome-arm tar from
http://pooldata.genetics.wisc.edu/dgrp_sequences.tar.bz2, e.g.
dgrp_Chr2L.tar, which holds one file per line, RAL-<n>_Chr2L.seq: the
whole arm as a single string without header or newlines, position i of the
arm at byte offset i-1, N where the line has no call. The tar is read in
place; nothing is unpacked.

Output: FASTA with one record per line (>RAL-<n>), positions --start to
--end (1-based, inclusive), in the order of the line names. With
--check, compares the output with an existing FASTA of the same region and
reports the number of differing records."""
import argparse
import tarfile


def extract(tar_path, start, end):
    length = end - start + 1
    out = {}
    with tarfile.open(tar_path) as tar:
        for m in tar.getmembers():
            if not (m.isfile() and m.name.endswith(".seq")):
                continue
            name = m.name.rsplit("/", 1)[-1].split("_")[0]
            fh = tar.extractfile(m)
            fh.seek(start - 1)
            seq = fh.read(length).decode("ascii")
            if len(seq) != length:
                raise SystemExit(f"{m.name}: got {len(seq)} bp, expected {length}")
            out[name] = seq
    return dict(sorted(out.items()))


def read_fasta(path):
    seqs, name = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith(">"):
            name = line[1:]
            seqs[name] = []
        elif name:
            seqs[name].append(line)
    return {k: "".join(v) for k, v in seqs.items()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tar", required=True, help="chromosome-arm tar, e.g. dgrp_Chr2L.tar")
    ap.add_argument("--start", type=int, required=True)
    ap.add_argument("--end", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", default=None, help="existing FASTA of the same region to compare with")
    args = ap.parse_args()
    seqs = extract(args.tar, args.start, args.end)
    with open(args.out, "w") as f:
        for name, seq in seqs.items():
            f.write(f">{name}\n{seq}\n")
    n_empty = sum(set(s) == {"N"} for s in seqs.values())
    print(f"wrote {len(seqs)} lines x {args.end - args.start + 1} bp to {args.out}; {n_empty} lines are all N")
    if args.check:
        ref = read_fasta(args.check)
        diff = [k for k in ref if seqs.get(k) != ref[k]]
        missing = sorted(set(ref) ^ set(seqs))
        print(f"check against {args.check}: {len(ref)} records, {len(diff)} differ, "
              f"{len(missing)} names only in one file")


if __name__ == "__main__":
    main()
