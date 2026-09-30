#!/usr/bin/env python3
"""
McDonald & Kreitman's (1991) test applied to the Adh coding region, using
this project's DGRP sample (inputs/adh/dgrp_adh_205lines.fasta: 205 lines,
176 of them with sequence data at Adh)
in place of their original 11 D. melanogaster sequences, to check whether
their result (excess replacement fixations vs. D. simulans/D. yakuba,
interpreted as adaptive protein evolution) holds with far more polymorphism
data. Divergence sequences are real NCBI RefSeq/GenBank orthologs (fetched
once and cached under data/real-adh-cache/mk-test/), not the original 1991
sequences, so the comparison is not a literal replication -- see the
Discussion/Appendix text of the paper for caveats (in particular: Kreitman's
original 11 sequences were sampled from geographically diverse populations
across several continents, while the DGRP lines come from one North Carolina
population, which is itself a difference in sampling scheme, not just n).

Accessions (chr2L, Adh CDS = join of the 3 exons listed):
  D. melanogaster: NCBI RefSeq NT_033779.5:14615552-14618902 (Release 6),
    CDS join(788..886,952..1356,1427..1693) relative to that window --
    the same genomic window used throughout this paper.
  D. simulans:      NCBI NC_052520.2:14437167-14438898 (Prin_Dsim_3.1),
    CDS join(813..911,979..1383,1450..1716).
  D. yakuba:         NCBI NC_052527.2:14861245-14863135 (Prin_Dyak_Tai18E2_2.1),
    CDS join(834..932,997..1401,1467..1733).
All three CDS's are 771bp (99+405+267), translating cleanly to a single
256-aa protein plus stop, confirming orthology and correct reading frame.

Classification follows McDonald & Kreitman's site-based convention: each
segregating (or divergent) nucleotide site is classified once, as
synonymous or replacement, by substituting the alternate base into the
D. melanogaster reference codon context and comparing the resulting amino
acid. A site is counted as "fixed" divergence only if it is NOT segregating
within the DGRP sample -- a site polymorphic in D. melanogaster is
excluded from Fixed and counted only under Polymorphic, regardless of
whether an outgroup also differs there (this is the mechanism by which a
larger within-species sample can move a site out of the "fixed" category
that a smaller sample missed).
"""
import json
import urllib.request
import os
from pathlib import Path

from Bio import SeqIO
from Bio.Seq import Seq
from scipy.stats import fisher_exact, chi2_contingency

HERE = Path(__file__).resolve().parent
CACHE = HERE / "data" / "real-adh-cache" / "mk-test"
DGRP_FASTA = HERE / "inputs" / "adh" / "dgrp_adh_205lines.fasta"
OUT_CSV = HERE.parent / "figures" / "mcdonald_kreitman_test.csv"
OUT_TEX = HERE.parent / "figures" / "tab_mk_test.tex"

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

SPECIES = {
    "dmel": dict(acc="NT_033779.5", start=14615552, stop=14618902,
                 exons=[(788, 886), (952, 1356), (1427, 1693)]),
    "dsim": dict(acc="NC_052520.2", start=14437167, stop=14438898,
                 exons=[(813, 911), (979, 1383), (1450, 1716)]),
    "dyak": dict(acc="NC_052527.2", start=14861245, stop=14863135,
                 exons=[(834, 932), (997, 1401), (1467, 1733)]),
}


def fetch_fasta(tag, spec):
    path = CACHE / f"{tag}_adh_region.fasta"
    if path.exists():
        return path
    CACHE.mkdir(parents=True, exist_ok=True)
    url = (f"{EUTILS}?db=nuccore&id={spec['acc']}&rettype=fasta&retmode=text"
           f"&seq_start={spec['start']}&seq_stop={spec['stop']}&strand=1")
    urllib.request.urlretrieve(url, path)
    return path


def load_cds(fasta_path, exon_ranges):
    seq = str(next(SeqIO.parse(fasta_path, "fasta")).seq).upper()
    return "".join(seq[a - 1:b] for a, b in exon_ranges)


def classify(pos0, alt_base, ref_cds):
    codon_start = (pos0 // 3) * 3
    ref_codon = ref_cds[codon_start:codon_start + 3]
    offset = pos0 - codon_start
    alt_codon = ref_codon[:offset] + alt_base + ref_codon[offset + 1:]
    same_aa = str(Seq(ref_codon).translate()) == str(Seq(alt_codon).translate())
    return "syn" if same_aa else "repl"


def main():
    cds = {}
    for tag, spec in SPECIES.items():
        path = fetch_fasta(tag, spec)
        cds[tag] = load_cds(path, spec["exons"])
    n = len(cds["dmel"])
    assert len(cds["dsim"]) == len(cds["dyak"]) == n == 771

    for tag, seq in cds.items():
        protein = str(Seq(seq).translate())
        assert protein.count("*") == 1 and protein.endswith("*"), (tag, protein)

    cds_to_local = []
    for a, b in SPECIES["dmel"]["exons"]:
        cds_to_local.extend(range(a, b + 1))

    dgrp_all = [str(r.seq).upper() for r in SeqIO.parse(DGRP_FASTA, "fasta")]
    # lines with no sequence data at Adh (entirely N) carry no information;
    # dropping them leaves the same lines the rest of the Adh analysis uses
    dgrp_seqs = [s for s in dgrp_all if s.count("N") < len(s)]
    n_lines = len(dgrp_seqs)
    print(f"DGRP sample: {len(dgrp_all)} lines in the alignment, {n_lines} with sequence data at Adh")

    poly_class = {}
    for i in range(n):
        local1 = cds_to_local[i]
        col = [s[local1 - 1] for s in dgrp_seqs]
        alleles = set(b for b in col if b in "ACGT")
        ref_base = cds["dmel"][i]
        non_ref = alleles - {ref_base}
        if len(alleles) <= 1 or not non_ref:
            poly_class[i] = None
            continue
        classes = {classify(i, b, cds["dmel"]) for b in non_ref}
        poly_class[i] = "repl" if "repl" in classes else "syn"

    Pn = sum(1 for v in poly_class.values() if v == "repl")
    Ps = sum(1 for v in poly_class.values() if v == "syn")

    Dn, Ds = 0, 0
    excluded = []
    for outgroup in ("dsim", "dyak"):
        for i in range(n):
            if cds["dmel"][i] == cds[outgroup][i]:
                continue
            if poly_class.get(i) is not None:
                excluded.append((outgroup, i))
                continue
            cls = classify(i, cds[outgroup][i], cds["dmel"])
            if cls == "repl":
                Dn += 1
            else:
                Ds += 1

    table = [[Dn, Ds], [Pn, Ps]]
    odds, p_fisher = fisher_exact(table)
    chi2, p_chi2, _, _ = chi2_contingency(table)
    alpha = 1 - (Ds * Pn) / (Dn * Ps) if Dn and Ps else float("nan")

    print(f"\n2x2 table (DGRP n={n_lines}, D.mel vs D.sim+D.yak combined):")
    print(f"           Repl  Syn")
    print(f"Fixed(D)   {Dn:4d}  {Ds:4d}")
    print(f"Poly(P)    {Pn:4d}  {Ps:4d}")
    print(f"Fisher exact p={p_fisher:.4g}, odds ratio={odds:.3f}")
    print(f"chi2 (Yates) p={p_chi2:.4g}")
    print(f"alpha={alpha:.3f}")
    print(f"\nsites excluded from Fixed because polymorphic in DGRP n={n_lines}: "
          f"{len(set(i for _, i in excluded))}")
    for outgroup, i in excluded:
        print(f"  codon {i // 3 + 1} vs {outgroup}: dmel={cds['dmel'][i]} "
              f"{outgroup}={cds[outgroup][i]}  within-dmel class={poly_class[i]}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w") as f:
        f.write("category,repl,syn\n")
        f.write(f"fixed,{Dn},{Ds}\n")
        f.write(f"polymorphic,{Pn},{Ps}\n")
        f.write(f"# fisher_exact_p,{p_fisher:.6g}\n")
        f.write(f"# odds_ratio,{odds:.6g}\n")
        f.write(f"# alpha,{alpha:.6g}\n")
    print(f"\nwrote {os.path.relpath(OUT_CSV)}")

    # tab:mk-test body; the right-hand columns are McDonald & Kreitman's (1991)
    # published counts, reproduced for comparison
    tex = ["\\begin{tabular}{lrrrr}", "\\toprule",
           f" & \\multicolumn{{2}}{{c}}{{Ours (DGRP $n{{=}}{n_lines}$)}} & "
           "\\multicolumn{2}{c}{\\citeauthor{mcdonaldkreitman1991} ($n{=}6$--11)} \\\\",
           " & Repl. & Syn. & Repl. & Syn. \\\\", "\\midrule",
           f"Fixed        & {Dn} & {Ds} & 7 & 17 \\\\",
           f"Polymorphic  & {Pn} & {Ps} & 2 & 42 \\\\",
           "\\bottomrule", "\\end{tabular}"]
    OUT_TEX.write_text("\n".join(tex) + "\n")
    print(f"wrote {os.path.relpath(OUT_TEX)}")


if __name__ == "__main__":
    main()
