# The Adh data: where they come from and how to rebuild them

`code/inputs/adh/dgrp_adh_205lines.fasta` holds the *Adh* region of
*Drosophila melanogaster* for the 205 lines of the Drosophila Genetic
Reference Panel (DGRP). It is 3351 bp long: chromosome arm 2L, positions
14,615,552 to 14,618,902 (1-based, inclusive). Twenty-nine lines have no
data in this region (all `N`), which leaves the 176 lines used in the
paper.

## Source

The sequences come from the DGRP release of the Drosophila Genome Nexus
(DGN):

- File: `http://pooldata.genetics.wisc.edu/dgrp_sequences.tar.bz2`
  (5,293,037,710 bytes; server date 2023-03-24; the DGN data page is
  `http://johnpool.net/genomes.html`).
- Citation: Lack JB, Cardeno CM, Crepeau MW, Taylor W, Corbett-Detig RB,
  Stevens KA, Langley CH, Pool JE (2015). The *Drosophila* Genome Nexus: a
  population genomic resource of 623 *Drosophila melanogaster* genomes,
  including 197 from a single ancestral range population. *Genetics*
  199(4):1229-1241. doi:10.1534/genetics.115.174664.
- The DGRP itself: Mackay et al. (2012), *Nature* 482:173-178, and Huang
  et al. (2014), *Genome Research* 24:1193-1208.

The DGN files are consensus sequences built from the DGRP reads by the
DGN pipeline. The DGRP Freeze 2.0 site (`dgrp2.gnets.ncsu.edu`) did not
respond on 2026-09-18 or 2026-10-16.

## File format

`dgrp_sequences.tar.bz2` contains one tar file per chromosome arm
(`dgrp_Chr2L.tar`, ...). Each holds one file per line,
`RAL-<n>_Chr2L.seq`: the whole arm as a single string, without header or
newlines, with `N` where the line has no call. Position *i* of the arm is
the character at byte offset *i*-1, so a region is read with one seek.

## Rebuilding the FASTA

```sh
curl -L -o dgrp_sequences.tar.bz2 http://pooldata.genetics.wisc.edu/dgrp_sequences.tar.bz2
tar xjf dgrp_sequences.tar.bz2 dgrp_Chr2L.tar      # about 1 GB per arm; the whole archive is not unpacked
python code/extract_dgrp_region.py --tar dgrp_Chr2L.tar \
    --start 14615552 --end 14618902 --out dgrp_adh_205lines.fasta \
    --check code/inputs/adh/dgrp_adh_205lines.fasta
```

`--check` compares the result with the file in the repository and reports
how many records differ.

The position of the window was checked when the file was made: the *Adh*
coding sequence, at window positions `join(788..886, 952..1356,
1427..1693)`, translates to the 256-residue Adh protein.

## Accession numbers

Reference and outgroup sequences of the McDonald-Kreitman test
(`code/mcdonald_kreitman_test.py`):

- *D. melanogaster*: RefSeq `NT_033779.5`:14,615,552-14,618,902 (chr2L),
  the same window as above.
- *D. simulans*: `NC_052520.2`:14,437,167-14,438,898 (Prin Dsim 3.1).
- *D. yakuba*: `NC_052527.2`:14,861,245-14,863,135 (Prin Dyak Tai18E2 2.1).

## Adh with its flanks

For the comparison of *Adh* with its neighbors, `reproduce.sh adh-flanks` cuts the region with
10 kb on either side (chr2L 14,605,552 to 14,628,902) and builds the
3-locus migrate infile; set `DGRP_TAR` to `dgrp_Chr2L.tar`. The flanks
alone are cut with:

```sh
python code/extract_dgrp_region.py --tar dgrp_Chr2L.tar --start 14605552 --end 14615551 --out dgrp_adh_up10kb.fasta
python code/extract_dgrp_region.py --tar dgrp_Chr2L.tar --start 14618903 --end 14628902 --out dgrp_adh_down10kb.fasta
```
