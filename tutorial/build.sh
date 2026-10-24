#!/bin/sh
# Build tutorial.pdf. The figures come from run_tutorial.sh (work/genealogy.pdf,
# work/recovery.pdf), so run that first.
set -e
cd "$(dirname "$0")"
[ -s work/genealogy.pdf ] && [ -s work/recovery.pdf ] || { echo "run ./run_tutorial.sh first"; exit 1; }
latexmk -pdf -interaction=nonstopmode -halt-on-error tutorial.tex >/dev/null || { echo "LaTeX failed; see tutorial.log"; exit 1; }
echo "tutorial.pdf built"
