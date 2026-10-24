#!/usr/bin/env python3
"""Share of each penalty in the training loss at the last epoch, main data
set, seed 67, at the chosen weights: w * penalty / ((1-w) * MSE + w *
penalty), read from the training logs (train_pginn.py prints both terms
already weighted). The weights w are not comparable across penalties
because the penalties differ in scale; this share is."""
import re

LOGS = [("local (PGINN), w=0.1, log-scaled", "data/rev/logs/pginn-{r}-featlog.log"),
        ("exact, w=0.01, linear", "data/rev/logs/exact-{r}-0.01.log"),
        ("exact, w=0.01, log-scaled", "data/rev/logs/exact-featlog-{r}-0.01.log"),
        ("Fisher, w=0.01, log-scaled", "data/rev/logs/fisher-featlog-{r}-0.01.log"),
        ("process pred||true, w=0.001, log-scaled", "data/rev/logs/process-featlog-{r}-0.001.log"),
        ("process true||pred, w=0.005, log-scaled", "data/rev/logs/process-fwd-featlog-{r}-0.005.log")]
PAT = re.compile(r"train [\d.]+ \(mse ([\d.eE+-]+) kl ([\d.eE+-]+)\)")


def main():
    for label, path in LOGS:
        cells = []
        for r in ["10l", "10l-n20", "1l", "1l-n20"]:
            last = [line for line in open(path.format(r=r)) if line.startswith("epoch")][-1]
            mse, pen = map(float, PAT.search(last).groups())
            cells.append(f"{r} {pen / (mse + pen):.2f}")
        print(f"{label:42s} " + "  ".join(cells))


if __name__ == "__main__":
    main()
