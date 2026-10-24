#!/usr/bin/env python3
"""Computation time of the loss variants (reviewer point 1.3), on the
first nested data set of the 10-locus and 1-locus regimes (n=10):
training seconds per epoch for MSE, the local penalty (PGINN), the exact
likelihood, the weighted MSE, the Fisher penalty and the process KL in
both directions (train_pginn.py, 2
threads, 1000 training examples), measured as the wall-time slope between
runs of 5 and 25 epochs so that start-up time cancels; and the time per
example of the maximum-likelihood fit (mle_baseline.fit) and of a network
prediction. Prints a table; writes figures/logs/loss_timing.txt via
reproduce.sh."""
import os
import subprocess
import sys
import tempfile
import time

import numpy as np

import mle_baseline
from compare import load_and_predict

REGIMES = ["10l", "1l"]
EPOCHS = (5, 25)


def variants(D):
    x = f"--x-train {D}/train-sweep-features.npy --y-train {D}/train-sweep-params.npy " \
        f"--x-val {D}/test-sweep-features.npy --y-val {D}/test-sweep-params.npy"
    a = f"--ages-train {D}/train-sweep-ages.npy --ages-val {D}/test-sweep-ages.npy"
    la = f"--ages-train {D}/train-sweep-locusages.npy --ages-val {D}/test-sweep-locusages.npy"
    fi = f"--ages-train {D}/train-sweep-fisher.npy --ages-val {D}/test-sweep-fisher.npy"
    return [("MSE", f"{x} {a} --pginnratio 0"),
            ("PGINN (local)", f"{x} {a} --pginnratio 0.1"),
            ("Exact likelihood", f"{x} {la} --loss exact --pginnratio 0.01"),
            ("Weighted MSE", f"{x} {a} --pginnratio 0 --s-weight 1.5"),
            ("Fisher penalty", f"{x} {fi} --loss fisher --pginnratio 0.01"),
            ("Process KL, p||t", f"{x} {la} --loss process --pginnratio 0.001"),
            ("Process KL, t||p", f"{x} {la} --loss process-fwd --pginnratio 0.001")]


def wall(args, epochs, tmp):
    cmd = [sys.executable, "train_pginn.py", *args.split(), "--epochs", str(epochs), "--threads", "2",
           "--out-model", os.path.join(tmp, "m.pt"), "--out-scaler", os.path.join(tmp, "s.npz")]
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)
    return time.perf_counter() - t0


def main():
    with tempfile.TemporaryDirectory(dir=".") as tmp:
        for r in REGIMES:
            D = f"data/nested/{r}-d1"
            print(f"== {r} (data set 1): training seconds per epoch (1000 examples)")
            for label, args in variants(D):
                t = [wall(args, e, tmp) for e in EPOCHS]
                print(f"  {label:17s} {(t[1] - t[0]) / (EPOCHS[1] - EPOCHS[0]):.3f}")
            la = np.load(f"{D}/predict-sweep-locusages.npy")[:50]
            t0 = time.perf_counter()
            mle_baseline.fit(la)
            print(f"  MLE fit: {1000 * (time.perf_counter() - t0) / len(la):.1f} ms per example")
            x = np.load(f"{D}/predict-sweep-features.npy")
            t0 = time.perf_counter()
            load_and_predict(f"{D}/model/mse-67.pt", f"{D}/model/mse-67.npz", x)
            print(f"  network prediction (incl. loading): {1000 * (time.perf_counter() - t0) / len(x):.3f} ms per example")


if __name__ == "__main__":
    main()
