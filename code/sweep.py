#!/usr/bin/env python3
"""
Deterministic hard-sweep rescaling of the single-population coalescent.

Backward time t=0 is the present. The sample is assumed to be drawn just
after a completed (or near-completed) selective sweep, so the beneficial
allele's backward-time frequency trajectory is the logistic

    x(t; s) = x0 / (x0 + (1 - x0) * exp(s * t))

with x0 fixed close to 1 (present-day frequency). s=0 gives x(t)=x0 for all
t (the neutral case); s>0 collapses x(t) toward 0 as t grows, compressing
coalescence into a star-like burst near the sweep's origin.

Here t is coalescent time (units of c generations) and s = c*s_g is the
population-scaled selection coefficient, with theta = c*mu. simtree.py
measures time in expected mutations per site (age = generations * mu =
t*theta; total coalescence rate k(k-1)/theta), so on simtree's time axis
the trajectory runs at rate_per_age(s, theta) = s_g/mu = s/theta.

Effective population size and coalescent rate for k lineages:

    theta_eff(t; theta, s) = theta * x(t; s)
    lambda(t, k)            = k*(k-1) / theta_eff(t; theta, s)

Breakpoints for the piecewise-constant simulator input are spaced
geometrically in ALLELE-FREQUENCY space (not in raw time), inverting the
logistic to get t(x). This makes the discretization self-adapt to whatever s
is drawn: a large s collapses x(t) within a very short t, and fixed
time-spaced breakpoints would either miss the collapse entirely (too coarse)
or waste resolution on a deep tail nothing reaches (too fine in the wrong
place) -- x-spaced breakpoints instead always resolve the transition itself.
"""
import numpy as np

X0 = 0.999      # fixed present-day frequency of the beneficial allele
X_MIN = 1e-4    # deepest allele frequency the schedule resolves down to


def x_trajectory(t, s, x0=X0):
    """Backward-time logistic allele-frequency trajectory, x(t;s)."""
    t = np.asarray(t, dtype=np.float64)
    return x0 / (x0 + (1.0 - x0) * np.exp(s * t))


def t_of_x(x, s, x0=X0):
    """Inverse of x_trajectory: backward time at which x(t;s) == x."""
    return np.log(x0 * (1.0 - x) / (x * (1.0 - x0))) / s


def rate_per_age(s, theta):
    """Logistic rate on simtree's time axis (age = t*theta) for population-scaled s."""
    return s / theta


def theta_eff(t, theta, s, x0=X0):
    """Effective theta at backward time t under selection strength s."""
    return theta * x_trajectory(t, s, x0)


def make_timepoints_file(path, theta, s, n_segments=40, x0=X0, x_min=X_MIN):
    """
    Write a -tp/--timepoints file for lib/simtree.py: n_segments lines of
    "age_i theta_eff(age_i)" (ages on simtree's time axis), with breakpoints x_i spaced geometrically from just
    below x0 down to x_min, then mapped through t(x) -- so resolution always
    sits on the s-dependent collapse itself, however early or late it falls.
    Only meaningful for s > 0 (the s=0 neutral case needs no timepoints file:
    theta_eff is constant at theta*x0 for all t).
    """
    assert s > 0, "make_timepoints_file is only defined for s > 0"
    x_points = np.geomspace(x0 * 0.999, x_min, n_segments)
    t_points = t_of_x(x_points, rate_per_age(s, theta), x0)
    with open(path, "w") as f:
        for t, x in zip(t_points, x_points):
            f.write(f"{t:.10g} {theta * x:.10g}\n")
    return t_points
