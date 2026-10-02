"""
Pairwise, optimally superimposed (aligned) RMSD between all frames and
first/second nearest neighbours per frame.

For every pair (i, j) the RMSD is computed AFTER an optimal translational +
rotational superposition of frame j onto frame i (QCP algorithm), i.e.

    RMSD(i, j) = min_{R, t} sqrt( 1/N * sum_k || x_k^(i) - (R x_k^(j) + t) ||^2 )

Then for every frame i:
    d1 = min_{j != i} RMSD(i, j)            (first nearest neighbour)
    d2 = second_min_{j != i} RMSD(i, j)     (second nearest neighbour)
"""

from functools import partial

import numpy as np
import pandas as pd
from MDAnalysis.analysis import rms
from MDAnalysis.analysis.diffusionmap import DistanceMatrix

# IMPORTANT: DistanceMatrix's default metric is rms.rmsd WITHOUT superposition.
# We therefore pass an explicitly aligned metric.
aligned_rmsd = partial(rms.rmsd, center=True, superposition=True)


def compute_aligned_rmsd_matrix(universe, selection="not name H*",
                                start=None, stop=None, step=None,
                                in_memory=True, verbose=True):
    """
    Returns the symmetric (n_frames x n_frames) matrix of pairwise
    optimally-aligned RMSD values in Å, plus the frame indices used.
    """
    # original frame numbers of the analysed frames (kept for the time lags)
    frame_indices = np.arange(universe.trajectory.n_frames)[slice(start, stop, step)]

    if in_memory:
        # DistanceMatrix re-reads the trajectory ~N^2/2 times; from RAM this is
        # fast, from a compressed .xtc on disk it is very slow.
        # NOTE: this modifies the universe in place (only the sliced frames
        # are kept afterwards).
        universe.transfer_to_memory(start=start, stop=stop, step=step)
        start = stop = step = None

    dm = DistanceMatrix(universe, select=selection, metric=aligned_rmsd)
    dm.run(start=start, stop=stop, step=step, verbose=verbose)

    D = dm.results.dist_matrix
    return D, frame_indices


def first_second_nearest_neighbours(D, frame_indices=None, dt=None):
    """
    From a pairwise distance matrix D, find for every frame the first and
    second nearest neighbour (self excluded).

    Returns a DataFrame with columns
        frame, d1, nn1, d2, nn2, lag1, lag2, mu (= d2/d1), [lag1_ps, lag2_ps]
    """
    D = np.array(D, dtype=np.float64, copy=True)
    n = D.shape[0]
    if frame_indices is None:
        frame_indices = np.arange(n)

    np.fill_diagonal(D, np.inf)  # exclude the frame itself

    # two smallest entries per row (O(N) per row), then order them
    idx2 = np.argpartition(D, kth=1, axis=1)[:, :2]
    rows = np.arange(n)[:, None]
    order = np.argsort(D[rows, idx2], axis=1)
    idx2 = np.take_along_axis(idx2, order, axis=1)

    j1, j2 = idx2[:, 0], idx2[:, 1]
    d1 = D[np.arange(n), j1]
    d2 = D[np.arange(n), j2]

    with np.errstate(divide="ignore", invalid="ignore"):
        mu = d2 / d1

    df = pd.DataFrame({
        "frame": frame_indices,
        "d1": d1,
        "nn1": frame_indices[j1],
        "d2": d2,
        "nn2": frame_indices[j2],
    })
    df["lag1"] = np.abs(df["nn1"] - df["frame"])   # time lag in frames
    df["lag2"] = np.abs(df["nn2"] - df["frame"])
    df["mu"] = mu
    if dt is not None:
        df["lag1_ps"] = df["lag1"] * dt
        df["lag2_ps"] = df["lag2"] * dt

    n_zero = int(np.sum(d1 <= 0))
    if n_zero:
        print(f"WARNING: {n_zero} frames have d1 = 0 (duplicate frames); "
              "mu = d2/d1 is undefined there.")
    return df


def sanity_check_pair(universe, i, j, D, selection="not name H*"):
    """Recompute RMSD(i, j) independently and compare with the matrix entry."""
    atoms = universe.select_atoms(selection)
    universe.trajectory[i]
    xi = atoms.positions.copy()
    universe.trajectory[j]
    xj = atoms.positions.copy()
    no_fit = rms.rmsd(xi, xj, superposition=False)
    fit = rms.rmsd(xi, xj, center=True, superposition=True)
    print(f"Frames {i}/{j}: RMSD without fit = {no_fit:.4f} Å, "
          f"with fit = {fit:.4f} Å, matrix entry = {D[i, j]:.4f} Å")
    return np.isclose(fit, D[i, j], atol=1e-4)





#NOT SURE IN NEEDED AND NOT CHECKED:
from scipy.stats import kstest

def homogeneity_vs_k(D, ks=(2, 4, 8, 16, 32, 64)):
    r = np.sort(D, axis=1)[:, 1:max(ks) + 1]              # r[:, j-1] = r_j, self dropped
    rows = []
    for k in ks:
        L = np.log(r[:, [k - 1]] / r[:, :k - 1])           # log(r_k / r_j), j < k
        L = L[np.isfinite(L)]                              # drop duplicate frames (r_j = 0)
        d_k = L.size / L.sum()                             # Levina–Bickel MLE at scale k
        u = np.exp(-d_k * L)                               # should be U(0,1)
        rows.append((k, d_k, kstest(u, "uniform").statistic, u))
    return rows

def null_distances(n_rows, kmax, d, rng):
    """Sorted neighbour distances of an exactly locally-Poisson process with Lambda ∝ r^d:
    Lambda(r_j) are arrival times of a unit-rate Poisson process (cumsum of Exp(1))."""
    G = np.cumsum(rng.exponential(size=(n_rows, kmax)), axis=1)
    return np.hstack([np.zeros((n_rows, 1)), G ** (1 / d)]) 


def twonn_mle(mu):
    """Maximum-likelihood TWO-NN dimension: d = N / sum(log mu)."""
    mu = mu[np.isfinite(mu)]            # drop duplicate frames (d1 = 0 -> mu = inf)
    return mu.size / np.log(mu).sum()

def ecdf(x, t):
    """Fraction of values in x that are <= t, for every t in the grid."""
    return np.searchsorted(np.sort(x), t, side="right") / x.size

def x_null_band(n, grid, n_sim=200, seed=0):
    """95% band of ECDF(x)-t and 95th percentile of the KS statistic,
    for datasets of size n where the local assumption holds exactly."""
    rng = np.random.default_rng(seed)
    dev = np.empty((n_sim, grid.size))
    ks = np.empty(n_sim)
    for s in range(n_sim):
        mu = rng.random(n) ** (-1.0)          # exact law with d = 1 (w.l.o.g.)
        x = mu ** (-twonn_mle(mu))            # same pipeline as for the data
        dev[s] = ecdf(x, grid) - grid
        ks[s] = kstest(x, "uniform").statistic
    return np.percentile(dev, [2.5, 97.5], axis=0), np.percentile(ks, 95)

