import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import poisson
import compute_dnn as nn

def plot_figure1_left(D, nn_df, save_path="fig1_left.png", in_nm=False):
    scale, unit = (0.1, "nm") if in_nm else (1.0, "Å")
    n = D.shape[0]
    fig, (ax_a, ax_c) = plt.subplots(2, 1, figsize=(6, 9))

    # 1a: mean and standard deviation of RMSD(t, t+k) = k-th off-diagonal of D
    lags = np.unique(np.logspace(0, np.log10(n // 2), 60).astype(int))
    means = np.array([np.diagonal(D, k).mean() for k in lags]) * scale
    stds = np.array([np.diagonal(D, k).std() for k in lags]) * scale
    baseline = D[np.triu_indices(n, k=1)].mean() * scale

    ax_a.errorbar(lags, means, yerr=stds, color="0.2", capsize=2)
    ax_a.axhline(baseline, color="crimson", ls="--", label="unrelated frames")
    ax_a.set_xscale("log")
    ax_a.set_xlabel("separation in frames")
    ax_a.set_ylabel(f"mean distance between frames [{unit}]")
    ax_a.set_title("1a  how fast the trajectory forgets")
    ax_a.legend()

    # 1c: query frame t vs. frame t' of its 1st and 2nd nearest neighbour
    t = nn_df["frame"]
    ax_c.scatter(t, nn_df["nn2"], s=2, color="tab:orange", label="2nd NN")
    ax_c.scatter(t, nn_df["nn1"], s=2, color="tab:blue", label="1st NN")
    ax_c.plot([t.min(), t.max()], [t.min(), t.max()], color="crimson", ls="--", label="t' = t")
    ax_c.set_xlabel("query frame t")
    ax_c.set_ylabel("frame t' of its spatial neighbour")
    ax_c.set_title("1c  where in time are the spatial neighbours?")
    ax_c.legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_poisson_tests(D, n_balls=None, save_path="fig2_3_poisson.png", in_nm=False, seed=0):
    scale, unit = (0.1, "nm") if in_nm else (1.0, "Å")
 
    # ball centres: n_balls randomly chosen frames (None = every row of D is a centre)
    if n_balls is not None:
        centres = np.random.default_rng(seed).choice(D.shape[0], size=n_balls, replace=False)
        D = D[centres]
    n = D.shape[0]                                   # number of balls
 
    # median r2 of the centres: column 0 = centre itself (0), 1 = d1, 2 = d2
    r2 = np.median(np.partition(D, 2, axis=1)[:, 2])
    radii = r2 * np.logspace(-0.5, 1, 20)
 
    # K[a, i] = number of other frames within radius radii[a] around centre i (centre excluded)
    K = np.array([(D < R).sum(axis=1) - 1 for R in radii])
    mean_K = K.mean(axis=1)
    fano = K.var(axis=1) / mean_K
    p0_obs = (K == 0).mean(axis=1)
    p0_poisson = np.exp(-mean_K)
    ok = (p0_obs > 0) & (n * p0_poisson >= 1)                   # resolvable with n balls
    band = 1.96 * np.sqrt((1 - p0_poisson) / (n * p0_poisson))  # 95% Poisson null
 
    # counts at R = median r2 for the histogram
    K_r2 = (D < r2).sum(axis=1) - 1
    k = np.arange(K_r2.max() + 1)
 
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 4.5))
    fig.suptitle(f"{n} balls, {D.shape[1]} frames")
 
    # 2a: Fano factor vs. R (Poisson: 1)
    ax_a.plot(radii * scale, fano, "o-", color="0.2", label="data")
    ax_a.axhline(1, color="crimson", ls="--", label="Poisson")
    ax_a.axhspan(1 - 1.96 * np.sqrt(2 / n), 1 + 1.96 * np.sqrt(2 / n), color="crimson", alpha=0.2)
    ax_a.axvline(r2 * scale, color="tab:blue", ls="-.", label="median $r_2$")
    ax_a.set_xscale("log")
    ax_a.set_yscale("log")
    ax_a.set_xlabel(f"ball radius R [{unit}]")
    ax_a.set_ylabel("Fano factor Var[K_R] / E[K_R]")
    ax_a.set_title("2a  dispersion of the ball counts")
    ax_a.legend()
 
    # 2b: distribution of K_R at R = median r2 vs. Poisson with the same mean
    ax_b.bar(k, np.bincount(K_r2) / n, color="tab:blue", alpha=0.6, label="observed")
    ax_b.plot(k, poisson.pmf(k, K_r2.mean()), "o-", color="k", label=f"Poisson(λ={K_r2.mean():.2f})")
    ax_b.set_xlabel("K_R = points in ball (centre excluded)")
    ax_b.set_ylabel("frequency")
    ax_b.set_title(f"2b  counts at R = median $r_2$ = {r2 * scale:.3g} {unit}")
    ax_b.legend()
 
    # 3a: void probability, observed / predicted exp(-E[K_R])
    ax_c.plot(radii[ok] * scale, p0_obs[ok] / p0_poisson[ok], "o-", color="0.2", label="data")
    ax_c.fill_between(radii[ok] * scale, 1 - band[ok], 1 + band[ok], color="crimson", alpha=0.2)
    ax_c.axhline(1, color="crimson", ls="--", label="Poisson")
    ax_c.axvline(r2 * scale, color="tab:blue", ls="-.", label="median $r_2$")
    ax_c.set_xscale("log")
    ax_c.set_yscale("log")
    ax_c.set_xlabel(f"ball radius R [{unit}]")
    ax_c.set_ylabel("observed P(K_R = 0) / exp(-E[K_R])")
    ax_c.set_title("3a  void probability vs. its prediction")
    ax_c.legend()
 
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def plot_homogeneity(D, ks, save_path="fig4_homogeneity.png", n_null=20, seed=0):
    res = nn.homogeneity_vs_k(D, ks)                          # ks[0] must be 2
    d_k, ks_stat = np.array([r[1] for r in res]), np.array([r[2] for r in res])
    rng = np.random.default_rng(seed)
    null = np.array([[r[1:3] for r in nn.homogeneity_vs_k(
        nn.null_distances(D.shape[0], max(ks), d_k[0], rng), ks)] for _ in range(n_null)])
    lo, hi = np.percentile(null, [2.5, 97.5], axis=0)        # shape (len(ks), 2)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(16, 4.5))
    ax_a.plot(ks, d_k, "o-", color="0.2", label="data")
    ax_a.fill_between(ks, lo[:, 0], hi[:, 0], color="crimson", alpha=0.2, label=f"null, d = {d_k[0]:.1f}")
    ax_a.set(xscale="log", xlabel="neighbour order k", ylabel=r"$\hat d_k$ (Levina–Bickel)",
             title="4a  dimension vs. scale")
    ax_b.plot(ks, ks_stat, "o-", color="0.2", label="data")
    ax_b.fill_between(ks, lo[:, 1], hi[:, 1], color="crimson", alpha=0.2, label="null 95%")
    ax_b.set(xscale="log", yscale="log", xlabel="neighbour order k", ylabel="KS distance to U(0,1)",
             title="4b  shape of the inner-neighbour law")
    for k, _, _, u in res[::3]:
        u = np.sort(u)
        ax_c.plot(u, np.arange(1, u.size + 1) / u.size - u, label=f"k = {k}")
    ax_c.axhline(0, color="crimson", ls="--")
    ax_c.set(xlabel=r"$u = (r_j / r_k)^d$", ylabel="ECDF(u) − u", title="4c  where the deviation sits")
    for ax in (ax_a, ax_b, ax_c):
        ax.legend()
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)



def plot_twonn(mu, save_path="fig5_twonn.png", discard=0.1, n_null=50, seed=0):
    mu = np.sort(mu[np.isfinite(mu)])                       # drops duplicate frames (d1 = 0)
    n = mu.size
    x = np.log(mu)
    y = -np.log(1 - np.arange(n) / n)                       # F(mu_i) = (i-1)/N, avoids log(0)
    keep = int(n * (1 - discard))                           # Facco: fit without the largest 10 %
    d = x[:keep] @ y[:keep] / (x[:keep] @ x[:keep])         # least-squares slope through origin
    rng = np.random.default_rng(seed)
    x_null = np.log(np.sort(rng.random((n_null, n)) ** (-1 / d), axis=1))   # exact Pareto(d), same n
    lo, hi = np.percentile(x_null, [2.5, 97.5], axis=0)

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax_a.fill_betweenx(y, lo, hi, color="crimson", alpha=0.2, label="Pareto null 95%")
    ax_a.plot(x, y, ".", ms=2, color="0.2", label="data")
    ax_a.plot(x, d * x, color="crimson", ls="--", label=f"fit, d = {d:.2f}")
    ax_a.axhline(y[keep], color="tab:blue", ls="-.", label=f"largest {discard:.0%} excluded from fit")
    ax_a.set(xlabel="log μ", ylabel="−log(1 − F(μ))", title="5a  TWO-NN linearity test")
    ax_a.legend()

    ax_b.fill_between(y, lo - y / d, hi - y / d, color="crimson", alpha=0.2)
    ax_b.plot(y, x - y / d, ".", ms=2, color="0.2")
    ax_b.axhline(0, color="crimson", ls="--")
    ax_b.axvline(y[keep], color="tab:blue", ls="-.")
    ax_b.set(xlabel="−log(1 − F(μ))", ylabel="log μ − (fitted line)", title="5b  deviation from the straight line")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


#NÄCHSTER CLSUDE ROTZ

import pandas as pd
from scipy.stats import kstest

def plot_x_uniformity(nn_df, d, save_path="fig6_x_uniform.png", n_bins=20):
    """
    Histogram of x = mu^(-d) = (r1/r2)^d. If the TWO-NN assumptions hold,
    x is uniformly distributed on [0, 1]. A one-sample Kolmogorov-Smirnov
    test against U(0, 1) quantifies the deviation; the result is written
    into the plot.
    """
    mu = nn_df["mu"].to_numpy()
    mu = mu[np.isfinite(mu)]               # drop duplicate frames (d1 = 0 -> mu = inf)
    x = mu ** (-d)

    ks = kstest(x, "uniform")              # H0: x ~ U(0, 1); returns statistic D and p-value

    fig, ax = plt.subplots(figsize=(6, 4.5))
    counts, _, _ = ax.hist(x, bins=n_bins, range=(0, 1), density=True,
                           color="tab:blue", alpha=0.6, label="data")
    ax.axhline(1, color="crimson", ls="--", label="U(0, 1)")
    ax.set_ylim(0, 1.45 * max(counts.max(), 1))   # headroom so text box and legend don't cover bars

    ax.text(0.03, 0.97,
            f"KS test vs. U(0, 1)\n"
            f"N = {x.size}\n"
            f"D = {ks.statistic:.3f}\n"
            f"p = {ks.pvalue:.3g}",
            transform=ax.transAxes, va="top", ha="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set(xlabel=r"$x = (r_1/r_2)^d$", ylabel="probability density",
           title=f"6  histogram of x (d = {d:.2f})", xlim=(0, 1))
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)