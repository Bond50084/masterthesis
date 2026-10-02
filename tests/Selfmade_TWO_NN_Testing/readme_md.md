# TWO-NN Intrinsic Dimension Estimation & Validation for MD Trajectories

This module implements the **TWO-NN** (Two Nearest Neighbors) algorithm to estimate the intrinsic dimension ($d$) of molecular dynamics (MD) trajectories, alongside a suite of statistical tests evaluating whether the underlying mathematical assumptions (e.g., local Poisson point process, spatial homogeneity, and independence of samples) hold for biomolecular conformational ensembles.

---

## Theoretical Overview

The TWO-NN estimator ([Facco et al., *Scientific Reports*, 2017](https://doi.org/10.1038/s41598-017-11873-y)) estimates the intrinsic dimension of data manifolds based solely on the ratio of the distance to the second nearest neighbor ($r_2$) to the distance to the first nearest neighbor ($r_1$):

$$\mu = \frac{r_2}{r_1}, \quad \mu \in [1, \infty)$$

Assuming points are drawn from a locally uniform Poisson point process with density $\rho$, the cumulative distribution function $F(\mu)$ depends solely on the dimension $d$:

$$F(\mu) = 1 - \mu^{-d}$$

Linearizing this relationship yields:

$$-\ln(1 - F(\mu)) = d \cdot \ln(\mu)$$

Alternatively, the intrinsic dimension can be computed directly via maximum likelihood estimation (MLE):

$$\hat{d}_{\text{MLE}} = \frac{N}{\sum_{i=1}^N \ln(\mu_i)}$$

Under the null hypothesis that the local Poisson assumption holds, the transformed variable $x = \mu^{-d} = (r_1 / r_2)^d$ is uniformly distributed on the interval $[0, 1]$.

---

## File Architecture

| File | Description |
| :--- | :--- |
| `loading_trajectory.py` | Trajectory I/O using MDAnalysis. Handles atom selection (e.g., discarding hydrogens), strided slicing, internal coordinate extraction (pairwise interatomic distance arrays), and Cartesian coordinate alignment to a reference structure. |
| `compute_dnn.py` | Core mathematical algorithms. Computes the all-to-all optimally superimposed RMSD matrix using QCP superposition, finds the 1st and 2nd nearest neighbors (and associated temporal lag $\Delta t$), and calculates TWO-NN MLEs, Levina–Bickel dimension estimates across scales $k$, and synthetic Poisson null models. |
| `visualization.py` | Generates publication-ready diagnostic plots verifying temporal decorrelation, Poisson point process statistics (Fano factor, void probability), Pareto linearity, and Kolmogorov–Smirnov uniformity tests for $x \sim \mathcal{U}(0, 1)$. |
| `main.py` | Orchestration script setting trajectory paths, slicing stride, coordinate selections, RMSD matrix execution, neighbor parsing, and figure rendering. |

---

## Diagnostic Workflow & Generated Plots

1. **Temporal Decorrelation & Neighbor Lag (`fig1_left.png`):**
   * **1a (Memory Loss):** Mean pairwise RMSD as a function of frame lag $k$. Identifies how quickly a trajectory decorrelates toward the ensemble baseline.
   * **1c (Temporal Distribution of Neighbors):** Maps query frame $t$ against the frame index $t'$ of its spatial nearest neighbors to reveal whether neighbors are temporally correlated (clustering around the diagonal $t' = t$) or true recurring conformational states.

2. **Poisson Point Process Verification (`fig2_3_poisson.png`):**
   * **2a (Dispersion / Fano Factor):** Computes $\text{Var}[K_R] / \mathbb{E}[K_R]$ within hyper-spheres of radius $R$. Deviations from 1 indicate non-Poissonian clustering or overdispersion.
   * **2b (Count Distribution):** Compares observed neighbor counts inside a ball of radius $R = \text{median}(r_2)$ against a theoretical $\text{Poisson}(\lambda)$ distribution.
   * **3a (Void Probability):** Evaluates $P(K_R = 0) / \exp(-\mathbb{E}[K_R])$ against the 95% Poisson confidence band.

3. **TWO-NN Linearity (`fig2_3_two_nn.png` / `fig5_twonn.png`):**
   * Plots $-\ln(1 - F(\mu))$ vs. $\ln(\mu)$. The slope gives $d$. Outliers (top 10% by default) are excluded to avoid finite-size boundary effects.
   * Evaluates deviations from the linear fit against Pareto null bands.

4. **Uniformity Test of $x$ (`fig6_x_uniform.png`):**
   * Computes $x = \mu^{-\hat{d}}$ and executes a one-sample Kolmogorov–Smirnov (KS) test against $\mathcal{U}(0, 1)$. Large deviations or low $p$-values indicate violations of local uniformity.

---

## Dependencies

* Python $\ge$ 3.10
* `MDAnalysis` $\ge$ 2.4.0
* `numpy`
* `scipy`
* `pandas`
* `matplotlib`

Install dependencies via `uv` or `pip`:

```bash
pip install MDAnalysis numpy scipy pandas matplotlib
```

---

## Configuration & Usage

1. Open `main.py` and adjust the trajectory settings:
   ```python
   COORDINATE_TYPE = "cartesian"      # "cartesian" or "internal"
   NUM_FRAMES = 1000000              # Maximum frames to read
   STEP = 200                        # Trajectory stride
   SELECTION = "not name H*"         # Exclude hydrogen atoms
   gro_file = "Data/input_files/coord.gro"
   xtc_file = "Data/trajectories/replica7.xtc"
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

3. The script outputs diagnostic statistics to `stdout` and saves output figures (`fig1_left.png`, `fig2_3_poisson.png`, `fig2_3_two_nn.png`, and `fig6_x_uniform.png`) into the working directory.