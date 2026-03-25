# Theory: DATE Decomposition

Dynamic Average Treatment Effect (DATE) decomposition separates pointwise
causal effects into interpretable components.

Reference: Schaffe-Odeleye et al. (2026), "Dynamic Causal Inference with Time
Series Data", arXiv:2602.00836.

## Motivation

Standard CausalImpact reports a single pointwise effect trajectory
$\tau_t = y_t - \hat{y}_t$ for each post-intervention time step $t$.
This answers "how large is the effect at each time?" but not "what type
of effect is this?"

DATE decomposition answers three distinct questions:

- Did the intervention cause an immediate, one-time impact? (Spot)
- Did the intervention shift the baseline permanently? (Persistent)
- Is the effect growing or decaying over time? (Trend)

## Design Matrix

For $T_{\text{post}}$ post-intervention time points, define a design matrix
$\mathbf{D} \in \mathbb{R}^{T_{\text{post}} \times 3}$:

$$
\mathbf{D} = \begin{bmatrix}
1 & 1 & 0 \\
0 & 1 & 1 \\
0 & 1 & 2 \\
\vdots & \vdots & \vdots \\
0 & 1 & T_{\text{post}}-1
\end{bmatrix}
$$

- Column 0 (spot): impulse at $t=0$ only
- Column 1 (persistent): constant step function from $t=0$ onward
- Column 2 (trend): linear ramp $0, 1, 2, \ldots, T_{\text{post}}-1$

## OLS Projection

For each MCMC sample $s$, the pointwise effect vector
$\boldsymbol{\tau}_s \in \mathbb{R}^{T_{\text{post}}}$ is projected:

$$
\hat{\boldsymbol{\delta}}_s = (\mathbf{D}^\top \mathbf{D})^{-1} \mathbf{D}^\top \boldsymbol{\tau}_s
$$

In practice, the pseudoinverse $\mathbf{D}^+$ is computed once and applied
to all samples via matrix multiplication.

The three components of $\hat{\boldsymbol{\delta}}_s$ are
$(\delta_{\text{spot}}, \delta_{\text{persistent}}, \delta_{\text{trend}})$.

## Posterior Summaries

Component trajectories are reconstructed as:

$$
\text{spot}_t^{(s)} = \delta_{\text{spot}}^{(s)} \cdot D_{t,0}, \quad
\text{persistent}_t^{(s)} = \delta_{\text{persistent}}^{(s)} \cdot D_{t,1}, \quad
\text{trend}_t^{(s)} = \delta_{\text{trend}}^{(s)} \cdot D_{t,2}
$$

Posterior means and credible intervals are computed across samples.

## Business Examples

| Component | Example scenario |
|---|---|
| Spot | Ad campaign launch: immediate spike on day 1, no lasting change |
| Persistent | Price reduction: permanent baseline shift in daily sales |
| Trend | New product feature: effect that grows as adoption increases |

## Edge Cases

| Condition | Behavior |
|---|---|
| $T_{\text{post}} \geq 3$ | All three components identifiable |
| $T_{\text{post}} = 2$ | Trend not identifiable; only spot and persistent estimated |
| $T_{\text{post}} = 1$ | Spot and persistent are collinear; minimum-norm OLS (equal split) |

## Implementation

The decomposition is a pure Python post-processing step applied to the
existing MCMC output. No changes to the Rust Gibbs sampler are required.

```python
from causal_impact import CausalImpact

ci = CausalImpact(data, pre_period, post_period)
dec = ci.decompose()

print(f"Spot:       {dec.spot.coefficient:.2f} [{dec.spot.ci_lower:.2f}, {dec.spot.ci_upper:.2f}]")
print(f"Persistent: {dec.persistent.coefficient:.2f} [{dec.persistent.ci_lower:.2f}, {dec.persistent.ci_upper:.2f}]")
if dec.trend is not None:
    print(f"Trend:      {dec.trend.coefficient:.2f} [{dec.trend.ci_lower:.2f}, {dec.trend.ci_upper:.2f}]")
```

## Citation

```bibtex
@article{schaffeodeleye2026dynamic,
  title   = {Dynamic Causal Inference with Time Series Data},
  author  = {Schaffe-Odeleye, Ayo and others},
  journal = {arXiv preprint arXiv:2602.00836},
  year    = {2026}
}
```
