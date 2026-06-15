# Data-Driven Invariant Set Computation — Benchmarks

Computes invariant sets for 14 benchmark dynamical systems using RBF feature lifting and linear programming (LP).

## Quick Start

Open MATLAB, `cd` to this folder, then run:

```matlab
invariant_set_all
```

Results (figures + barrier `.mat` files) are saved to `results/<example>/`.

---

## File Overview

| File | Purpose |
|------|---------|
| `invariant_set_all.m` | **Main script** — set sweep parameters here, then run |
| `get_constraint.m` | Returns the constraint function `g(x)` for each benchmark |
| `compute_invariant_set.m` | Builds RBF features and solves the LP to get barrier coefficients `θ` |
| `plot_invariant_2d/3d/nd.m` | Visualization for 2D, 3D, and ≥4D systems |
| `gen_data.m` | Generates grid-sampled trajectory data `(z, z_p)` |
| `generate_all_data.m` | Calls `gen_data` for all 14 benchmarks; run once to populate `data_grid/` |
| `rbf.m` | RBF kernel evaluations |
| `data_grid/` | Pre-generated `.mat` files — one per benchmark |
| `results/` | Output figures (`.fig`, `.png`) and barrier files (`_barrier.mat`) |

---

## Configuring a Run

All parameters are set at the top of `invariant_set_all.m`.
Any parameter that accepts a vector will sweep all combinations:

```matlab
%% Sweep Parameters
example_id        = 1;                         % scalar or vector, e.g. 1:14
Guarantee_id      = 1;                         % 1 = probabilistic, 2 = deterministic
Lyapunov_discount = [0, 0.1, 0.2, 0.3, 0.5]; % scalar or vector
RBFtype_id        = 4;                         % scalar or vector, see table below
n_centers         = 31;                        % scalar or vector, e.g. [15, 31, 51]
```

Total runs = product of all vector lengths.  
Example: `example_id=1:3`, `Guarantee_id=1:2`, `Lyapunov_discount=[0,0.1]` → 12 runs.

### Example IDs

| ID | Name | Dim | Constraint set |
|----|------|-----|----------------|
| 1  | linear | 2 | `\|x1\| ≤ 1` |
| 2  | soft_landing | 2 | `x1 ≤ 0.01, x2 ≥ 0` |
| 3  | soft_landing_relaxed | 2 | same as soft_landing |
| 4  | pendulum | 2 | `\|x1\| ≤ 1` |
| 5  | van_der_pol | 2 | `‖x‖∞ ≤ 3` |
| 6  | inverted_pendulum | 2 | `\|x1\|/2.5, \|x2\|/5 ≤ 1` |
| 7  | duffing | 2 | `\|x2\| ≤ 0.7` |
| 8  | julia | 2 | `‖x‖₂ ≤ 1` |
| 9  | bicycle | 2 | `\|x1\| ≤ 2` |
| 10 | power | 2 | `‖x‖∞ ≤ 3` |
| 11 | lorenz | 3 | box: `\|x1\|≤20, \|x2\|≤30, 0≤x3≤50` |
| 12 | cornhole | 4 | non-convex (bag through hole) |
| 13 | double_pendulum | 4 | tip positions bounded |
| 14 | moon_lander | 5 | two landing cones + fuel ≥ 0 |

### RBF Type IDs

| ID | Kernel |
|----|--------|
| 1  | pyramid |
| 2  | triangle |
| 3  | thinplate |
| 4  | gauss |
| 5  | invquad |
| 6  | invmultquad |
| 7  | polyharmonic |
| 8  | bump |

---

## Regenerating Data

Pre-generated data is already in `data_grid/`. To regenerate from scratch:

```matlab
cd data_grid
generate_all_data
```

This calls `gen_data` for all 14 benchmarks and overwrites the `.mat` files.

---

## LP Timeout

The LP solver timeout is controlled by `lp_timeout_min` in the Fixed Parameters section of `invariant_set_all.m`:

```matlab
lp_timeout_min = 5;   % LP wall-clock timeout in minutes (0 = no limit)
```

If the LP does not finish within the allotted time, a warning is printed and the sweep moves on to the next configuration. Set to `0` to disable the timeout entirely.

Increase `n_centers` gradually to find the largest value that completes within the timeout.

---

## Output

Each completed run saves to `results/<example>/`:

- `<name>.fig` / `<name>.png` — visualization (2D–5D systems only)
- `<name>_barrier.mat` — barrier coefficients `θ`, centers, RBF type, all parameters
