# HQCDBVP.jl

`HQCDBVP.jl` is a lightweight Julia toolkit for nonlinear two-point boundary value problems that appear in holographic QCD and related coupled radial ODE systems.

This repository is currently released as **v0.1**.

## Scope of v0.1

The current version focuses on a small and transparent workflow:

- Chebyshev-Lobatto collocation on a finite interval
- First- and second-order spectral differentiation matrices
- Multi-field coupled ODE systems
- Explicit UV / IR boundary residuals supplied by the user
- Nonlinear algebraic solves via `NonlinearSolve.jl`
- Simple natural continuation by parameter scanning
- Plotting and CSV export helpers for solutions

The code is intentionally organized as plain Julia function files rather than a registered Julia package layout. The main entry point is:

```julia
include("../src/hqcdbvp.jl")
```

## Design choices

This version deliberately does **not** attempt to handle the following automatically:

- regular singular endpoints
- UV asymptotic expansion generation
- horizon regularity generation
- pseudo-arclength continuation
- multi-domain spectral elements
- adaptive order refinement
- PDE support

The intended assumption is that the user has already rewritten the problem into a numerically regular ODE system with explicit boundary residuals.

## File layout

- `src/hqcdbvp.jl`: top-level include file
- `src/grid.jl`: nodes, weights, and differentiation matrices
- `src/problem.jl`: problem construction, parameter updates, initial guesses, boundary helpers
- `src/solver.jl`: residual assembly, nonlinear solve, natural continuation
- `src/plotting.jl`: plotting and CSV export
- `scripts/`: runnable examples and smoke tests

## Core API

### Grid

```julia
grid = make_grid(a, b, n)
```

### Problem

```julia
problem = make_bvp_problem(f!, bc_left!, bc_right!, grid;
    nfields=2,
    p=make_model_params(...),
    field_names=[:phi, :chi],
)
```

Residual function signatures:

```julia
f!(res, u, du, d2u, x, p)
bc_left!(res, u, du, d2u, x, p)
bc_right!(res, u, du, d2u, x, p)
```

### Initial guesses

```julia
guess = constant_guess(problem; value=0.0)
```

or

```julia
guess = stacked_guess(phi0, chi0, g0)
```

### Solve

```julia
result = solve_bvp(problem, guess; abstol=1e-12, reltol=1e-12, maxiters=200)
```

### Boundary helpers

```julia
uv_bc! = make_dirichlet_bc([0.0, 1.0])
ir_bc! = make_robin_bc([0.0, 0.0], [1.0, 1.0], [1.0, -1.0])
```

### Natural continuation

```julia
scan = continuation_solve(problem, :lambda, 0.0:0.2:2.0, guess;
    abstol=1e-12,
    reltol=1e-12,
    maxiters=200,
)
```

### Plotting and export

```julia
plt = plot_solution(result; field_names=[:phi, :chi], quantity=:u)
save_solution_plot(result, "solution.png"; field_names=[:phi, :chi])
save_solution_csv(result, "solution.csv"; field_names=[:phi, :chi])
```

## Example scripts

```powershell
julia scripts/smoke_test.jl
julia scripts/coupled_two_field_example.jl
julia scripts/hqcd_uv_ir_mixed_bc.jl
julia scripts/hqcd_three_field_template.jl
julia scripts/continuation_demo.jl
julia scripts/plotting_demo.jl
```

## Dependencies

The scripts expect these packages to be available in the active Julia environment:

- `FastGaussQuadrature`
- `NonlinearSolve`
- `Plots`
- `LaTeXStrings`
- `DataFrames`
- `CSV`

## Version

Current release target: **v0.1**.