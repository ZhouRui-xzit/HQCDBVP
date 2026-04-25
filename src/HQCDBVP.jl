module HQCDBVP

using CSV
using DataFrames
using FastGaussQuadrature
using LaTeXStrings
using LinearAlgebra
using NonlinearSolve
using Plots

export make_chebyshev_lobatto_nodes,
    make_chebyshev_diff_matrices,
    make_grid,
    make_model_params,
    update_model_params,
    remake_problem,
    make_bvp_problem,
    constant_guess,
    stacked_guess,
    make_dirichlet_bc,
    make_robin_bc,
    flatten_state,
    reshape_state,
    residual_vector,
    solve_bvp,
    continuation_solve,
    result_field_names,
    make_solution_dataframe,
    save_solution_csv,
    plot_solution,
    save_solution_plot

include("grid.jl")
include("problem.jl")
include("solver.jl")
include("plotting.jl")

end
