module HQCDBVP

include("grid.jl")
include("problem.jl")
include("solver.jl")
include("plt.jl")

using .Grid: make_chebyshev_lobatto_nodes,
    make_chebyshev_diff_matrices,
    make_grid
using .Problems: make_model_params,
    update_model_params,
    remake_problem,
    make_bvp_problem,
    constant_guess,
    stacked_guess,
    make_dirichlet_bc,
    make_robin_bc,
    flatten_state,
    reshape_state
using .Solvers: residual_vector,
    solve_bvp,
    continuation_solve
using .plt: result_field_names,
    make_solution_dataframe,
    save_solution_csv,
    plot_solution,
    save_solution_plot

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

end
