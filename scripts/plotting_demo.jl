include("../src/hqcdbvp.jl")

function bulk_equation!(res, u, du, d2u, x, p)
    res[1] = d2u[1]
    res[2] = d2u[2]
    return res
end

left_bc! = make_dirichlet_bc([0.0, 1.0])
right_bc! = make_dirichlet_bc([1.0, 0.0])

grid = make_grid(0.0, 1.0, 24)
problem = make_bvp_problem(bulk_equation!, left_bc!, right_bc!, grid; nfields=2, field_names=[:phi, :chi])
guess = stacked_guess(
    collect(range(0.0, 1.0; length=grid.n + 1)),
    collect(range(1.0, 0.0; length=grid.n + 1)),
)
result = solve_bvp(problem, guess; abstol=1e-12, reltol=1e-12, maxiters=200)

csv_path = save_solution_csv(result, joinpath(@__DIR__, "plotting_demo.csv"); field_names=[:phi, :chi], include_du=true, include_d2u=true)
png_path = save_solution_plot(result, joinpath(@__DIR__, "plotting_demo.png"); field_names=[:phi, :chi], quantity=:u, title="Two-field solution")

println("csv saved to: ", csv_path)
println("plot saved to: ", png_path)