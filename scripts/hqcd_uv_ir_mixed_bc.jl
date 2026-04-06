include("../src/hqcdbvp.jl")

function bulk_equation!(res, u, du, d2u, x, p)
    # 这里用最简单的线性系统演示 UV / IR 混合边界。
    res[1] = d2u[1]
    res[2] = d2u[2]
    return res
end

uv_bc! = make_dirichlet_bc([0.0, 1.0])
ir_bc! = make_robin_bc([0.0, 0.0], [1.0, 1.0], [1.0, -1.0])

grid = make_grid(0.0, 1.0, 24)
params = make_model_params(label = "uv_ir_mixed_bc")
problem = make_bvp_problem(bulk_equation!, uv_bc!, ir_bc!, grid; nfields=2, p=params, field_names=[:phi, :chi])

guess = stacked_guess(
    collect(range(0.0, 1.0; length=grid.n + 1)),
    collect(range(1.0, 0.0; length=grid.n + 1)),
)
result = solve_bvp(problem, guess; abstol=1e-12, reltol=1e-12, maxiters=200)

println("retcode = ", result.retcode)
println("converged = ", result.converged)
println("residual_norm = ", result.residual_norm)
println("phi(left), phi'(right) = ", (result.u[1, 1], result.du[1, end]))
println("chi(left), chi'(right) = ", (result.u[2, 1], result.du[2, end]))

@assert result.converged
@assert result.residual_norm < 1e-8
@assert isapprox(result.u[1, 1], 0.0; atol=1e-8)
@assert isapprox(result.u[2, 1], 1.0; atol=1e-8)
@assert isapprox(result.du[1, end], 1.0; atol=1e-8)
@assert isapprox(result.du[2, end], -1.0; atol=1e-8)

println("uv/ir mixed boundary example passed")