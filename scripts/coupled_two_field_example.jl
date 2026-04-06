include("../src/hqcdbvp.jl")

function bulk_equation!(res, u, du, d2u, x, p)
    phi = u[1]
    chi = u[2]
    d2phi = d2u[1]
    d2chi = d2u[2]

    # 双场线性耦合示例：
    # phi'' + a * phi + b * chi = 0
    # chi'' + c * phi + d * chi = 0
    res[1] = d2phi + p.a * phi + p.b * chi
    res[2] = d2chi + p.c * phi + p.d * chi
    return res
end

function left_bc!(res, u, du, d2u, x, p)
    # 左端边界：phi(0) = 0, chi(0) = 1
    res[1] = u[1]
    res[2] = u[2] - 1.0
    return res
end

function right_bc!(res, u, du, d2u, x, p)
    # 右端边界：phi(1) = 1, chi(1) = 0
    res[1] = u[1] - 1.0
    res[2] = u[2]
    return res
end

grid = make_grid(0.0, 1.0, 24)
params = (; a = 0.0, b = 0.15, c = -0.10, d = 0.0)
problem = make_bvp_problem(bulk_equation!, left_bc!, right_bc!, grid; nfields=2, p=params, field_names=[:phi, :chi])

phi_guess = collect(range(0.0, 1.0; length=grid.n + 1))
chi_guess = collect(range(1.0, 0.0; length=grid.n + 1))
guess = stacked_guess(phi_guess, chi_guess)

result = solve_bvp(problem, guess; abstol=1e-12, reltol=1e-12, maxiters=200)

println("retcode = ", result.retcode)
println("converged = ", result.converged)
println("residual_norm = ", result.residual_norm)
println("phi(left), phi(right) = ", (result.u[1, 1], result.u[1, end]))
println("chi(left), chi(right) = ", (result.u[2, 1], result.u[2, end]))

@assert result.converged
@assert result.residual_norm < 1e-8
@assert isapprox(result.u[1, 1], 0.0; atol=1e-8)
@assert isapprox(result.u[1, end], 1.0; atol=1e-8)
@assert isapprox(result.u[2, 1], 1.0; atol=1e-8)
@assert isapprox(result.u[2, end], 0.0; atol=1e-8)

println("coupled two-field example passed")