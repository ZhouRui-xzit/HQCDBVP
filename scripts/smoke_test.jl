include("../src/hqcdbvp.jl")

function bulk_equation!(res, u, du, d2u, x, p)
    res[1] = d2u[1] + p.lambda * u[1]
    return res
end

function left_bc!(res, u, du, d2u, x, p)
    res[1] = u[1]
    return res
end

function right_bc!(res, u, du, d2u, x, p)
    res[1] = u[1] - 1.0
    return res
end

function main()
    grid = make_grid(0.0, 1.0, 24)
    problem = make_bvp_problem(bulk_equation!, left_bc!, right_bc!, grid; nfields=1, p=(; lambda=0.0), field_names=[:u])
    guess = constant_guess(problem; value=0.5)
    result = solve_bvp(problem, guess; abstol=1e-12, reltol=1e-12, maxiters=200)

    println("retcode = ", result.retcode)
    println("converged = ", result.converged)
    println("x[1], x[end] = ", (grid.x[1], grid.x[end]))
    println("u(left), u(right) = ", (result.u[1, 1], result.u[1, end]))
    println("residual_norm = ", result.residual_norm)

    @assert result.converged
    @assert isapprox(grid.x[1], 0.0; atol=1e-12)
    @assert isapprox(grid.x[end], 1.0; atol=1e-12)
    @assert isapprox(result.u[1, 1], 0.0; atol=1e-8)
    @assert isapprox(result.u[1, end], 1.0; atol=1e-8)
    @assert result.residual_norm < 1e-8

    println("smoke test passed")
end