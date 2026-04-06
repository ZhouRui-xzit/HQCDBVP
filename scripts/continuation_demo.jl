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

grid = make_grid(0.0, 1.0, 24)
problem = make_bvp_problem(
    bulk_equation!,
    left_bc!,
    right_bc!,
    grid;
    nfields=1,
    p=make_model_params(lambda=0.0),
    field_names=[:u],
)

guess = constant_guess(problem; value=0.5)
lambda_values = collect(range(0.0, 2.0; length=6))
scan = continuation_solve(problem, :lambda, lambda_values, guess; abstol=1e-12, reltol=1e-12, maxiters=200)

for item in scan.branches
    result = item.result
    println("lambda = ", item.param_value,
        ", converged = ", result.converged,
        ", residual_norm = ", result.residual_norm,
        ", u(right) = ", result.u[1, end])
end

@assert length(scan.branches) == length(lambda_values)
@assert all(item.result.converged for item in scan.branches)

println("continuation demo passed")