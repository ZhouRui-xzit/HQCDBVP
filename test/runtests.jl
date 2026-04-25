using HQCDBVP
using Test

function bulk_equation!(res, u, du, d2u, x, p)
    res[1] = d2u[1] - 2.0
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

@testset "HQCDBVP smoke test" begin
    grid = make_grid(0.0, 1.0, 12)
    problem = make_bvp_problem(
        bulk_equation!,
        left_bc!,
        right_bc!,
        grid;
        nfields=1,
        field_names=[:u],
    )

    guess = stacked_guess(collect(range(0.0, 1.0; length=grid.n + 1)))
    result = solve_bvp(problem, guess; abstol=1e-10, reltol=1e-10, maxiters=100)

    @test result.converged
    @test maximum(abs.(result.u[1, :] .- grid.x .^ 2)) < 1e-7
end
