include("../src/hqcdbvp.jl")

function bulk_equation!(res, u, du, d2u, x, p)
    phi = u[1]
    chi = u[2]
    g = u[3]

    # 三场 HQCD 风格模板。
    # 这里用线性耦合示意接口，后续可直接替换成你的真实势函数和耦合项。
    res[1] = d2u[1] + p.mphi2 * phi + p.k12 * chi + p.k13 * g
    res[2] = d2u[2] + p.k21 * phi + p.mchi2 * chi + p.k23 * g
    res[3] = d2u[3] + p.k31 * phi + p.k32 * chi + p.mg2 * g
    return res
end

function uv_bc!(res, u, du, d2u, x, p)
    res[1] = u[1]
    res[2] = u[2] - 1.0
    res[3] = u[3] + 0.5
    return res
end

function ir_bc!(res, u, du, d2u, x, p)
    res[1] = u[1] - 1.0
    res[2] = u[2]
    res[3] = u[3] - 0.25
    return res
end

grid = make_grid(0.0, 1.0, 20)
params = make_model_params(
    mphi2 = 0.0,
    mchi2 = 0.0,
    mg2 = 0.0,
    k12 = 0.05,
    k13 = -0.02,
    k21 = 0.03,
    k23 = 0.04,
    k31 = -0.01,
    k32 = 0.02,
)
problem = make_bvp_problem(bulk_equation!, uv_bc!, ir_bc!, grid; nfields=3, p=params, field_names=[:phi, :chi, :G])

guess = stacked_guess(
    collect(range(0.0, 1.0; length=grid.n + 1)),
    collect(range(1.0, 0.0; length=grid.n + 1)),
    collect(range(-0.5, 0.25; length=grid.n + 1)),
)
result = solve_bvp(problem, guess; abstol=1e-11, reltol=1e-11, maxiters=200)

println("retcode = ", result.retcode)
println("converged = ", result.converged)
println("residual_norm = ", result.residual_norm)
println("left values = ", result.u[:, 1])
println("right values = ", result.u[:, end])

@assert result.converged
@assert result.residual_norm < 1e-7

println("three-field HQCD template passed")