include("model.jl")

u_profile(result) = result.grid.x
z_profile(result) = result.params.zh .* u_profile(result)
yu_profile(result) = vec(result.u[1, :])
ys_profile(result) = vec(result.u[2, :])
chiu_profile(result) = [
    chi_from_y(y, u, uv_source_u(result.params), uv_linear_u_coefficient(result.params), uv_log_u_coefficient(result.params))
    for (y, u) in zip(yu_profile(result), u_profile(result))
]
chis_profile(result) = [
    chi_from_y(y, u, uv_source_s(result.params), uv_linear_s_coefficient(result.params), uv_log_s_coefficient(result.params))
    for (y, u) in zip(ys_profile(result), u_profile(result))
]

function sigma_from_y0(y0, log_coeff, p)
    return (y0 - log_coeff * log(p.zh)) * ZETA_3COLOR / p.zh^3
end

function fit_sigma_from_y_uv(y, result; nfit::Integer=result.params.uv_fit_points)
    u = u_profile(result)
    order = sortperm(u)
    n = min(length(u) - 1, max(4, nfit))
    idx = order[2:n+1]
    ufit = u[idx]
    yfit = y[idx]

    A = hcat(ones(length(ufit)), ufit .^ 2, ufit .^ 2 .* log.(ufit))
    coeffs = A \ yfit
    return sigma_from_y0(coeffs[1], 0.0, result.params)
end

function sigma_u_from_result(result)
    return sigma_from_y0(result.u[1, 1], uv_log_u_coefficient(result.params), result.params)
end

function sigma_s_from_result(result)
    return sigma_from_y0(result.u[2, 1], uv_log_s_coefficient(result.params), result.params)
end

function fit_sigma_u_from_result(result; nfit::Integer=result.params.uv_fit_points)
    u = u_profile(result)
    chi = chiu_profile(result)
    order = sortperm(u)
    n = min(length(u) - 1, max(4, nfit))
    idx = order[2:n+1]
    zfit = result.params.zh .* u[idx]
    rhs = @. chi[idx] -
        result.params.mu * ZETA_3COLOR * zfit -
        (-result.params.mu * result.params.ms * result.params.gamma * ZETA_3COLOR^2 / (2 * sqrt(2))) * zfit^2
    A = hcat(zfit .^ 3 ./ ZETA_3COLOR, zfit .^ 3 .* log.(zfit), zfit .^ 4)
    coeffs = A \ rhs
    return coeffs[1]
end

function fit_sigma_s_from_result(result; nfit::Integer=result.params.uv_fit_points)
    u = u_profile(result)
    chi = chis_profile(result)
    order = sortperm(u)
    n = min(length(u) - 1, max(4, nfit))
    idx = order[2:n+1]
    zfit = result.params.zh .* u[idx]
    rhs = @. chi[idx] -
        result.params.ms * ZETA_3COLOR * zfit -
        (-result.params.mu^2 * result.params.gamma * ZETA_3COLOR^2 / (2 * sqrt(2))) * zfit^2
    A = hcat(zfit .^ 3 ./ ZETA_3COLOR, zfit .^ 3 .* log.(zfit), zfit .^ 4)
    coeffs = A \ rhs
    return coeffs[1]
end

function uv_residual_from_result(result)
    return maximum(abs.(result.du[:, 1]))
end

function ir_residual_from_result(result)
    res = zeros(2)
    j = size(result.u, 2)
    two_plus_one_ir_bc!(
        res,
        view(result.u, :, j),
        view(result.du, :, j),
        view(result.d2u, :, j),
        result.grid.x[j],
        result.params,
    )
    return maximum(abs.(res))
end

function make_2plus1_initial_guess(problem, case)
    npoints = problem.grid.n + 1
    yu0 = fill(case.yu_guess, npoints)
    ys0 = fill(case.ys_guess, npoints)
    return stacked_guess(yu0, ys0)
end

function solve_2plus1_branch(case, temperatures; initial_guess=nothing, reuse_previous::Bool=true, solver_kwargs...)
    Ts = collect(float.(temperatures))
    isempty(Ts) && throw(ArgumentError("temperatures cannot be empty"))

    results = Vector{Any}(undef, length(Ts))
    running_guess = initial_guess

    for (i, T) in enumerate(Ts)
        problem = make_2plus1_problem(case, T)
        guess = if running_guess === nothing
            make_2plus1_initial_guess(problem, case)
        elseif reuse_previous
            running_guess
        else
            initial_guess
        end

        result = solve_bvp(problem, guess; abstol=1e-9, reltol=1e-9, maxiters=600, solver_kwargs...)
        results[i] = result
        result.converged || throw(ErrorException("2+1 branch failed for $(case.name) at T = $(T / GEV_PER_MEV) MeV"))
        running_guess = result.u
    end

    return (
        case = case,
        temperatures = Ts,
        results = results,
        sigma_u = sigma_u_from_result.(results),
        sigma_s = sigma_s_from_result.(results),
        uv_residuals = uv_residual_from_result.(results),
        ir_residuals = ir_residual_from_result.(results),
    )
end
