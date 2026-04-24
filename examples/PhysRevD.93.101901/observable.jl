include("model.jl")

function y_profile(result)
    return vec(result.u[1, :])
end

function u_profile(result)
    return result.grid.x
end

function z_profile(result)
    return result.params.zh .* u_profile(result)
end

function chi_profile(result)
    u = u_profile(result)
    y = y_profile(result)
    return chi_from_y.(y, u, Ref(result.params))
end

function chi_u_profile(result)
    u = u_profile(result)
    y = y_profile(result)
    y_u = vec(result.du[1, :])
    return chi_u_from_y.(y, y_u, u, Ref(result.params))
end

function leading_uv_chi(z, result)
    return result.params.mq * FIG1_ZETA * z
end

function sigma_from_y0(result)
    β_tilde = result.u[1, 1]
    correction = uv_log_coefficient(result.params) * log(result.params.zh)
    return (β_tilde - correction) * FIG1_ZETA / result.params.zh^3
end

function fit_sigma_from_chi_uv(result; nfit::Integer=result.params.uv_fit_points)
    z = z_profile(result)
    chi = chi_profile(result)
    order = sortperm(z)
    n = min(length(z), max(4, nfit))
    idx = order[1:n]
    zfit = z[idx]
    chifit = chi[idx]

    # Hold mq fixed to the input value and fit only the z^3 response term.
    rhs = @. chifit - result.params.mq * FIG1_ZETA * zfit
    A = reshape(zfit .^ 3 ./ FIG1_ZETA, :, 1)
    sigma = (A \ rhs)[1]
    return sigma
end

function fit_sigma_from_chi_uv_rich(result; nfit::Integer=result.params.uv_fit_points)
    z = z_profile(result)
    chi = chi_profile(result)
    order = sortperm(z)
    n = min(length(z) - 1, max(4, nfit))
    idx = order[2:n+1]  # skip the left endpoint, same idea as the reference implementation
    zfit = z[idx]
    chifit = chi[idx]

    remainder = chifit .- leading_uv_chi.(zfit, Ref(result))
    # Use sigma z^3 / zeta as the leading unknown response term, plus higher-order UV contamination terms.
    A = hcat(zfit .^ 3 ./ FIG1_ZETA, zfit .^ 5, zfit .^ 5 .* log.(zfit))
    coeffs = A \ remainder
    return coeffs[1]
end

function fit_beta_from_uv(result; nfit::Integer=result.params.uv_fit_points)
    sigma = fit_sigma_from_chi_uv_rich(result; nfit=nfit)
    beta = sigma * result.params.zh^3 / FIG1_ZETA
    return beta, [beta]
end

function sigma_from_result(result)
    return sigma_from_y0(result)
end

mq_from_result(result) = result.params.mq

function uv_residual_from_result(result)
    return result.du[1, 1]
end

function ir_residual_from_result(result)
    y = result.u[1, end]
    y_u = result.du[1, end]
    chi = chi_from_y(y, 1.0, result.params)
    chi_u = chi_u_from_y(y, y_u, 1.0, result.params)
    return 4 * chi_u + dVdchi(chi, result.params)
end

function trapezoid_integral(x, y)
    length(x) == length(y) || throw(DimensionMismatch("x and y must have the same length"))
    total = zero(promote_type(eltype(x), eltype(y)))
    for i in 1:length(x)-1
        total += (x[i+1] - x[i]) * (y[i+1] + y[i]) / 2
    end
    return total
end

function free_energy_bulk_integrand(z, chi, p)
    z == 0 && return 0.0
    sqrt_minus_g = z^(-5)
    return sqrt_minus_g * exp(-dilaton(z, p)) * (-0.5 * p.v3 * chi^3 - p.v4 * chi^4)
end

function free_energy_uv_boundary(result; uv_index::Integer=2)
    p = result.params
    u = result.grid.x[uv_index]
    z = p.zh * u
    chi = chi_from_y(result.u[1, uv_index], u, p)
    chi_u = chi_u_from_y(result.u[1, uv_index], result.du[1, uv_index], u, p)
    chi_z = chi_u / p.zh
    as_factor = z^(-3)
    return -0.5 * chi * as_factor * exp(-dilaton(z, p)) * blackening_u(u) * chi_z
end

function free_energy(result; require_chiral_limit::Bool=true)
    if require_chiral_limit && abs(result.params.mq) > 100 * eps(Float64)
        throw(ArgumentError("Eq. (10) is UV divergent for mq != 0 unless counterterms are added"))
    end

    z = z_profile(result)
    chi = chi_profile(result)
    order = sortperm(z)
    zsorted = z[order]
    chisorted = chi[order]
    integrand = free_energy_bulk_integrand.(zsorted, chisorted, Ref(result.params))
    bulk = trapezoid_integral(zsorted, integrand)
    boundary = free_energy_uv_boundary(result)
    return bulk + boundary
end

function make_fig1_initial_guess(problem, case)
    return constant_guess(problem; value=case.y_guess)
end

function solve_fig1_branch(case, temperatures; initial_guess=nothing, reuse_previous::Bool=true, solver_kwargs...)
    Ts = collect(float.(temperatures))
    isempty(Ts) && throw(ArgumentError("temperatures cannot be empty"))

    results = Vector{Any}(undef, length(Ts))
    running_guess = initial_guess

    for (i, T) in enumerate(Ts)
        problem = make_fig1_problem(case, T)
        guess = if running_guess === nothing
            make_fig1_initial_guess(problem, case)
        elseif reuse_previous
            running_guess
        else
            initial_guess
        end

        guess === nothing && (guess = make_fig1_initial_guess(problem, case))

        result = solve_bvp(problem, guess; abstol=1e-10, reltol=1e-10, maxiters=300, solver_kwargs...)
        results[i] = result
        result.converged || throw(ErrorException("Fig. 1 branch failed for case $(case.name) at T = $(T) GeV"))
        running_guess = result.u
    end

    return (
        case = case,
        temperatures = Ts,
        results = results,
        mqs = mq_from_result.(results),
        sigmas = sigma_from_result.(results),
        uv_residuals = uv_residual_from_result.(results),
        ir_residuals = ir_residual_from_result.(results),
    )
end
