include("model.jl")
using OrdinaryDiffEq
using Plots

struct ShootingResult
    temperature::Float64
    zh::Float64
    chi_h::Float64
    mq_fit::Float64
    sigma_fit::Float64
    z::Vector{Float64}
    chi::Vector{Float64}
end

function chi_ode!(du, u, p, z)
    chi = u[1]
    dchi = u[2]

    f = 1 - (z / p.zh)^4
    fp = -4 * z^3 / p.zh^4
    coeff = -3 / z - dilaton_prime(z, p) + fp / f

    du[1] = dchi
    du[2] = -coeff * dchi + dVdchi(chi, p) / (z^2 * f)
end

function horizon_slope(chi_h, zh, p)
    return -dVdchi(chi_h, p) / (4 * zh)
end

function integrate_from_horizon(case, temperature, chi_h; delta_frac=1e-5, save_points=2000)
    zh = zh_from_temperature(temperature)
    p = merge(case, (; zh=zh, temperature=temperature))
    delta = delta_frac * zh
    z0 = zh - delta
    eps = 1e-5

    slope_h = horizon_slope(chi_h, zh, p)
    chi0 = chi_h - delta * slope_h
    u0 = [chi0, slope_h]
    tspan = (z0, eps)
    saveat = collect(range(z0, eps; length=save_points))

    prob = ODEProblem(chi_ode!, u0, tspan, p)
    sol = solve(prob, Tsit5(); abstol=1e-10, reltol=1e-10, saveat=saveat)
    z = collect(sol.t)
    chi = first.(sol.u)
    return z, chi, zh
end

function fit_mq_sigma_uv(z, chi; fit_points=8)
    order = sortperm(z)
    zasc = z[order]
    chiasc = chi[order]
    n = min(length(zasc) - 1, max(5, fit_points))
    idx = 2:n+1
    zfit = zasc[idx]
    chifit = chiasc[idx]

    A = hcat(FIG1_ZETA .* zfit, zfit .^ 3 ./ FIG1_ZETA, zfit .^ 4)
    coeffs = A \ chifit
    return coeffs[1], coeffs[2]
end

function mq_residual(case, temperature, chi_h; fit_points=8)
    z, chi, zh = integrate_from_horizon(case, temperature, chi_h)
    mq_fit, sigma_fit = fit_mq_sigma_uv(z, chi; fit_points=fit_points)
    return mq_fit - case.mq, ShootingResult(temperature, zh, chi_h, mq_fit, sigma_fit, z, chi)
end

function bracket_root(case, temperature; fit_points=8, scan_values=nothing)
    values = scan_values === nothing ? vcat(range(-0.02, 0.02; length=21), range(0.03, 0.12; length=10)) : scan_values
    prev_val = nothing
    prev_res = nothing
    prev_x = nothing
    for x in values
        val, res = mq_residual(case, temperature, x; fit_points=fit_points)
        if prev_val !== nothing && sign(val) != sign(prev_val)
            return prev_x, x, prev_val, val, prev_res, res
        end
        prev_x, prev_val, prev_res = x, val, res
    end
    error("failed to bracket chi_h at T=$(temperature)")
end

function find_candidate_brackets(case, temperature; fit_points=8, scan_values=collect(range(0.0, 0.35; length=71)))
    candidates = Tuple[]
    prev_x = nothing
    prev_val = nothing
    prev_res = nothing
    for x in scan_values
        val, res = mq_residual(case, temperature, x; fit_points=fit_points)
        if prev_val !== nothing && sign(val) != sign(prev_val)
            push!(candidates, (prev_x, x, prev_val, val, prev_res, res))
        end
        prev_x, prev_val, prev_res = x, val, res
    end
    return candidates
end

function solve_shooting_at_temperature(case, temperature; fit_points=8, initial_bracket=nothing, maxiters=40)
    left, right, fleft, fright, left_res, right_res = if initial_bracket === nothing
        bracket_root(case, temperature; fit_points=fit_points)
    else
        a, b = initial_bracket
        fa, ra = mq_residual(case, temperature, a; fit_points=fit_points)
        fb, rb = mq_residual(case, temperature, b; fit_points=fit_points)
        if sign(fa) != sign(fb)
            (a, b, fa, fb, ra, rb)
        else
            center = (a + b) / 2
            width = max(abs(b - a), 0.002)
            expanded = vcat(range(center - 4width, center + 4width; length=17), range(-0.02, 0.12; length=15))
            bracket_root(case, temperature; fit_points=fit_points, scan_values=expanded)
        end
    end

    best = abs(fleft) < abs(fright) ? left_res : right_res
    for _ in 1:maxiters
        mid = (left * fright - right * fleft) / (fright - fleft)
        if !isfinite(mid) || mid <= min(left, right) || mid >= max(left, right)
            mid = (left + right) / 2
        end
        fmid, rmid = mq_residual(case, temperature, mid; fit_points=fit_points)
        abs(fmid) < abs(best.mq_fit - case.mq) && (best = rmid)
        if abs(fmid) < 1e-8
            return rmid, (left, right)
        end
        if sign(fmid) == sign(fleft)
            left, fleft, left_res = mid, fmid, rmid
        else
            right, fright, right_res = mid, fmid, rmid
        end
        best = abs(fleft) < abs(fright) ? left_res : right_res
    end
    return best, (left, right)
end

function run_shooting_branch(; mq_mev=7.0, ntemps=41, fit_points=8)
    case = make_fig1_case(
        "interpolating dilaton, mq = $(mq_mev) MeV",
        dilaton_mode=:interpolating,
        mq_mev=mq_mev,
    )
    temperatures = collect(range(0.220, 0.020; length=ntemps))
    results = Vector{ShootingResult}(undef, length(temperatures))
    prev_chi_h = nothing

    for (i, T) in enumerate(temperatures)
        if prev_chi_h === nothing
            result, _ = solve_shooting_at_temperature(case, T; fit_points=fit_points, initial_bracket=(0.004, 0.006))
        else
            candidates = find_candidate_brackets(case, T; fit_points=fit_points)
            isempty(candidates) && error("no positive-root candidates at T=$(T)")
            scores = map(c -> min(abs(c[1] - prev_chi_h), abs(c[2] - prev_chi_h)), candidates)
            bracket = candidates[argmin(scores)]
            result, _ = solve_shooting_at_temperature(case, T; fit_points=fit_points, initial_bracket=(bracket[1], bracket[2]))
        end
        results[i] = result
        prev_chi_h = result.chi_h
    end

    return reverse(results)
end

function save_shooting_outputs(results; stem="shooting_mq7")
    outdir = @__DIR__
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")

    open(csv_path, "w") do io
        println(io, "T_MeV,sigma_GeV3,mq_fit_GeV,chi_h")
        for r in results
            println(io, string(r.temperature / GEV_PER_MEV, ",", r.sigma_fit, ",", r.mq_fit, ",", r.chi_h))
        end
    end

    T_mev = [r.temperature / GEV_PER_MEV for r in results]
    sigmas = [r.sigma_fit for r in results]
    plt = plot(T_mev, sigmas; lw=2.5, xlabel="T [MeV]", ylabel="sigma [GeV^3]",
        title="Shooting sigma(T), mq = 7 MeV", label="shooting")
    savefig(plt, svg_path)
    return csv_path, svg_path
end

function main()
    results = run_shooting_branch()
    csv_path, svg_path = save_shooting_outputs(results)
    println("csv = ", csv_path)
    println("svg = ", svg_path)
    println("sigma(T=20 MeV) = ", results[1].sigma_fit)
    println("sigma(T=220 MeV) = ", results[end].sigma_fit)
    return results
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
