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

function select_bracket_by_sigma(candidates; min_chi_h=1e-3)
    filtered = filter(c -> max(abs(c[1]), abs(c[2])) > min_chi_h, candidates)
    isempty(filtered) && return nothing
    scores = map(c -> max(abs(c[5].sigma_fit), abs(c[6].sigma_fit)), filtered)
    return filtered[argmax(scores)]
end

function select_bracket_by_continuity(candidates, prev_chi_h; min_chi_h=1e-3)
    filtered = filter(c -> max(abs(c[1]), abs(c[2])) > min_chi_h, candidates)
    isempty(filtered) && return nothing
    scores = map(c -> min(abs(c[1] - prev_chi_h), abs(c[2] - prev_chi_h)), filtered)
    return filtered[argmin(scores)]
end

function select_zero_bracket(candidates; max_chi_h=1e-3)
    filtered = filter(c -> max(abs(c[1]), abs(c[2])) <= max_chi_h, candidates)
    isempty(filtered) && return nothing
    scores = map(c -> max(abs(c[1]), abs(c[2])), filtered)
    return filtered[argmin(scores)]
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

function run_shooting_case(;
    case,
    ntemps=57,
    fit_points=8,
    Tmin_mev=20.0,
    Tmax_mev=300.0,
    scan_max=0.35,
    scan_points=701,
    prefer_nontrivial::Bool=true,
    continue_with_zero_branch::Bool=false,
)
    temperatures = collect(range(Tmin_mev * GEV_PER_MEV, Tmax_mev * GEV_PER_MEV; length=ntemps))
    results = ShootingResult[]
    prev_chi_h = nothing
    following_zero_branch = false

    for T in temperatures
        candidates = find_candidate_brackets(
            case,
            T;
            fit_points=fit_points,
            scan_values=collect(range(0.0, scan_max; length=scan_points)),
        )

        bracket = if prev_chi_h === nothing
            prefer_nontrivial ? select_bracket_by_sigma(candidates) : select_zero_bracket(candidates)
        elseif following_zero_branch
            select_zero_bracket(candidates)
        else
            select_bracket_by_continuity(candidates, prev_chi_h)
        end

        if bracket === nothing && continue_with_zero_branch
            bracket = select_zero_bracket(candidates)
            following_zero_branch = bracket !== nothing
        end
        bracket === nothing && error("no admissible shooting bracket at T=$(T) for $(case.name)")

        result = solve_shooting_at_temperature(
            case,
            T;
            fit_points=fit_points,
            initial_bracket=(bracket[1], bracket[2]),
            maxiters=80,
        )[1]
        push!(results, result)
        prev_chi_h = result.chi_h
    end

    return results
end

function run_shooting_branch(; mq_mev=7.0, kwargs...)
    case = make_fig1_case(
        "interpolating dilaton, mq = $(mq_mev) MeV",
        dilaton_mode=:interpolating,
        mq_mev=mq_mev,
    )
    return run_shooting_case(; case=case, kwargs...)
end

result_temperature(r) = r isa ShootingResult ? r.temperature : r.params.temperature
result_sigma(r) = r isa ShootingResult ? r.sigma_fit : sigma_from_result(r)
result_mq(r) = r isa ShootingResult ? r.mq_fit : mq_from_result(r)
result_chi_h(r) = r isa ShootingResult ? r.chi_h : NaN

function save_shooting_outputs(results; stem="shooting_mq7", title="Shooting sigma(T)", label="shooting", outdir=joinpath(@__DIR__, "data"))
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")

    open(csv_path, "w") do io
        println(io, "T_MeV,sigma_GeV3,mq_fit_GeV,chi_h")
        for r in results
            println(io, string(
                result_temperature(r) / GEV_PER_MEV, ",",
                result_sigma(r), ",",
                result_mq(r), ",",
                result_chi_h(r),
            ))
        end
    end

    T_mev = [result_temperature(r) / GEV_PER_MEV for r in results]
    sigmas = [result_sigma(r) for r in results]
    plt = plot(T_mev, sigmas; lw=2.5, xlabel="T [MeV]", ylabel="sigma [GeV^3]",
        title=title, label=label)
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
