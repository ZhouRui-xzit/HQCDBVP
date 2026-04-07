include("ploting.jl")
include("shooting_mq7.jl")
using DelimitedFiles

function make_positive_case()
    positive_case = make_fig1_case(
        "positive dilaton, mq = 0",
        dilaton_mode=:positive,
        mq_mev=0.0,
        y_guess=0.0,
        nz=40,
    )
    return positive_case
end

function branch_from_shooting_results(case, results)
    return (
        case = case,
        temperatures = [r.temperature for r in results],
        results = results,
        mqs = [r.mq_fit for r in results],
        sigmas = [r.sigma_fit for r in results],
        uv_residuals = fill(NaN, length(results)),
        ir_residuals = fill(NaN, length(results)),
    )
end

function interpolate_profile(zsrc, valsrc, z)
    j = searchsortedfirst(zsrc, z)
    if j <= 1
        return valsrc[1]
    elseif j > length(zsrc)
        return valsrc[end]
    else
        z1, z2 = zsrc[j - 1], zsrc[j]
        v1, v2 = valsrc[j - 1], valsrc[j]
        t = (z - z1) / (z2 - z1)
        return v1 + t * (v2 - v1)
    end
end

function seed_guess_from_shooting(case, temperature; nz::Integer=case.nz)
    result = only(run_shooting_case(
        case=case,
        ntemps=1,
        Tmin_mev=temperature / GEV_PER_MEV,
        Tmax_mev=temperature / GEV_PER_MEV,
        continue_with_zero_branch=false,
    ))

    problem = make_fig1_problem(case, temperature; nz=nz)
    zsrc = reverse(result.z)
    chisrc = reverse(result.chi)
    yvals = zeros(length(problem.grid.x))
    zh = zh_from_temperature(temperature)
    params = merge(case, (; zh=zh, temperature=float(temperature)))

    for (i, u) in pairs(problem.grid.x)
        if iszero(u)
            continue
        end
        chi = interpolate_profile(zsrc, chisrc, zh * u)
        yvals[i] = (chi - alpha_source(params) * u) / u^3 - uv_log_coefficient(params) * log(u)
    end
    yvals[1] = yvals[2]
    return reshape(yvals, 1, :)
end

function solve_mq0_spectral_branch(; ntemps::Integer=57, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, nz::Integer=128)
    case = make_fig1_case(
        "interpolating dilaton, mq = 0",
        dilaton_mode=:interpolating,
        mq_mev=0.0,
        nz=nz,
    )
    temperatures = collect(range(Tmin_mev * GEV_PER_MEV, Tmax_mev * GEV_PER_MEV; length=ntemps))
    initial_guess = seed_guess_from_shooting(case, first(temperatures); nz=nz)
    raw_branch = solve_fig1_branch(case, temperatures; initial_guess=initial_guess, reuse_previous=true)
    branch = (
        case = raw_branch.case,
        temperatures = raw_branch.temperatures,
        results = raw_branch.results,
        mqs = raw_branch.mqs,
        sigmas = sigma_from_y0.(raw_branch.results),
        uv_residuals = raw_branch.uv_residuals,
        ir_residuals = raw_branch.ir_residuals,
    )
    return case, branch
end

function solve_single_temperature(case, temperature; initial_guess, nz::Integer=case.nz)
    problem = make_fig1_problem(case, temperature; nz=nz)
    result = solve_bvp(problem, initial_guess; abstol=1e-10, reltol=1e-10, maxiters=300)
    result.converged || throw(ErrorException("single-temperature solve failed for $(case.name) at T=$(temperature / GEV_PER_MEV) MeV"))
    return result
end

function make_interp_case(mq_mev::Real; nz::Integer=128)
    return make_fig1_case(
        mq_mev == 0 ? "interpolating dilaton, mq = 0" : "interpolating dilaton, mq = $(mq_mev) MeV";
        dilaton_mode=:interpolating,
        mq_mev=mq_mev,
        nz=nz,
    )
end

function continue_in_mq_at_lowT(; mq_target_mev::Real=7.0, nmq::Integer=15, temperature_mev::Real=20.0, nz::Integer=128)
    T = temperature_mev * GEV_PER_MEV
    mq_values = collect(range(0.0, mq_target_mev; length=nmq))
    mq_cases = [make_interp_case(mq; nz=nz) for mq in mq_values]
    seed = seed_guess_from_shooting(first(mq_cases), T; nz=nz)
    result = solve_single_temperature(first(mq_cases), T; initial_guess=seed, nz=nz)
    results = [result]

    for case in mq_cases[2:end]
        try
            result = solve_single_temperature(case, T; initial_guess=result.u, nz=nz)
        catch
            throw(ErrorException("mq continuation failed at mq=$(case.mq / GEV_PER_MEV) MeV; increase nmq or use adaptive stepping"))
        end
        push!(results, result)
    end

    return mq_cases, results
end

function adaptive_continue_in_mq_at_lowT(; mq_target_mev::Real=7.0, temperature_mev::Real=20.0, nz::Integer=128, max_step_mev::Real=0.25)
    T = temperature_mev * GEV_PER_MEV
    mq_values = collect(0.0:max_step_mev:mq_target_mev)
    last(mq_values) < mq_target_mev && push!(mq_values, mq_target_mev)
    mq_cases = [make_interp_case(mq; nz=nz) for mq in mq_values]

    seed = seed_guess_from_shooting(first(mq_cases), T; nz=nz)
    result = solve_single_temperature(first(mq_cases), T; initial_guess=seed, nz=nz)
    results = [result]

    for case in mq_cases[2:end]
        result = solve_single_temperature(case, T; initial_guess=result.u, nz=nz)
        push!(results, result)
    end
    return mq_cases, results
end

function solve_mq7_spectral_branch(;
    ntemps::Integer=57,
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    mq_target_mev::Real=7.0,
    nmq::Integer=15,
    nz::Integer=128,
)
    mq_cases, lowT_results = try
        continue_in_mq_at_lowT(
            mq_target_mev=mq_target_mev,
            nmq=nmq,
            temperature_mev=Tmin_mev,
            nz=nz,
        )
    catch
        adaptive_continue_in_mq_at_lowT(
            mq_target_mev=mq_target_mev,
            temperature_mev=Tmin_mev,
            nz=nz,
            max_step_mev=0.25,
        )
    end
    case = last(mq_cases)
    temperatures = collect(range(Tmin_mev * GEV_PER_MEV, Tmax_mev * GEV_PER_MEV; length=ntemps))
    raw_branch = solve_fig1_branch(case, temperatures; initial_guess=last(lowT_results).u, reuse_previous=true)
    branch = (
        case = raw_branch.case,
        temperatures = raw_branch.temperatures,
        results = raw_branch.results,
        mqs = raw_branch.mqs,
        sigmas = sigma_from_y0.(raw_branch.results),
        uv_residuals = raw_branch.uv_residuals,
        ir_residuals = raw_branch.ir_residuals,
    )
    return case, branch, mq_cases, lowT_results
end

function save_spectral_branch_outputs(branch; stem="spectral_mq0_T20_300", outdir=joinpath(@__DIR__, "data"))
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")
    Ts_mev = branch.temperatures ./ GEV_PER_MEV

    open(csv_path, "w") do io
        println(io, "T_MeV,sigma_GeV3,mq_GeV,uv_residual,ir_residual")
        for i in eachindex(Ts_mev)
            println(io, string(
                Ts_mev[i], ",",
                branch.sigmas[i], ",",
                branch.mqs[i], ",",
                branch.uv_residuals[i], ",",
                branch.ir_residuals[i],
            ))
        end
    end

    plt = plot(
        Ts_mev,
        branch.sigmas;
        lw=2.5,
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Spectral sigma(T), mq = 0",
        label="spectral mq=0",
    )
    savefig(plt, svg_path)
    return csv_path, svg_path, plt
end

function save_mq_continuation_outputs(mq_cases, results; stem="spectral_mq_continuation_T20", outdir=joinpath(@__DIR__, "data"))
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    open(csv_path, "w") do io
        println(io, "mq_MeV,sigma_GeV3,uv_residual,ir_residual")
        for (case, result) in zip(mq_cases, results)
            println(io, string(
                case.mq / GEV_PER_MEV, ",",
                sigma_from_y0(result), ",",
                uv_residual_from_result(result), ",",
                ir_residual_from_result(result),
            ))
        end
    end
    return csv_path
end

function run_shooting_reproduction(;
    mq_mev::Real,
    ntemps::Integer=57,
    fit_points::Integer=8,
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    continue_with_zero_branch::Bool=false,
)
    case = make_fig1_case(
        mq_mev == 0 ? "interpolating dilaton, mq = 0" : "interpolating dilaton, mq = $(mq_mev) MeV",
        dilaton_mode=:interpolating,
        mq_mev=mq_mev,
    )
    results = run_shooting_case(
        ; case=case,
        ntemps=ntemps,
        fit_points=fit_points,
        Tmin_mev=Tmin_mev,
        Tmax_mev=Tmax_mev,
        continue_with_zero_branch=continue_with_zero_branch,
    )
    branch = branch_from_shooting_results(case, results)
    return case, branch
end

function main(; mode::Symbol=:mq0_spectral, save_plot::Bool=true, ntemps::Integer=57, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0)
    if mode == :mq0_spectral
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        positive_case = make_positive_case()
        mq0_case, mq0_branch = solve_mq0_spectral_branch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)
        fig = make_fig1_plot([positive_case, mq0_case], [mq0_branch])
        fig_path = joinpath(data_dir, "fig1_mq0_spectral_T20_300.png")
        save_plot && savefig(fig, fig_path)
        csv_path, svg_path, _ = save_spectral_branch_outputs(mq0_branch; outdir=data_dir)

        println("mq=0 spectral sigma(T=20 MeV) = ", mq0_branch.sigmas[1], " GeV^3")
        println("mq=0 spectral sigma(T=300 MeV) = ", mq0_branch.sigmas[end], " GeV^3")
        println("uv residual max = ", maximum(abs.(mq0_branch.uv_residuals)))
        println("ir residual max = ", maximum(abs.(mq0_branch.ir_residuals)))
        println("csv = ", csv_path)
        println("svg = ", svg_path)
        println("fig = ", fig_path)

        return (
            mq0_branch = mq0_branch,
            plot = fig,
        )
    elseif mode == :mq0_mq7_spectral
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        positive_case = make_positive_case()
        mq0_case, mq0_branch = solve_mq0_spectral_branch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)
        mq7_case, mq7_branch, mq_cases, lowT_results = solve_mq7_spectral_branch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)

        fig = make_fig1_plot([positive_case, mq0_case, mq7_case], [mq0_branch, mq7_branch])
        fig_path = joinpath(data_dir, "fig1_mq0_mq7_spectral_T20_300.png")
        save_plot && savefig(fig, fig_path)

        mq0_csv, mq0_svg, _ = save_spectral_branch_outputs(mq0_branch; stem="spectral_mq0_T20_300", outdir=data_dir)
        mq7_csv, mq7_svg, _ = save_spectral_branch_outputs(mq7_branch; stem="spectral_mq7_T20_300", outdir=data_dir)
        mqcont_csv = save_mq_continuation_outputs(mq_cases, lowT_results; outdir=data_dir)

        combined_csv = joinpath(data_dir, "spectral_sigma_T_mq0_mq7_T20_300.csv")
        Ts_mev = mq0_branch.temperatures ./ GEV_PER_MEV
        open(combined_csv, "w") do io
            println(io, "T_MeV,sigma_mq0_GeV3,sigma_mq7_GeV3")
            for i in eachindex(Ts_mev)
                println(io, string(Ts_mev[i], ",", mq0_branch.sigmas[i], ",", mq7_branch.sigmas[i]))
            end
        end

        println("mq=0 spectral sigma(T=20 MeV) = ", mq0_branch.sigmas[1], " GeV^3")
        println("mq=0 spectral sigma(T=300 MeV) = ", mq0_branch.sigmas[end], " GeV^3")
        println("mq=7 spectral sigma(T=20 MeV) = ", mq7_branch.sigmas[1], " GeV^3")
        println("mq=7 spectral sigma(T=300 MeV) = ", mq7_branch.sigmas[end], " GeV^3")
        println("mq continuation csv = ", mqcont_csv)
        println("mq0 csv = ", mq0_csv)
        println("mq7 csv = ", mq7_csv)
        println("combined csv = ", combined_csv)
        println("fig = ", fig_path)

        return (
            mq0_branch = mq0_branch,
            mq7_branch = mq7_branch,
            mq_lowT_results = lowT_results,
            plot = fig,
        )
    end

    data_dir = joinpath(@__DIR__, "data")
    mkpath(data_dir)

    positive_case = make_positive_case()
    chiral_case, chiral_branch = run_shooting_reproduction(
        mq_mev=0.0,
        ntemps=ntemps,
        Tmin_mev=Tmin_mev,
        Tmax_mev=Tmax_mev,
        continue_with_zero_branch=true,
    )
    massive_case, massive_branch = run_shooting_reproduction(
        mq_mev=7.0,
        ntemps=ntemps,
        Tmin_mev=Tmin_mev,
        Tmax_mev=Tmax_mev,
        continue_with_zero_branch=false,
    )

    plt = make_fig1_plot(
        [positive_case, chiral_case, massive_case],
        [chiral_branch, massive_branch];
    )

    fig_path = joinpath(data_dir, "fig1_mq0_mq7_T20_300.png")
    save_plot && savefig(plt, fig_path)

    mq0_csv, mq0_svg = save_shooting_outputs(
        chiral_branch.results;
        stem="shooting_mq0_T20_300",
        title="Shooting sigma(T), mq = 0",
        label="mq=0",
        outdir=data_dir,
    )
    mq7_csv, mq7_svg = save_shooting_outputs(
        massive_branch.results;
        stem="shooting_mq7_T20_300",
        title="Shooting sigma(T), mq = 7 MeV",
        label="mq=7 MeV",
        outdir=data_dir,
    )

    combined_csv = joinpath(data_dir, "sigma_T_mq0_mq7_T20_300.csv")
    Ts_mev = chiral_branch.temperatures ./ GEV_PER_MEV
    open(combined_csv, "w") do io
        println(io, "T_MeV,sigma_mq0_GeV3,mqfit_mq0_GeV,chi_h_mq0,sigma_mq7_GeV3,mqfit_mq7_GeV,chi_h_mq7")
        for i in eachindex(Ts_mev)
            println(io, string(
                Ts_mev[i], ",",
                chiral_branch.sigmas[i], ",",
                chiral_branch.mqs[i], ",",
                result_chi_h(chiral_branch.results[i]), ",",
                massive_branch.sigmas[i], ",",
                massive_branch.mqs[i], ",",
                result_chi_h(massive_branch.results[i]),
            ))
        end
    end

    println("mq=0 sigma(T=20 MeV) = ", chiral_branch.sigmas[1], " GeV^3")
    println("mq=0 sigma(T=300 MeV) = ", chiral_branch.sigmas[end], " GeV^3")
    println("mq=7 sigma(T=20 MeV) = ", massive_branch.sigmas[1], " GeV^3")
    println("mq=7 sigma(T=300 MeV) = ", massive_branch.sigmas[end], " GeV^3")
    println("mq0 csv = ", mq0_csv)
    println("mq0 svg = ", mq0_svg)
    println("mq7 csv = ", mq7_csv)
    println("mq7 svg = ", mq7_svg)
    println("combined csv = ", combined_csv)
    println("saved combined figure = ", save_plot)

    return (
        chiral_branch = chiral_branch,
        massive_branch = massive_branch,
        plot = plt,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
