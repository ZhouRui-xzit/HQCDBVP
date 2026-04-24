include("ploting.jl")
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

const FIG1_SCHEME_PARAMS = Dict(
    :A => (mu1=0.83^2, mu2=0.176^2),
    :B => (mu1=0.56, mu2=0.03),
    :C => (mu1=0.42, mu2=0.03),
)

const FIG2_MU1 = 0.83^2
const FIG2_MU2 = 0.176^2
const FIG2_V3 = -3.0
const FIG2_V4 = 8.0

function make_scheme_case(scheme::Symbol; mq_mev::Real=0.0, nz::Integer=128)
    haskey(FIG1_SCHEME_PARAMS, scheme) ||
        throw(ArgumentError("unsupported Fig. 1 scheme: $(scheme)"))
    p = FIG1_SCHEME_PARAMS[scheme]
    mq_label = mq_mev == 0 ? "mq = 0" : "mq = $(mq_mev) MeV"
    return make_fig1_case(
        "scheme $(scheme), $(mq_label)";
        dilaton_mode=:interpolating,
        mq_mev=mq_mev,
        mu1=p.mu1,
        mu2=p.mu2,
        y_guess=20.0,
        nz=nz,
    )
end

function make_fig2_case(; mq_mev::Real=55.0, nz::Integer=128)
    mq_label = mq_mev == 0 ? "mq = 0" : "mq = $(mq_mev) MeV"
    return make_fig1_case(
        "Fig. 2, $(mq_label)";
        dilaton_mode=:interpolating,
        mq_mev=mq_mev,
        v3=FIG2_V3,
        v4=FIG2_V4,
        mu1=FIG2_MU1,
        mu2=FIG2_MU2,
        y_guess=20.0,
        nz=nz,
    )
end

function solve_mq0_spectral_branch(; ntemps::Integer=57, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, nz::Integer=128, scheme::Symbol=:A)
    case = make_scheme_case(scheme; mq_mev=0.0, nz=nz)
    temperatures = collect(range(Tmin_mev * GEV_PER_MEV, Tmax_mev * GEV_PER_MEV; length=ntemps))
    problem = make_fig1_problem(case, first(temperatures); nz=nz)
    initial_guess = make_fig1_initial_guess(problem, case)
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

function solve_mq0_scheme_spectral_branches(; schemes=(:A, :B, :C), ntemps::Integer=57, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, nz::Integer=128)
    cases = []
    branches = []
    for scheme in schemes
        case, branch = solve_mq0_spectral_branch(
            ntemps=ntemps,
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            nz=nz,
            scheme=scheme,
        )
        push!(cases, case)
        push!(branches, branch)
    end
    return cases, branches
end

function solve_single_temperature(case, temperature; initial_guess, nz::Integer=case.nz)
    problem = make_fig1_problem(case, temperature; nz=nz)
    result = solve_bvp(problem, initial_guess; abstol=1e-10, reltol=1e-10, maxiters=300)
    result.converged || throw(ErrorException("single-temperature solve failed for $(case.name) at T=$(temperature / GEV_PER_MEV) MeV"))
    return result
end

function solve_branch_from_initial_guess(case, temperatures, initial_guess; solver_kwargs...)
    raw_branch = solve_fig1_branch(case, temperatures; initial_guess=initial_guess, reuse_previous=true, solver_kwargs...)
    return (
        case = raw_branch.case,
        temperatures = raw_branch.temperatures,
        results = raw_branch.results,
        mqs = raw_branch.mqs,
        sigmas = sigma_from_y0.(raw_branch.results),
        uv_residuals = raw_branch.uv_residuals,
        ir_residuals = raw_branch.ir_residuals,
    )
end

function continue_in_mq_at_lowT(; scheme::Symbol=:A, mq_target_mev::Real=7.0, nmq::Integer=15, temperature_mev::Real=20.0, nz::Integer=128)
    T = temperature_mev * GEV_PER_MEV
    mq_values = collect(range(0.0, mq_target_mev; length=nmq))
    mq_cases = [make_scheme_case(scheme; mq_mev=mq, nz=nz) for mq in mq_values]

    problem = make_fig1_problem(first(mq_cases), T; nz=nz)
    initial_guess = make_fig1_initial_guess(problem, first(mq_cases))
    result = solve_single_temperature(first(mq_cases), T; initial_guess=initial_guess, nz=nz)
    results = [result]

    for case in mq_cases[2:end]
        result = solve_single_temperature(case, T; initial_guess=result.u, nz=nz)
        push!(results, result)
    end

    return mq_cases, results
end

function solve_mq7_spectral_branch(;
    scheme::Symbol=:A,
    ntemps::Integer=57,
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    mq_target_mev::Real=7.0,
    nmq::Integer=15,
    nz::Integer=128,
)
    mq_cases, lowT_results = continue_in_mq_at_lowT(
        scheme=scheme,
        mq_target_mev=mq_target_mev,
        nmq=nmq,
        temperature_mev=Tmin_mev,
        nz=nz,
    )
    case = last(mq_cases)
    temperatures = collect(range(Tmin_mev * GEV_PER_MEV, Tmax_mev * GEV_PER_MEV; length=ntemps))
    branch = solve_branch_from_initial_guess(case, temperatures, last(lowT_results).u)
    return case, branch, mq_cases, lowT_results
end

function solve_mq7_scheme_spectral_branches(; schemes=(:A, :B, :C), ntemps::Integer=57, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, mq_target_mev::Real=7.0, nmq::Integer=15, nz::Integer=128)
    cases = []
    branches = []
    mq_continuations = []
    for scheme in schemes
        case, branch, mq_cases, lowT_results = solve_mq7_spectral_branch(
            scheme=scheme,
            ntemps=ntemps,
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            mq_target_mev=mq_target_mev,
            nmq=nmq,
            nz=nz,
        )
        push!(cases, case)
        push!(branches, branch)
        push!(mq_continuations, (scheme=scheme, cases=mq_cases, results=lowT_results))
    end
    return cases, branches, mq_continuations
end

function solve_fig2_mq55_spectral_branch(; ntemps::Integer=57, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, nz::Integer=128)
    case = make_fig2_case(mq_mev=55.0, nz=nz)
    temperatures = collect(range(Tmin_mev * GEV_PER_MEV, Tmax_mev * GEV_PER_MEV; length=ntemps))
    problem = make_fig1_problem(case, first(temperatures); nz=nz)
    initial_guess = make_fig1_initial_guess(problem, case)
    branch = solve_branch_from_initial_guess(case, temperatures, initial_guess; abstol=1e-8, reltol=1e-8, maxiters=1000)
    return case, branch
end

function solve_fig2_mq0_branch_from_guess(y_guess; ntemps::Integer=57, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, nz::Integer=128)
    case = make_fig2_case(mq_mev=0.0, nz=nz)
    case = merge(case, (; name="Fig. 2, mq = 0, y0 guess = $(y_guess)", y_guess=float(y_guess)))
    temperatures = collect(range(Tmin_mev * GEV_PER_MEV, Tmax_mev * GEV_PER_MEV; length=ntemps))
    results = Vector{Any}(undef, length(temperatures))
    sigmas = fill(NaN, length(temperatures))
    free_energies = fill(NaN, length(temperatures))
    uv_residuals = fill(NaN, length(temperatures))
    ir_residuals = fill(NaN, length(temperatures))

    problem = make_fig1_problem(case, first(temperatures); nz=nz)
    guess = make_fig1_initial_guess(problem, case)
    for (i, T) in enumerate(temperatures)
        problem = make_fig1_problem(case, T; nz=nz)
        try
            result = solve_bvp(problem, guess; abstol=1e-9, reltol=1e-9, maxiters=1000)
            results[i] = result
            if result.converged
                sigmas[i] = sigma_from_y0(result)
                free_energies[i] = free_energy(result)
                uv_residuals[i] = uv_residual_from_result(result)
                ir_residuals[i] = ir_residual_from_result(result)
                guess = result.u
            end
        catch
            results[i] = nothing
        end
    end

    return (
        case=case,
        temperatures=temperatures,
        results=results,
        sigmas=sigmas,
        free_energies=free_energies,
        uv_residuals=uv_residuals,
        ir_residuals=ir_residuals,
    )
end

function unique_fig2_candidates(candidates; sigma_tol::Real=1e-6)
    valid = [c for c in candidates if isfinite(c.sigma) && isfinite(c.free_energy)]
    sort!(valid; by=c -> c.sigma)
    unique = []
    for candidate in valid
        if isempty(unique) || all(abs(candidate.sigma - item.sigma) > sigma_tol for item in unique)
            push!(unique, candidate)
        else
            near_idx = findfirst(item -> abs(candidate.sigma - item.sigma) <= sigma_tol, unique)
            if candidate.free_energy < unique[near_idx].free_energy
                unique[near_idx] = candidate
            end
        end
    end
    return unique
end

function solve_fig2_mq0_multibranch(;
    ntemps::Integer=57,
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    nz::Integer=128,
    y_guesses=(-500.0, -100.0, -20.0, -5.0, 0.0, 5.0, 20.0, 100.0, 500.0),
)
    branches = [
        solve_fig2_mq0_branch_from_guess(
            guess;
            ntemps=ntemps,
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            nz=nz,
        )
        for guess in y_guesses
    ]
    temperatures = first(branches).temperatures
    stable_sigmas = fill(NaN, length(temperatures))
    stable_free_energies = fill(NaN, length(temperatures))
    candidate_counts = zeros(Int, length(temperatures))

    for i in eachindex(temperatures)
        candidates = [
            (branch_index=j, sigma=branch.sigmas[i], free_energy=branch.free_energies[i])
            for (j, branch) in enumerate(branches)
        ]
        unique = unique_fig2_candidates(candidates)
        candidate_counts[i] = length(unique)
        if !isempty(unique)
            best = argmin([candidate.free_energy for candidate in unique])
            stable_sigmas[i] = unique[best].sigma
            stable_free_energies[i] = unique[best].free_energy
        end
    end

    return (
        case=make_fig2_case(mq_mev=0.0, nz=nz),
        temperatures=temperatures,
        branches=branches,
        stable_sigmas=stable_sigmas,
        stable_free_energies=stable_free_energies,
        candidate_counts=candidate_counts,
    )
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
        title="Spectral sigma(T)",
        label=branch.case.name,
    )
    savefig(plt, svg_path)
    return csv_path, svg_path, plt
end

function save_scheme_branches_outputs(branches; stem="spectral_sigma_T_mq0_ABC_T20_300", outdir=joinpath(@__DIR__, "data"))
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")
    Ts_mev = first(branches).temperatures ./ GEV_PER_MEV
    open(csv_path, "w") do io
        labels = [
            "scheme_$(split(split(branch.case.name, ",")[1], " ")[2])_mq$(round(Int, branch.case.mq / GEV_PER_MEV))"
            for branch in branches
        ]
        header = join(["T_MeV"; ["sigma_$(label)_GeV3" for label in labels]], ",")
        println(io, header)
        for i in eachindex(Ts_mev)
            row = join(string.([Ts_mev[i]; [branch.sigmas[i] for branch in branches]]), ",")
            println(io, row)
        end
    end

    plt = plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Spectral sigma(T)",
        legend=:topright,
        lw=2.5,
    )
    for branch in branches
        plot!(plt, Ts_mev, branch.sigmas; label=branch.case.name)
    end
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

function save_branch_outputs(branch; stem="spectral_fig2_mq55_T20_300", outdir=joinpath(@__DIR__, "data"))
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
        title="Fig. 2 spectral sigma(T)",
        label=branch.case.name,
    )
    savefig(plt, svg_path)
    return csv_path, svg_path, plt
end

function save_fig2_mq0_multibranch_outputs(result; stem="spectral_fig2_mq0_T20_300", outdir=joinpath(@__DIR__, "data"))
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")
    Ts_mev = result.temperatures ./ GEV_PER_MEV

    open(csv_path, "w") do io
        branch_headers = reduce(vcat, [["sigma_branch$(i)_GeV3", "free_energy_branch$(i)"] for i in eachindex(result.branches)])
        println(io, join(vcat(["T_MeV", "stable_sigma_GeV3", "stable_free_energy", "candidate_count"], branch_headers), ","))
        for i in eachindex(Ts_mev)
            branch_values = reduce(vcat, [[branch.sigmas[i], branch.free_energies[i]] for branch in result.branches])
            println(io, join(string.(vcat([Ts_mev[i], result.stable_sigmas[i], result.stable_free_energies[i], result.candidate_counts[i]], branch_values)), ","))
        end
    end

    plt = plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 2 mq = 0 spectral sigma(T)",
        legend=:topright,
    )
    for (i, branch) in enumerate(result.branches)
        plot!(plt, Ts_mev, branch.sigmas; label="candidate $(i)", lw=1.2, linestyle=:dash, alpha=0.55)
    end
    plot!(plt, Ts_mev, result.stable_sigmas; label="minimum free energy", lw=3.0, color=:red)
    savefig(plt, svg_path)
    return csv_path, svg_path, plt
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
    elseif mode == :mq0_ABC_spectral
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        scheme_cases, scheme_branches = solve_mq0_scheme_spectral_branches(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)

        combined_csv, combined_svg, fig = save_scheme_branches_outputs(scheme_branches; outdir=data_dir)

        first_T_mev = first(first(scheme_branches).temperatures) / GEV_PER_MEV
        last_T_mev = last(first(scheme_branches).temperatures) / GEV_PER_MEV
        for branch in scheme_branches
            println(branch.case.name, " sigma(T=", first_T_mev, " MeV) = ", branch.sigmas[1], " GeV^3")
            println(branch.case.name, " sigma(T=", last_T_mev, " MeV) = ", branch.sigmas[end], " GeV^3")
        end
        println("combined csv = ", combined_csv)
        println("combined svg = ", combined_svg)

        return (
            scheme_branches = scheme_branches,
            plot = fig,
        )
    elseif mode == :mq7_ABC_spectral
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        scheme_cases, scheme_branches, mq_continuations = solve_mq7_scheme_spectral_branches(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)

        combined_csv, combined_svg, fig = save_scheme_branches_outputs(scheme_branches; stem="spectral_sigma_T_mq7_ABC_T20_300", outdir=data_dir)

        first_T_mev = first(first(scheme_branches).temperatures) / GEV_PER_MEV
        last_T_mev = last(first(scheme_branches).temperatures) / GEV_PER_MEV
        for branch in scheme_branches
            println(branch.case.name, " sigma(T=", first_T_mev, " MeV) = ", branch.sigmas[1], " GeV^3")
            println(branch.case.name, " sigma(T=", last_T_mev, " MeV) = ", branch.sigmas[end], " GeV^3")
        end
        println("combined csv = ", combined_csv)
        println("combined svg = ", combined_svg)

        return (
            scheme_branches = scheme_branches,
            mq_continuations = mq_continuations,
            plot = fig,
        )
    elseif mode == :fig2_mq55_spectral
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        fig2_case, fig2_branch = solve_fig2_mq55_spectral_branch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)
        csv_path, svg_path, fig = save_branch_outputs(fig2_branch; outdir=data_dir)

        first_T_mev = first(fig2_branch.temperatures) / GEV_PER_MEV
        last_T_mev = last(fig2_branch.temperatures) / GEV_PER_MEV
        println(fig2_case.name, " sigma(T=", first_T_mev, " MeV) = ", fig2_branch.sigmas[1], " GeV^3")
        println(fig2_case.name, " sigma(T=", last_T_mev, " MeV) = ", fig2_branch.sigmas[end], " GeV^3")
        println("csv = ", csv_path)
        println("svg = ", svg_path)

        return (
            fig2_branch = fig2_branch,
            plot = fig,
        )
    elseif mode == :fig2_mq0_spectral
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        fig2_result = solve_fig2_mq0_multibranch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)
        csv_path, svg_path, fig = save_fig2_mq0_multibranch_outputs(fig2_result; outdir=data_dir)

        first_T_mev = first(fig2_result.temperatures) / GEV_PER_MEV
        last_T_mev = last(fig2_result.temperatures) / GEV_PER_MEV
        println("Fig. 2, mq = 0 stable sigma(T=", first_T_mev, " MeV) = ", fig2_result.stable_sigmas[1], " GeV^3")
        println("Fig. 2, mq = 0 stable sigma(T=", last_T_mev, " MeV) = ", fig2_result.stable_sigmas[end], " GeV^3")
        println("csv = ", csv_path)
        println("svg = ", svg_path)

        return (
            fig2_result = fig2_result,
            plot = fig,
        )
    end

    throw(ArgumentError("unsupported mode: $(mode)"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
