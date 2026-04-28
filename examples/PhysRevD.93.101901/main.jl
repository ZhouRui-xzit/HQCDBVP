using DelimitedFiles
using LinearAlgebra
using HQCDBVP

import Plots
import BifurcationKit
import NonlinearSolve

include("ploting.jl")

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
const BK_T_SCALE_MEV = 100.0

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
        println("Fig. 2 mq=0 spectral scan: y0 guess = ", y_guess, ", T = ", T / GEV_PER_MEV, " MeV")
        flush(stdout)
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

function fig1_result_from_state(case, state, temperature; nz::Integer=case.nz)
    problem = make_fig1_problem(case, temperature; nz=nz)
    U = reshape_state(problem, collect(state))
    dU = U * transpose(problem.grid.D1)
    d2U = U * transpose(problem.grid.D2)
    res = residual_vector(problem, vec(U))
    return (
        converged = norm(res, Inf) <= 1e-7,
        retcode = :from_bifurcationkit,
        grid = problem.grid,
        u = copy(U),
        du = copy(dU),
        d2u = copy(d2U),
        residual = res,
        residual_norm = norm(res, Inf),
        solution = nothing,
        field_names = problem.field_names,
        params = problem.p,
    )
end

function fig2_mq0_bk_residual(case, state, pvec; nz::Integer=case.nz)
    T = pvec[1] * BK_T_SCALE_MEV * GEV_PER_MEV
    problem = make_fig1_problem(case, T; nz=nz)
    return residual_vector(problem, state)
end

function fig2_mq0_bk_record(case, state, pvec; nz::Integer=case.nz, compute_free_energy::Bool=true)
    T_mev = pvec[1] * BK_T_SCALE_MEV
    T = T_mev * GEV_PER_MEV
    result = fig1_result_from_state(case, state, T; nz=nz)
    F = if compute_free_energy
        try
            free_energy(result)
        catch
            NaN
        end
    else
        NaN
    end
    return (
        T_MeV = T_mev,
        sigma = sigma_from_y0(result),
        free_energy = F,
        residual_norm = result.residual_norm,
    )
end

function make_fig2_mq0_bk_seed(;
    initial_temperature_mev::Real=20.0,
    y_guess::Real=100.0,
    nz::Integer=128,
    verbose::Bool=true,
    nsigma::Integer=241,
    compute_free_energy::Bool=true,
)
    case = make_fig2_case(mq_mev=0.0, nz=nz)
    case = merge(case, (; name="Fig. 2, mq = 0, BK single seed", y_guess=float(y_guess)))
    T = initial_temperature_mev * GEV_PER_MEV
    problem = make_fig1_problem(case, T; nz=nz)
    guess = constant_guess(problem; value=y_guess)
    verbose && println("Fig. 2 mq=0 BK initial solve: T = ", initial_temperature_mev, " MeV, y0 guess = ", y_guess)
    flush(stdout)
    result = solve_bvp(problem, guess; abstol=1e-10, reltol=1e-10, maxiters=1200)
    result.converged || throw(ErrorException("initial BVP solve failed at T=$(initial_temperature_mev) MeV, y0 guess=$(y_guess)"))
    F = if compute_free_energy
        free_energy(result)
    else
        NaN
    end
    seed = (
        temperature = T,
        temperature_mev = float(initial_temperature_mev),
        y_guess = float(y_guess),
        sigma = sigma_from_y0(result),
        free_energy = F,
        state = vec(result.u),
    )
    if verbose
        println("  initial seed accepted: sigma = ", seed.sigma, ", free energy = ", seed.free_energy)
        flush(stdout)
    end
    return case, seed
end

function find_fig2_mq0_bk_seeds(;
    seed_temperatures_mev=(20.0, 120.0, 150.0, 165.0, 175.0, 190.0, 220.0),
    y_guesses=(-500.0, -100.0, -20.0, -5.0, 0.0, 5.0, 20.0, 100.0, 500.0),
    nz::Integer=128,
    sigma_tol::Real=1e-7,
    verbose::Bool=true,
)
    base_case = make_fig2_case(mq_mev=0.0, nz=nz)
    seeds = []
    for T_mev in seed_temperatures_mev
        if verbose
            println("Fig. 2 mq=0 BK seed search: T = ", T_mev, " MeV")
            flush(stdout)
        end
        T = T_mev * GEV_PER_MEV
        local_sigmas = Float64[]
        for y_guess in y_guesses
            if verbose
                println("  trying y0 guess = ", y_guess)
                flush(stdout)
            end
            case = merge(base_case, (; name="Fig. 2, mq = 0, BK seed", y_guess=float(y_guess)))
            problem = make_fig1_problem(case, T; nz=nz)
            guess = constant_guess(problem; value=y_guess)
            try
                result = solve_bvp(problem, guess; abstol=1e-10, reltol=1e-10, maxiters=1200)
                result.converged || continue
                sigma = sigma_from_y0(result)
                if all(abs(sigma - old) > sigma_tol for old in local_sigmas)
                    push!(local_sigmas, sigma)
                    push!(seeds, (
                        temperature = T,
                        temperature_mev = T_mev,
                        y_guess = y_guess,
                        sigma = sigma,
                        free_energy = free_energy(result),
                        state = vec(result.u),
                    ))
                    if verbose
                        println("    seed accepted: sigma = ", sigma, ", free energy = ", free_energy(result))
                        flush(stdout)
                    end
                end
            catch
                if verbose
                    println("    solve failed")
                    flush(stdout)
                end
            end
        end
        if verbose
            println("  unique seeds at T = ", T_mev, " MeV: ", length(local_sigmas))
            flush(stdout)
        end
    end
    sort!(seeds; by=s -> (s.temperature, s.sigma))
    return base_case, seeds
end

function solve_fig2_mq0_bifurcationkit_branch(case, seed;
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    nz::Integer=case.nz,
    ds_mev::Real=1.0,
    dsmax_mev::Real=5.0,
    max_steps::Integer=700,
    palc_theta::Real=0.5,
    verbose::Bool=true,
    nsigma::Integer=241,
    compute_free_energy::Bool=true,
    stop_at_zero::Bool=false,
    sigma_stop_tol::Real=1e-8,
    stop_margin_mev::Real=2.0,
    bothside::Bool=true,
)
    BK = BifurcationKit
    if verbose
        println(
            "Fig. 2 mq=0 BK continuation: seed T = ",
            seed.temperature_mev,
            " MeV, seed sigma = ",
            seed.sigma,
        )
        flush(stdout)
    end
    F = (x, p) -> fig2_mq0_bk_residual(case, x, p; nz=nz)
    last_recorded_T_mev = Ref(NaN)
    record = (x, p; k...) -> begin
        row = fig2_mq0_bk_record(case, x, p; nz=nz, compute_free_energy=compute_free_energy)
        if verbose
            T_bucket = round(row.T_MeV; digits=2)
            if !isfinite(last_recorded_T_mev[]) || abs(T_bucket - last_recorded_T_mev[]) >= 0.5
                println("  BK accepted point: T = ", row.T_MeV, " MeV, sigma = ", row.sigma)
                flush(stdout)
                last_recorded_T_mev[] = T_bucket
            end
        end
        row
    end
    finalise = (z, tau, step, contResult; k...) -> begin
        stop_at_zero || return true
        p_raw = hasproperty(z, :p) ? z.p : z[end]
        p = p_raw isa AbstractVector ? p_raw[1] : p_raw
        x = hasproperty(z, :u) ? z.u : z[1:end-1]
        T_mev = p * BK_T_SCALE_MEV
        if T_mev <= seed.temperature_mev + stop_margin_mev
            return true
        end
        row = fig2_mq0_bk_record(case, x, [p]; nz=nz, compute_free_energy=false)
        keep_going = !(isfinite(row.sigma) && abs(row.sigma) <= sigma_stop_tol)
        if verbose && !keep_going
            println("  BK stopped at zero branch: T = ", T_mev, " MeV, sigma = ", row.sigma)
            flush(stdout)
        end
        return keep_going
    end
    prob = BK.BifurcationProblem(F, seed.state, [seed.temperature_mev / BK_T_SCALE_MEV], 1; record_from_solution=record)
    opts = BK.ContinuationPar(
        p_min = Tmin_mev / BK_T_SCALE_MEV,
        p_max = Tmax_mev / BK_T_SCALE_MEV,
        ds = ds_mev / BK_T_SCALE_MEV,
        dsmin = 1.0e-3 / BK_T_SCALE_MEV,
        dsmax = dsmax_mev / BK_T_SCALE_MEV,
        max_steps = max_steps,
        newton_options = BK.NewtonPar(tol = 1e-9, max_iterations = 20),
        save_sol_every_step = 0,
    )
    branch = BK.continuation(prob, BK.PALC(θ=float(palc_theta)), opts; bothside=bothside, plot=false, finalise_solution=finalise)
    return (
        seed = seed,
        branch = branch,
    )
end

function solve_fig2_mq0_bifurcationkit_multibranch(;
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    nz::Integer=128,
    seed_temperatures_mev=(20.0, 120.0, 150.0, 165.0, 175.0, 190.0, 220.0),
    y_guesses=(-500.0, -100.0, -20.0, -5.0, 0.0, 5.0, 20.0, 100.0, 500.0),
    ds_mev::Real=1.0,
    dsmax_mev::Real=5.0,
    max_steps::Integer=700,
    palc_theta::Real=0.5,
    verbose::Bool=true,
    nsigma::Integer=241,
    smart_bk::Bool=true,
    sigma_stop_tol::Real=1e-8,
)
    seed_temperatures_mev = tuple([
        T for T in seed_temperatures_mev
        if Tmin_mev <= T <= Tmax_mev
    ]...)
    isempty(seed_temperatures_mev) &&
        throw(ArgumentError("no seed temperatures remain inside [$(Tmin_mev), $(Tmax_mev)] MeV"))

    case, seeds = find_fig2_mq0_bk_seeds(
        seed_temperatures_mev=seed_temperatures_mev,
        y_guesses=y_guesses,
        nz=nz,
        verbose=verbose,
    )
    if verbose
        println("Fig. 2 mq=0 BK seed total = ", length(seeds))
        flush(stdout)
    end
    branches = []
    for seed in seeds
        try
            push!(branches, solve_fig2_mq0_bifurcationkit_branch(
                case,
                seed;
                Tmin_mev=Tmin_mev,
                Tmax_mev=Tmax_mev,
                nz=nz,
                ds_mev=ds_mev,
                dsmax_mev=dsmax_mev,
                max_steps=max_steps,
                palc_theta=palc_theta,
                verbose=verbose,
            ))
        catch err
            if verbose
                println(
                    "Fig. 2 mq=0 BK continuation failed: seed T = ",
                    seed.temperature_mev,
                    " MeV, seed sigma = ",
                    seed.sigma,
                    ", error = ",
                    err,
                )
                flush(stdout)
            end
        end
    end
    return (
        case = case,
        seeds = seeds,
        branches = branches,
    )
end

function solve_fig2_mq0_bifurcationkit_singlebranch(;
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    nz::Integer=128,
    initial_temperature_mev::Real=Tmin_mev,
    y_guess::Real=100.0,
    ds_mev::Real=1.0,
    dsmax_mev::Real=5.0,
    max_steps::Integer=700,
    palc_theta::Real=0.5,
    verbose::Bool=true,
    compute_free_energy::Bool=true,
    stop_at_zero::Bool=false,
    sigma_stop_tol::Real=1e-8,
)
    case, seed = make_fig2_mq0_bk_seed(
        initial_temperature_mev=initial_temperature_mev,
        y_guess=y_guess,
        nz=nz,
        verbose=verbose,
        compute_free_energy=compute_free_energy,
    )
    branch = solve_fig2_mq0_bifurcationkit_branch(
        case,
        seed;
        Tmin_mev=Tmin_mev,
        Tmax_mev=Tmax_mev,
        nz=nz,
        ds_mev=ds_mev,
        dsmax_mev=dsmax_mev,
        max_steps=max_steps,
        palc_theta=palc_theta,
        verbose=verbose,
        compute_free_energy=compute_free_energy,
        stop_at_zero=stop_at_zero,
        sigma_stop_tol=sigma_stop_tol,
    )
    return (
        case = case,
        seeds = [seed],
        branches = [branch],
    )
end

function fig2_mq0_row_from_result(result; source::Symbol, step::Integer)
    return (
        source = source,
        step = step,
        T_MeV = result.params.temperature / GEV_PER_MEV,
        sigma = sigma_from_y0(result),
        free_energy = NaN,
        residual_norm = result.residual_norm,
    )
end

function fig2_mq0_prediction_error(rows, T_mev, sigma; sigma_scale::Real=1e-3)
    length(rows) < 2 && return 0.0
    prev = rows[end]
    prevprev = rows[end-1]
    dT = prev.T_MeV - prevprev.T_MeV
    abs(dT) <= eps(Float64) && return 0.0
    slope = (prev.sigma - prevprev.sigma) / dT
    predicted = prev.sigma + slope * (T_mev - prev.T_MeV)
    return abs(sigma - predicted) / max(abs(sigma), abs(prev.sigma), sigma_scale)
end

function solve_fig2_mq0_hybrid_bk(;
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    nz::Integer=128,
    y_guess::Real=100.0,
    t_ds_mev::Real=5.0,
    t_dsmin_mev::Real=0.2,
    t_dsmax_mev::Real=10.0,
    prediction_error_tol::Real=0.35,
    min_prediction_points::Integer=4,
    sigma_scale::Real=1e-3,
    sigma_stop_tol::Real=1e-8,
    bk_backtrack_mev::Real=5.0,
    bk_ds_mev::Real=1.0,
    bk_dsmax_mev::Real=5.0,
    bk_max_steps::Integer=700,
    palc_theta::Real=0.5,
    verbose::Bool=true,
)
    case = make_fig2_case(mq_mev=0.0, nz=nz)
    case = merge(case, (; name="Fig. 2, mq = 0, hybrid T/BK", y_guess=float(y_guess)))

    T_mev = float(Tmin_mev)
    problem = make_fig1_problem(case, T_mev * GEV_PER_MEV; nz=nz)
    guess = constant_guess(problem; value=y_guess)
    verbose && println("Fig. 2 mq=0 hybrid: T continuation starts at T = ", T_mev, " MeV")
    flush(stdout)
    current = solve_bvp(problem, guess; abstol=1e-10, reltol=1e-10, maxiters=1200)
    current.converged || throw(ErrorException("initial T-continuation solve failed at T=$(T_mev) MeV"))

    natural_rows = [fig2_mq0_row_from_result(current; source=:t_cont, step=0)]
    dt = float(t_ds_mev)
    trigger_reason = :none

    while T_mev < Tmax_mev
        if abs(natural_rows[end].sigma) <= sigma_stop_tol
            trigger_reason = :zero_solution_reached
            break
        end

        T_try_mev = min(Tmax_mev, T_mev + dt)
        problem_try = make_fig1_problem(case, T_try_mev * GEV_PER_MEV; nz=nz)
        result_try = try
            solve_bvp(problem_try, current.u; abstol=1e-10, reltol=1e-10, maxiters=500)
        catch
            nothing
        end

        accept = false
        pred_error = Inf
        if result_try !== nothing && result_try.converged
            row_try = fig2_mq0_row_from_result(result_try; source=:t_cont, step=length(natural_rows))
            pred_error = fig2_mq0_prediction_error(natural_rows, T_try_mev, row_try.sigma; sigma_scale=sigma_scale)
            sign_flip = natural_rows[end].sigma > sigma_scale && row_try.sigma <= 0
            accept = !sign_flip && (length(natural_rows) < min_prediction_points || pred_error <= prediction_error_tol)
        end

        if accept
            push!(natural_rows, fig2_mq0_row_from_result(result_try; source=:t_cont, step=length(natural_rows)))
            current = result_try
            T_mev = T_try_mev
            dt = min(t_dsmax_mev, 1.25 * dt)
            if verbose
                println("  T-cont accepted: T = ", T_mev, " MeV, sigma = ", natural_rows[end].sigma, ", prediction error = ", pred_error)
                flush(stdout)
            end
            continue
        end

        old_dt = dt
        dt *= 0.5
        if verbose
            reason = result_try === nothing || !getproperty(result_try, :converged) ? "solve failed" : "prediction error = $(pred_error)"
            println("  T-cont rejected at T = ", T_try_mev, " MeV (", reason, "), ΔT ", old_dt, " -> ", dt, " MeV")
            flush(stdout)
        end

        if dt < t_dsmin_mev
            trigger_reason = result_try === nothing || !getproperty(result_try, :converged) ? :solve_failure : :prediction_error
            break
        end
    end

    seed = (
        temperature = T_mev * GEV_PER_MEV,
        temperature_mev = T_mev,
        y_guess = float(y_guess),
        sigma = natural_rows[end].sigma,
        free_energy = NaN,
        state = vec(current.u),
    )

    bk_result = nothing
    bk_rows = []
    if trigger_reason in (:solve_failure, :prediction_error)
        verbose && println("Fig. 2 mq=0 hybrid: switching to BK at T = ", T_mev, " MeV, reason = ", trigger_reason)
        flush(stdout)
        branch = solve_fig2_mq0_bifurcationkit_branch(
            case,
            seed;
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            nz=nz,
            ds_mev=bk_ds_mev,
            dsmax_mev=bk_dsmax_mev,
            max_steps=bk_max_steps,
            palc_theta=palc_theta,
            verbose=verbose,
            compute_free_energy=false,
            stop_at_zero=true,
            sigma_stop_tol=sigma_stop_tol,
            bothside=true,
        )
        bk_result = (
            case = case,
            seeds = [seed],
            branches = [branch],
        )
        bk_rows = only(bk_branch_table.(bk_result.branches))
    end

    Tc_mev = isempty(bk_rows) ? (
        abs(natural_rows[end].sigma) <= sigma_stop_tol ? natural_rows[end].T_MeV : NaN
    ) : estimate_fig2_mq0_restoration_temperature([bk_rows]; sigma_tol=sigma_stop_tol)
    if !isfinite(Tc_mev)
        positive_rows = [row for row in vcat(natural_rows, bk_rows) if isfinite(row.T_MeV) && isfinite(row.sigma) && row.sigma > sigma_stop_tol]
        Tc_mev = isempty(positive_rows) ? NaN : maximum(getproperty.(positive_rows, :T_MeV))
    end

    return (
        case = case,
        natural_rows = natural_rows,
        bk_result = bk_result,
        bk_rows = bk_rows,
        trigger_reason = trigger_reason,
        seed = seed,
        Tc_mev = Tc_mev,
    )
end

function fig2_mq0_sigma_control_residual(case, unknown, sigma_target; nz::Integer=case.nz, sigma_scale::Real=0.1)
    state = view(unknown, 1:length(unknown)-1)
    T_mev = exp(unknown[end])
    T = T_mev * GEV_PER_MEV
    problem = make_fig1_problem(case, T; nz=nz)
    result = fig1_result_from_state(case, state, T; nz=nz)
    return vcat(
        residual_vector(problem, state),
        (sigma_from_y0(result) - sigma_target) / sigma_scale,
    )
end

function solve_fig2_mq0_sigma_control(;
    nz::Integer=64,
    nsigma::Integer=241,
    initial_temperature_mev::Real=20.0,
    y_guess::Real=100.0,
    sigma_min::Real=0.0,
    sigma_max=nothing,
    sigma_scale::Real=0.1,
    verbose::Bool=true,
)
    sigma_min >= 0 || throw(ArgumentError("sigma_min must be >= 0 for the physical branch"))
    case, seed = make_fig2_mq0_bk_seed(
        initial_temperature_mev=initial_temperature_mev,
        y_guess=y_guess,
        nz=nz,
        verbose=verbose,
    )
    max_sigma = sigma_max === nothing ? seed.sigma : float(sigma_max)
    max_sigma >= sigma_min || throw(ArgumentError("sigma_max must be >= sigma_min"))
    sigma_targets = collect(range(max_sigma, sigma_min; length=nsigma))

    unknown = vcat(seed.state, log(seed.temperature_mev))
    rows = []
    for (i, target) in enumerate(sigma_targets)
        if verbose
            println("Fig. 2 mq=0 sigma-control: target sigma = ", target)
            flush(stdout)
        end
        F! = (res, x, p) -> begin
            res .= fig2_mq0_sigma_control_residual(case, x, p; nz=nz, sigma_scale=sigma_scale)
            return res
        end
        prob = NonlinearSolve.NonlinearProblem(
            NonlinearSolve.NonlinearFunction(F!; resid_prototype=zeros(length(unknown))),
            unknown,
            target,
        )
        sol = NonlinearSolve.solve(prob, NonlinearSolve.NewtonRaphson(); abstol=1e-10, reltol=1e-10, maxiters=100)
        unknown = collect(sol.u)
        T_mev = exp(unknown[end])
        result = fig1_result_from_state(case, view(unknown, 1:length(unknown)-1), T_mev * GEV_PER_MEV; nz=nz)
        residual_norm = norm(fig2_mq0_sigma_control_residual(case, unknown, target; nz=nz, sigma_scale=sigma_scale), Inf)
        sigma = sigma_from_y0(result)
        F = free_energy(result)
        retcode_text = sprint(show, sol.retcode)
        converged = occursin("Success", retcode_text) || residual_norm <= 1e-7
        if !converged
            if verbose
                println("  sigma-control failed: retcode = ", sol.retcode, ", residual = ", residual_norm)
                flush(stdout)
            end
            break
        end
        push!(rows, (
            index=i,
            sigma_target=target,
            sigma=sigma,
            T_MeV=T_mev,
            free_energy=F,
            residual_norm=residual_norm,
            retcode=sol.retcode,
        ))
        if verbose
            println("  T = ", T_mev, " MeV, sigma = ", sigma, ", F = ", F, ", residual = ", residual_norm)
            flush(stdout)
        end
    end
    return (
        case=case,
        seed=seed,
        sigma_targets=sigma_targets,
        rows=rows,
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

    plt = Plots.plot(
        Ts_mev,
        branch.sigmas;
        lw=2.5,
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Spectral sigma(T)",
        label=branch.case.name,
    )
    Plots.savefig(plt, svg_path)
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

    plt = Plots.plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Spectral sigma(T)",
        legend=:topright,
        lw=2.5,
    )
    for branch in branches
        Plots.plot!(plt, Ts_mev, branch.sigmas; label=branch.case.name)
    end
    Plots.savefig(plt, svg_path)
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

    plt = Plots.plot(
        Ts_mev,
        branch.sigmas;
        lw=2.5,
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 2 spectral sigma(T)",
        label=branch.case.name,
    )
    Plots.savefig(plt, svg_path)
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

    plt = Plots.plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 2 mq = 0 spectral sigma(T)",
        legend=:topright,
    )
    for (i, branch) in enumerate(result.branches)
        Plots.plot!(plt, Ts_mev, branch.sigmas; label="candidate $(i)", lw=1.2, linestyle=:dash, alpha=0.55)
    end
    Plots.plot!(plt, Ts_mev, result.stable_sigmas; label="minimum free energy", lw=3.0, color=:red)
    Plots.savefig(plt, svg_path)
    return csv_path, svg_path, plt
end

function bk_point_value(point, name::Symbol, default=NaN)
    if hasproperty(point, name)
        value = getproperty(point, name)
        value isa AbstractVector && return first(value)
        return value
    end
    return default
end

function bk_branch_table(branch_result)
    raw_branch = branch_result.branch
    points = hasproperty(raw_branch, :branch) ? getproperty(raw_branch, :branch) : raw_branch
    rows = []
    for point in points
        T_mev = bk_point_value(point, :T_MeV)
        if !isfinite(T_mev)
            param = bk_point_value(point, :param)
            T_mev = isfinite(param) ? param * BK_T_SCALE_MEV : NaN
        end
        push!(rows, (
            step = bk_point_value(point, :step, length(rows)),
            T_MeV = T_mev,
            sigma = bk_point_value(point, :sigma),
            free_energy = bk_point_value(point, :free_energy),
            residual_norm = bk_point_value(point, :residual_norm),
        ))
    end
    return rows
end

function stable_bk_points(branch_rows; temperature_digits::Integer=1)
    bins = Dict{Float64, Any}()
    for (branch_index, rows) in enumerate(branch_rows)
        for row in rows
            isfinite(row.T_MeV) && isfinite(row.free_energy) || continue
            key = round(row.T_MeV; digits=temperature_digits)
            candidate = merge(row, (; branch_index=branch_index))
            if !haskey(bins, key) || candidate.free_energy < bins[key].free_energy
                bins[key] = candidate
            end
        end
    end
    values_sorted = collect(values(bins))
    sort!(values_sorted; by=r -> r.T_MeV)
    return values_sorted
end

function stable_fig2_mq0_points(branch_rows; Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, ntemps::Integer=281, temperature_digits::Integer=1)
    nonzero = stable_bk_points(branch_rows; temperature_digits=temperature_digits)
    Ts = collect(range(Tmin_mev, Tmax_mev; length=ntemps))
    stable = []
    for T in Ts
        near = [row for row in nonzero if abs(row.T_MeV - T) <= 0.5]
        if !isempty(near)
            best = near[argmin(abs.(getproperty.(near, :T_MeV) .- T))]
            if isfinite(best.free_energy) && best.free_energy < 0.0
                push!(stable, (T_MeV=T, sigma=best.sigma, free_energy=best.free_energy, branch_index=best.branch_index))
                continue
            end
        end
        push!(stable, (T_MeV=T, sigma=0.0, free_energy=0.0, branch_index=0))
    end
    return stable
end

function positive_branch_envelope(branch_rows; sigma_tol::Real=1e-8, temperature_digits::Integer=3)
    bins = Dict{Float64, Any}()
    for (branch_index, rows) in enumerate(branch_rows)
        for row in rows
            isfinite(row.T_MeV) && isfinite(row.sigma) || continue
            row.sigma > sigma_tol || continue
            key = round(row.T_MeV; digits=temperature_digits)
            candidate = merge(row, (; branch_index=branch_index))
            if !haskey(bins, key) || candidate.sigma > bins[key].sigma
                bins[key] = candidate
            end
        end
    end
    envelope = collect(values(bins))
    sort!(envelope; by=r -> r.T_MeV)
    return envelope
end

function estimate_fig2_mq0_restoration_temperature(branch_rows; sigma_tol::Real=1e-8)
    all_rows = sort(
        [row for rows in branch_rows for row in rows if isfinite(row.T_MeV) && isfinite(row.sigma)];
        by=r -> r.T_MeV,
    )
    isempty(all_rows) && return NaN
    positive_rows = [row for row in all_rows if row.sigma > sigma_tol]
    isempty(positive_rows) && return NaN
    last_positive = last(positive_rows)
    zero_like = [row for row in all_rows if row.T_MeV >= last_positive.T_MeV && abs(row.sigma) <= sigma_tol]
    isempty(zero_like) && return NaN
    first_zero = first(zero_like)
    if first_zero.T_MeV == last_positive.T_MeV
        return first_zero.T_MeV
    end
    weight = last_positive.sigma / (last_positive.sigma - first_zero.sigma)
    return last_positive.T_MeV + weight * (first_zero.T_MeV - last_positive.T_MeV)
end

function interpolate_sigma_on_positive_envelope(envelope, T_mev)
    isempty(envelope) && return NaN
    T_mev <= first(envelope).T_MeV && return first(envelope).sigma
    T_mev >= last(envelope).T_MeV && return last(envelope).sigma
    hi = searchsortedfirst([row.T_MeV for row in envelope], T_mev)
    lo = hi - 1
    Tlo = envelope[lo].T_MeV
    Thi = envelope[hi].T_MeV
    σlo = envelope[lo].sigma
    σhi = envelope[hi].sigma
    Thi == Tlo && return max(σlo, σhi)
    return σlo + (σhi - σlo) * (T_mev - Tlo) / (Thi - Tlo)
end

function stable_fig2_mq0_points_geometric(branch_rows; Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, ntemps::Integer=281, sigma_tol::Real=1e-8)
    envelope = positive_branch_envelope(branch_rows; sigma_tol=sigma_tol)
    Tc_mev = estimate_fig2_mq0_restoration_temperature(branch_rows; sigma_tol=sigma_tol)
    cutoff_mev = isfinite(Tc_mev) ? Tc_mev : (isempty(envelope) ? NaN : last(envelope).T_MeV)
    Ts = collect(range(Tmin_mev, Tmax_mev; length=ntemps))
    stable = []
    for T in Ts
        if !isempty(envelope) && isfinite(cutoff_mev) && T <= cutoff_mev
            sigma = interpolate_sigma_on_positive_envelope(envelope, min(T, last(envelope).T_MeV))
            push!(stable, (T_MeV=T, sigma=sigma, free_energy=NaN, branch_index=0))
        else
            push!(stable, (T_MeV=T, sigma=0.0, free_energy=NaN, branch_index=0))
        end
    end
    return stable, cutoff_mev, envelope
end

function save_fig2_mq0_bifurcationkit_outputs(result; stem="bifurcationkit_fig2_mq0_T20_300", outdir=joinpath(@__DIR__, "data"), stable_method::Symbol=:free_energy, Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, ntemps::Integer=281, sigma_tol::Real=1e-8)
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")
    branch_rows = bk_branch_table.(result.branches)

    open(csv_path, "w") do io
        println(io, "branch_index,step,seed_T_MeV,seed_sigma,T_MeV,sigma_GeV3,free_energy,residual_norm")
        for (branch_index, rows) in enumerate(branch_rows)
            seed = result.branches[branch_index].seed
            for row in rows
                println(io, join(string.([
                    branch_index,
                    row.step,
                    seed.temperature_mev,
                    seed.sigma,
                    row.T_MeV,
                    row.sigma,
                    row.free_energy,
                    row.residual_norm,
                ]), ","))
            end
        end
    end

    plt = Plots.plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 2 mq = 0 BifurcationKit branches",
        legend=:topright,
    )
    for (branch_index, rows) in enumerate(branch_rows)
        Ts = [row.T_MeV for row in rows]
        sigmas = [row.sigma for row in rows]
        seed = result.branches[branch_index].seed
        Plots.plot!(plt, Ts, sigmas; lw=1.5, alpha=0.55, label="seed $(branch_index): T=$(seed.temperature_mev) MeV")
    end

    zero_Ts = collect(range(Tmin_mev, Tmax_mev; length=ntemps))
    Plots.plot!(plt, zero_Ts, zero.(zero_Ts); lw=1.4, color=:black, linestyle=:dash, label="sigma = 0")

    stable, Tc_mev, positive_envelope = if stable_method == :geometry
        stable_fig2_mq0_points_geometric(branch_rows; Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev, ntemps=ntemps, sigma_tol=sigma_tol)
    elseif stable_method == :free_energy
        (stable_fig2_mq0_points(branch_rows; Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev, ntemps=ntemps), NaN, [])
    else
        throw(ArgumentError("unsupported stable_method: $(stable_method)"))
    end
    if !isempty(stable)
        Plots.plot!(
            plt,
            [row.T_MeV for row in stable],
            [row.sigma for row in stable];
            lw=3.0,
            color=:red,
            label=stable_method == :geometry ? "geometric stable branch" : "minimum free energy",
        )
    end

    Plots.savefig(plt, svg_path)
    return csv_path, svg_path, plt, branch_rows, stable, Tc_mev, positive_envelope
end

function save_fig2_mq0_hybrid_bk_outputs(result; stem="bifurcationkit_fig2_mq0_T20_300", outdir=joinpath(@__DIR__, "data"), Tmin_mev::Real=20.0, Tmax_mev::Real=300.0, ntemps::Integer=281, sigma_tol::Real=1e-8)
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")

    combined_nonzero_rows = vcat(result.natural_rows, result.bk_rows)
    stable, Tc_mev, positive_envelope = stable_fig2_mq0_points_geometric(
        [combined_nonzero_rows];
        Tmin_mev=Tmin_mev,
        Tmax_mev=Tmax_mev,
        ntemps=ntemps,
        sigma_tol=sigma_tol,
    )

    open(csv_path, "w") do io
        println(io, "source,step,T_MeV,sigma_GeV3,free_energy,residual_norm")
        for row in result.natural_rows
            println(io, join(string.([row.source, row.step, row.T_MeV, row.sigma, row.free_energy, row.residual_norm]), ","))
        end
        for row in result.bk_rows
            println(io, join(string.(["bk", row.step, row.T_MeV, row.sigma, row.free_energy, row.residual_norm]), ","))
        end
        for row in stable
            row.sigma == 0.0 || continue
            println(io, join(string.(["zero", 0, row.T_MeV, row.sigma, row.free_energy, 0.0]), ","))
        end
    end

    plt = Plots.plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 2 mq = 0 hybrid T-cont/BK branches",
        legend=:topright,
    )
    Plots.plot!(
        plt,
        [row.T_MeV for row in result.natural_rows],
        [row.sigma for row in result.natural_rows];
        lw=2.0,
        label="T continuation",
    )
    if !isempty(result.bk_rows)
        Plots.plot!(
            plt,
            [row.T_MeV for row in result.bk_rows],
            [row.sigma for row in result.bk_rows];
            lw=1.7,
            linestyle=:dash,
            label="BK transition window",
        )
    end
    zero_Ts = collect(range(Tmin_mev, Tmax_mev; length=ntemps))
    Plots.plot!(plt, zero_Ts, zero.(zero_Ts); lw=1.2, color=:black, linestyle=:dash, label="sigma = 0")
    Plots.plot!(
        plt,
        [row.T_MeV for row in stable],
        [row.sigma for row in stable];
        lw=3.0,
        color=:red,
        label="hybrid stable branch",
    )

    Plots.savefig(plt, svg_path)
    return csv_path, svg_path, plt, stable, Tc_mev, positive_envelope
end

function save_fig2_mq0_reconstructed_outputs(;
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    nz::Integer=64,
    bk_initial_temperature_mev::Real=140.0,
    bk_y_guess::Real=100.0,
    bk_ds_mev::Real=0.2,
    bk_dsmax_mev::Real=0.5,
    bk_max_steps::Integer=2600,
    stem="fig2_mq0_reconstructed",
    outdir=joinpath(@__DIR__, "data"),
)
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")

    bk_result = solve_fig2_mq0_bifurcationkit_singlebranch(
        Tmin_mev=Tmin_mev,
        Tmax_mev=Tmax_mev,
        nz=nz,
        initial_temperature_mev=bk_initial_temperature_mev,
        y_guess=bk_y_guess,
        ds_mev=bk_ds_mev,
        dsmax_mev=bk_dsmax_mev,
        max_steps=bk_max_steps,
        verbose=false,
    )
    bk_rows = only(bk_branch_table.(bk_result.branches))

    low_Tmax = bk_initial_temperature_mev
    positive_low = solve_fig2_mq0_branch_from_guess(
        100.0;
        ntemps=121,
        Tmin_mev=Tmin_mev,
        Tmax_mev=low_Tmax,
        nz=nz,
    )
    negative_low = solve_fig2_mq0_branch_from_guess(
        -500.0;
        ntemps=121,
        Tmin_mev=Tmin_mev,
        Tmax_mev=low_Tmax,
        nz=nz,
    )

    open(csv_path, "w") do io
        println(io, "source,T_MeV,sigma_GeV3,free_energy,residual_norm")
        for row in bk_rows
            println(io, join(string.(["bk_s_curve", row.T_MeV, row.sigma, row.free_energy, row.residual_norm]), ","))
        end
        for i in eachindex(positive_low.temperatures)
            println(io, join(string.([
                "positive_low",
                positive_low.temperatures[i] / GEV_PER_MEV,
                positive_low.sigmas[i],
                positive_low.free_energies[i],
                positive_low.results[i] === nothing ? NaN : positive_low.results[i].residual_norm,
            ]), ","))
        end
        for i in eachindex(negative_low.temperatures)
            println(io, join(string.([
                "negative_low",
                negative_low.temperatures[i] / GEV_PER_MEV,
                negative_low.sigmas[i],
                negative_low.free_energies[i],
                negative_low.results[i] === nothing ? NaN : negative_low.results[i].residual_norm,
            ]), ","))
        end
        for T in range(Tmin_mev, Tmax_mev; length=281)
            println(io, join(string.(["zero", T, 0.0, 0.0, 0.0]), ","))
        end
    end

    plt = Plots.plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 2 mq = 0 reconstructed branches",
        legend=:topright,
    )
    Plots.plot!(plt, [row.T_MeV for row in bk_rows], [row.sigma for row in bk_rows]; label="BK S curve", lw=2.0, color=:red, linestyle=:dash)
    Plots.plot!(plt, positive_low.temperatures ./ GEV_PER_MEV, positive_low.sigmas; label="positive low-T branch", lw=1.5, color=:red)
    Plots.plot!(plt, negative_low.temperatures ./ GEV_PER_MEV, negative_low.sigmas; label="negative low-T branch", lw=1.5, color=:blue)
    zero_Ts = collect(range(Tmin_mev, Tmax_mev; length=281))
    Plots.plot!(plt, zero_Ts, zero.(zero_Ts); label="sigma = 0", lw=1.5, color=:black)

    stable_rows = stable_fig2_mq0_points([bk_rows]; Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev)
    Plots.plot!(plt, [row.T_MeV for row in stable_rows], [row.sigma for row in stable_rows]; label="minimum free energy", lw=3.0, color=:red)
    Plots.savefig(plt, svg_path)
    return (
        csv_path=csv_path,
        svg_path=svg_path,
        plot=plt,
        bk_rows=bk_rows,
        positive_low=positive_low,
        negative_low=negative_low,
        stable=stable_rows,
    )
end

function save_fig2_mq0_sigma_control_outputs(result; stem="fig2_mq0_sigma_control_positive", outdir=joinpath(@__DIR__, "data"))
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")

    open(csv_path, "w") do io
        println(io, "index,sigma_target_GeV3,T_MeV,sigma_GeV3,free_energy,residual_norm,retcode")
        for row in result.rows
            println(io, join(string.([
                row.index,
                row.sigma_target,
                row.T_MeV,
                row.sigma,
                row.free_energy,
                row.residual_norm,
                row.retcode,
            ]), ","))
        end
    end

    plt = Plots.plot(
        [row.T_MeV for row in result.rows],
        [row.sigma for row in result.rows];
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 2 mq = 0 sigma-controlled positive branch",
        label="sigma >= 0",
        lw=2.5,
        legend=:topright,
    )
    zero_Ts = collect(range(20.0, 300.0; length=281))
    Plots.plot!(plt, zero_Ts, zero.(zero_Ts); label="sigma = 0", lw=1.5, color=:black, linestyle=:dash)
    Plots.savefig(plt, svg_path)
    return csv_path, svg_path, plt
end

function main(;
    mode::Symbol=:mq0_spectral,
    save_plot::Bool=true,
    ntemps::Integer=57,
    Tmin_mev::Real=20.0,
    Tmax_mev::Real=300.0,
    nz::Integer=128,
    initial_temperature_mev::Real=Tmin_mev,
    y_guess::Real=100.0,
    ds_mev::Real=1.0,
    dsmax_mev::Real=5.0,
    max_steps::Integer=700,
    palc_theta::Real=0.5,
    verbose::Bool=true,
    nsigma::Integer=241,
    smart_bk::Bool=true,
    sigma_stop_tol::Real=1e-8,
)
    if mode == :mq0_spectral
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        positive_case = make_positive_case()
        mq0_case, mq0_branch = solve_mq0_spectral_branch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev, nz=nz)
        fig = make_fig1_plot([positive_case, mq0_case], [mq0_branch])
        fig_path = joinpath(data_dir, "fig1_mq0_spectral_T20_300.png")
        save_plot && Plots.savefig(fig, fig_path)
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
        scheme_cases, scheme_branches = solve_mq0_scheme_spectral_branches(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev, nz=nz)

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
        scheme_cases, scheme_branches, mq_continuations = solve_mq7_scheme_spectral_branches(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev, nz=nz)

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
        fig2_case, fig2_branch = solve_fig2_mq55_spectral_branch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev, nz=nz)
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
        fig2_result = solve_fig2_mq0_multibranch(ntemps=ntemps, Tmin_mev=Tmin_mev, Tmax_mev=Tmax_mev, nz=nz)
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
    elseif mode == :fig2_mq0_bk
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        fig2_result = solve_fig2_mq0_hybrid_bk(
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            nz=nz,
            y_guess=y_guess,
            t_ds_mev=dsmax_mev,
            t_dsmin_mev=max(0.05, ds_mev / 5),
            t_dsmax_mev=max(dsmax_mev, ds_mev),
            bk_ds_mev=ds_mev,
            bk_dsmax_mev=dsmax_mev,
            bk_max_steps=max_steps,
            palc_theta=palc_theta,
            verbose=verbose,
            sigma_stop_tol=sigma_stop_tol,
        )
        csv_path, svg_path, fig, stable, Tc_mev, positive_envelope = save_fig2_mq0_hybrid_bk_outputs(
            fig2_result;
            outdir=data_dir,
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            ntemps=ntemps,
            sigma_tol=sigma_stop_tol,
        )

        println("Fig. 2, mq = 0 hybrid trigger reason = ", fig2_result.trigger_reason)
        println("T-continuation points = ", length(fig2_result.natural_rows))
        println("BK transition-window points = ", length(fig2_result.bk_rows))
        println("stable sampled points = ", length(stable))
        println("geometric restoration T ≈ ", Tc_mev, " MeV, positive envelope points = ", length(positive_envelope))
        println("csv = ", csv_path)
        println("svg = ", svg_path)

        return (
            fig2_result = fig2_result,
            natural_rows = fig2_result.natural_rows,
            bk_rows = fig2_result.bk_rows,
            stable = stable,
            Tc_mev = Tc_mev,
            positive_envelope = positive_envelope,
            plot = fig,
        )
    elseif mode == :fig2_mq0_bk_multiseed
        data_dir = joinpath(@__DIR__, "data")
        mkpath(data_dir)
        fig2_result = solve_fig2_mq0_bifurcationkit_multibranch(
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            nz=nz,
            ds_mev=ds_mev,
            dsmax_mev=dsmax_mev,
            max_steps=max_steps,
            palc_theta=palc_theta,
            verbose=verbose,
        )
        csv_path, svg_path, fig, branch_rows, stable, Tc_mev, positive_envelope = save_fig2_mq0_bifurcationkit_outputs(
            fig2_result;
            stem="bifurcationkit_fig2_mq0_multiseed_T20_300",
            outdir=data_dir,
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            ntemps=ntemps,
        )

        println("Fig. 2, mq = 0 BifurcationKit seed count = ", length(fig2_result.seeds))
        println("Fig. 2, mq = 0 BifurcationKit branch count = ", length(fig2_result.branches))
        println("stable sampled points = ", length(stable))
        println("csv = ", csv_path)
        println("svg = ", svg_path)

        return (
            fig2_result = fig2_result,
            branch_rows = branch_rows,
            stable = stable,
            Tc_mev = Tc_mev,
            positive_envelope = positive_envelope,
            plot = fig,
        )
    elseif mode == :fig2_mq0_reconstructed
        result = save_fig2_mq0_reconstructed_outputs(
            Tmin_mev=Tmin_mev,
            Tmax_mev=Tmax_mev,
            nz=nz,
            bk_initial_temperature_mev=initial_temperature_mev,
            bk_y_guess=y_guess,
            bk_ds_mev=ds_mev,
            bk_dsmax_mev=dsmax_mev,
            bk_max_steps=max_steps,
        )
        println("csv = ", result.csv_path)
        println("svg = ", result.svg_path)
        println("BK S curve points = ", length(result.bk_rows))
        return result
    elseif mode == :fig2_mq0_sigma_control
        result = solve_fig2_mq0_sigma_control(
            nz=nz,
            nsigma=nsigma,
            initial_temperature_mev=initial_temperature_mev,
            y_guess=y_guess,
            sigma_min=0.0,
            verbose=verbose,
        )
        csv_path, svg_path, fig = save_fig2_mq0_sigma_control_outputs(result)
        println("sigma-control points = ", length(result.rows))
        println("csv = ", csv_path)
        println("svg = ", svg_path)
        return (
            result=result,
            csv_path=csv_path,
            svg_path=svg_path,
            plot=fig,
        )
    end

    throw(ArgumentError("unsupported mode: $(mode)"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
