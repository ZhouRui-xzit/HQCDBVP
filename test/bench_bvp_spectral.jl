#!/usr/bin/env julia

const SCRIPT_START = time()

using LinearAlgebra
using Printf

const REPO_ROOT = normpath(joinpath(@__DIR__, ".."))

function parse_orders(text::AbstractString)
    stripped = strip(text)
    isempty(stripped) && return Int[]

    orders = Int[]
    for item in split(stripped, ",")
        value = parse(Int, strip(item))
        value >= 2 || throw(ArgumentError("all spectral orders must be >= 2"))
        push!(orders, value)
    end
    return orders
end

function parse_args(args)
    
    opts = Dict{String,Any}(
        "orders" => parse_orders("8,16,32,64,128"),
        "solve-orders" => parse_orders("8,16,32,64,128"),
        "repeat-residual" => 5,
        "repeat-solve" => 3,
        "maxiters" => 200,
        "abstol" => 1e-12,
        "reltol" => 1e-12,
        "no-solve" => false,
    )

    for arg in args
        if arg == "--no-solve"
            opts["no-solve"] = true
        elseif startswith(arg, "--orders=")
            opts["orders"] = parse_orders(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--solve-orders=")
            opts["solve-orders"] = parse_orders(split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--repeat-residual=")
            opts["repeat-residual"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--repeat-solve=")
            opts["repeat-solve"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--maxiters=")
            opts["maxiters"] = parse(Int, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--abstol=")
            opts["abstol"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif startswith(arg, "--reltol=")
            opts["reltol"] = parse(Float64, split(arg, "=", limit=2)[2])
        elseif arg in ("-h", "--help")
            print_help()
            exit(0)
        else
            @printf("Ignoring unknown argument: %s\n", arg)
        end
    end

    if opts["no-solve"]
        opts["solve-orders"] = Int[]
    end

    return opts
end

function print_help()
    println("""
    Usage:
      julia test/bench_bvp_spectral.jl [options]

    Options:
      --orders=8,16,32,64,128        grid/residual spectral orders
      --solve-orders=8,16,32,64,128  solve_bvp spectral orders
      --repeat-residual=5            hot residual repeats
      --repeat-solve=3               hot solve repeats
      --maxiters=200                 NonlinearSolve max iterations
      --abstol=1e-12                 solve absolute tolerance
      --reltol=1e-12                 solve relative tolerance
      --no-solve                     skip solve_bvp timings
    """)
end

const SOURCE_INCLUDE_START = time()
include(joinpath(REPO_ROOT, "src", "grid.jl"))
include(joinpath(REPO_ROOT, "src", "problem.jl"))
include(joinpath(REPO_ROOT, "src", "solver.jl"))
const SOURCE_INCLUDE_SECONDS = time() - SOURCE_INCLUDE_START

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

function linear_guess(problem)
    x = problem.grid.x
    a = problem.grid.a
    b = problem.grid.b
    U = Matrix{Float64}(undef, problem.nfields, length(x))
    @inbounds for j in eachindex(x)
        U[1, j] = (x[j] - a) / (b - a)
    end
    return U
end

function make_test_problem(n::Integer)
    grid = make_grid(0.0, 1.0, n)
    problem = make_bvp_problem(
        bulk_equation!,
        left_bc!,
        right_bc!,
        grid;
        nfields=1,
        p=nothing,
        field_names=[:u],
    )
    guess = linear_guess(problem)
    return grid, problem, guess
end

function time_grid(n::Integer)
    elapsed = @elapsed grid = make_grid(0.0, 1.0, n)
    return grid, elapsed
end

function time_residual(problem, guess; repeat::Integer)
    u0 = flatten_state(guess)

    cold_residual = nothing
    cold_seconds = @elapsed cold_residual = residual_vector(problem, u0)

    hot_residual = cold_residual
    hot_seconds = @elapsed begin
        for _ in 1:repeat
            hot_residual = residual_vector(problem, u0)
        end
    end

    return (
        cold_seconds = cold_seconds,
        hot_avg_seconds = hot_seconds / repeat,
        residual_norm = norm(hot_residual, Inf),
        size = length(u0),
    )
end

function time_solve(problem, guess; repeat::Integer, abstol::Real, reltol::Real, maxiters::Integer)
    cold_result = nothing
    cold_seconds = @elapsed cold_result = solve_bvp(
        problem,
        guess;
        abstol=abstol,
        reltol=reltol,
        maxiters=maxiters,
    )

    hot_result = cold_result
    hot_seconds = @elapsed begin
        for _ in 1:repeat
            hot_result = solve_bvp(
                problem,
                guess;
                abstol=abstol,
                reltol=reltol,
                maxiters=maxiters,
            )
        end
    end

    exact = problem.grid.x .^ 2
    max_err = maximum(abs.(hot_result.u[1, :] .- exact))

    return (
        cold_seconds = cold_seconds,
        hot_avg_seconds = hot_seconds / repeat,
        converged = hot_result.converged,
        residual_norm = hot_result.residual_norm,
        max_err = max_err,
        size = problem.nfields * (problem.grid.n + 1),
    )
end

function main()
    opts = parse_args(ARGS)

    @printf("julia_version: %s\n", string(VERSION))
    @printf("threads: %d\n", Threads.nthreads())
    @printf("script_start_to_after_include_seconds: %.6f\n", time() - SCRIPT_START)
    @printf("source_include_seconds: %.6f\n", SOURCE_INCLUDE_SECONDS)
    println()

    println("[grid construction]")
    println("n  npoints  seconds")
    for n in opts["orders"]
        grid, seconds = time_grid(n)
        @printf("%d  %d  %.6f\n", n, grid.n + 1, seconds)
    end
    println()

    println("[residual evaluation]")
    println("n  size  cold_seconds  hot_avg_seconds  residual_inf_norm")
    for n in opts["orders"]
        _, problem, guess = make_test_problem(n)
        result = time_residual(problem, guess; repeat=opts["repeat-residual"])
        @printf(
            "%d  %d  %.6f  %.6f  %.6e\n",
            n,
            result.size,
            result.cold_seconds,
            result.hot_avg_seconds,
            result.residual_norm,
        )
    end
    println()

    println("[solve_bvp]")
    if isempty(opts["solve-orders"])
        println("skipped")
        return
    end

    println("n  size  cold_seconds  hot_avg_seconds  converged  residual_inf_norm  max_err")
    for n in opts["solve-orders"]
        _, problem, guess = make_test_problem(n)
        result = time_solve(
            problem,
            guess;
            repeat=opts["repeat-solve"],
            abstol=opts["abstol"],
            reltol=opts["reltol"],
            maxiters=opts["maxiters"],
        )
        @printf(
            "%d  %d  %.6f  %.6f  %s  %.6e  %.6e\n",
            n,
            result.size,
            result.cold_seconds,
            result.hot_avg_seconds,
            string(result.converged),
            result.residual_norm,
            result.max_err,
        )
    end
end

main()
