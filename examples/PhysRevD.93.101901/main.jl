include("ploting.jl")
using DelimitedFiles

function reverse_branch(branch_desc)
    return (
        case = branch_desc.case,
        temperatures = reverse(branch_desc.temperatures),
        results = reverse(branch_desc.results),
        mqs = reverse(branch_desc.mqs),
        sigmas = reverse(branch_desc.sigmas),
        uv_residuals = reverse(branch_desc.uv_residuals),
        ir_residuals = reverse(branch_desc.ir_residuals),
    )
end

function run_massive_branch(; nz::Integer=128, y_guess::Real=0.02, ntemps::Integer=41, uv_fit_points::Integer=8)
    positive_case = make_fig1_case(
        "positive dilaton, mq = 0",
        dilaton_mode=:positive,
        mq_mev=0.0,
        y_guess=0.0,
        nz=40,
    )
    massive_case = make_fig1_case(
        "interpolating dilaton, mq = 7 MeV",
        dilaton_mode=:interpolating,
        mq_mev=7.0,
        y_guess=y_guess,
        nz=nz,
        uv_fit_points=uv_fit_points,
    )

    T_grid_massive_desc = collect(range(0.220, 0.020; length=ntemps))

    massive_branch_desc = solve_fig1_branch(massive_case, T_grid_massive_desc; reuse_previous=true)
    massive_branch = reverse_branch(massive_branch_desc)
    return positive_case, massive_case, massive_branch
end

function main(; save_plot::Bool=true)
    positive_case, massive_case, massive_branch = run_massive_branch()

    plt = make_fig1_plot(
        [positive_case, massive_case],
        [massive_branch],
    )

    outdir = @__DIR__
    if save_plot
        savefig(plt, joinpath(outdir, "fig1_reproduction.png"))
    end

    Ts_mev = massive_branch.temperatures ./ GEV_PER_MEV
    csv_path = joinpath(outdir, "sigma_T_N128.csv")
    svg_path = joinpath(outdir, "sigma_T_N128.svg")
    open(csv_path, "w") do io
        println(io, "T_MeV,sigma_GeV3,mq_GeV,uv_residual,ir_residual")
        for i in eachindex(Ts_mev)
            println(io, string(
                Ts_mev[i], ",",
                massive_branch.sigmas[i], ",",
                massive_branch.mqs[i], ",",
                massive_branch.uv_residuals[i], ",",
                massive_branch.ir_residuals[i],
            ))
        end
    end
    sigma_plot = plot(
        Ts_mev,
        massive_branch.sigmas;
        lw=2.5,
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="sigma(T) with N = 128, mq = 7 MeV",
        label="N=128",
    )
    savefig(sigma_plot, svg_path)

    println("target mq = ", massive_case.mq, " GeV")
    println("effective mq(T=20 MeV) = ", massive_branch.mqs[1], " GeV")
    println("sigma(T=20 MeV) = ", massive_branch.sigmas[1], " GeV^3")
    println("sigma(T=220 MeV) = ", massive_branch.sigmas[end], " GeV^3")
    println("uv residual max = ", maximum(abs.(massive_branch.uv_residuals)))
    println("ir residual max = ", maximum(abs.(massive_branch.ir_residuals)))
    println("csv = ", csv_path)
    println("svg = ", svg_path)
    println("saved fig1_reproduction.png = ", save_plot)

    return (
        massive_branch = massive_branch,
        plot = plt,
    )
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
