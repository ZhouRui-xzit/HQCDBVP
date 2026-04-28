using DelimitedFiles
using HQCDBVP

import Plots

include("observable.jl")

function make_physical_case(; nz::Integer=96)
    return make_2plus1_case(
        "physical point, mu = 3.336 MeV, ms = 95 MeV";
        mu_mev=3.336,
        ms_mev=95.0,
        nz=nz,
        yu_guess=0.25,
        ys_guess=0.55,
    )
end

function save_2plus1_branch_outputs(branch; stem="physical_point_T100_200", outdir=joinpath(@__DIR__, "data"))
    mkpath(outdir)
    csv_path = joinpath(outdir, stem * ".csv")
    svg_path = joinpath(outdir, stem * ".svg")

    rows = hcat(
        branch.temperatures ./ GEV_PER_MEV,
        branch.sigma_u,
        branch.sigma_s,
        branch.uv_residuals,
        branch.ir_residuals,
    )
    writedlm(csv_path, vcat(["T_MeV" "sigma_u_GeV3" "sigma_s_GeV3" "uv_residual" "ir_residual"], rows), ',')

    plt = Plots.plot(
        branch.temperatures ./ GEV_PER_MEV,
        branch.sigma_u;
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="PhysRevD.98.114003 physical point",
        label="sigma_u",
        lw=2.5,
    )
    Plots.plot!(plt, branch.temperatures ./ GEV_PER_MEV, branch.sigma_s; label="sigma_s", lw=2.5)
    Plots.savefig(plt, svg_path)
    return csv_path, svg_path, plt
end

function reverse_2plus1_branch(branch)
    return (
        case = branch.case,
        temperatures = reverse(branch.temperatures),
        results = reverse(branch.results),
        sigma_u = reverse(branch.sigma_u),
        sigma_s = reverse(branch.sigma_s),
        uv_residuals = reverse(branch.uv_residuals),
        ir_residuals = reverse(branch.ir_residuals),
    )
end

function main(;
    mode::Symbol=:physical_scan,
    ntemps::Integer=100, # 温度网格
    Tmin_mev::Real=50.0,
    Tmax_mev::Real=300.0,
    nz::Integer=96, # 谱网格
    save_plot::Bool=true,
)
    if mode == :physical_scan
        case = make_physical_case(nz=nz)
        temperatures_down = collect(range(Tmax_mev * GEV_PER_MEV, Tmin_mev * GEV_PER_MEV; length=ntemps))
        branch = reverse_2plus1_branch(solve_2plus1_branch(case, temperatures_down))
        csv_path, svg_path, plt = save_2plus1_branch_outputs(branch; stem="physical_point_T$(round(Int, Tmin_mev))_$(round(Int, Tmax_mev))")

        println(case.name)
        println("sigma_u(T=", Tmin_mev, " MeV) = ", branch.sigma_u[1], " GeV^3")
        println("sigma_s(T=", Tmin_mev, " MeV) = ", branch.sigma_s[1], " GeV^3")
        println("sigma_u(T=", Tmax_mev, " MeV) = ", branch.sigma_u[end], " GeV^3")
        println("sigma_s(T=", Tmax_mev, " MeV) = ", branch.sigma_s[end], " GeV^3")
        println("uv residual max = ", maximum(branch.uv_residuals))
        println("ir residual max = ", maximum(branch.ir_residuals))
        println("csv = ", csv_path)
        save_plot && println("svg = ", svg_path)

        return (
            case = case,
            branch = branch,
            plot = plt,
            csv_path = csv_path,
            svg_path = svg_path,
        )
    end

    throw(ArgumentError("unsupported mode: $(mode)"))
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
