include("observable.jl")

function make_fig1_plot(cases, branches; zmax=6.0)
    dilaton_plot = Plots.plot(
        xlabel="z [GeV^-1]",
        ylabel="Phi(z) [GeV^2]",
        title="Fig. 1(a) dilaton profile",
        legend=:bottomright,
        lw=2,
    )

    z = collect(range(0.0, zmax; length=400))
    for case in cases
        style = case.dilaton_mode == :positive ? (:dash, 2.5) : (:solid, 3.0)
        Plots.plot!(dilaton_plot, z, dilaton.(z, Ref(case)); label=case.name, linestyle=style[1], lw=style[2])
    end

    sigma_plot = Plots.plot(
        xlabel="T [MeV]",
        ylabel="sigma [GeV^3]",
        title="Fig. 1(b) chiral condensate",
        legend=:topright,
        lw=2.5,
    )

    for branch in branches
        T_mev = branch.temperatures ./ GEV_PER_MEV
        Plots.plot!(sigma_plot, T_mev, branch.sigmas; label=branch.case.name)
    end

    return Plots.plot(dilaton_plot, sigma_plot; layout=(1, 2), size=(1200, 450))
end
