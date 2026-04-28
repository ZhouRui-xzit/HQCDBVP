using BifurcationKit, Plots
using DataFrames, CSV
function main()
    F(x, p) = @. p[1] + x - x^3 / 3

    prob = BifurcationProblem(
        F,
        [-2.0],      # 初始解 x0
        [-1.0],      # 初始参数 p
        1;           # 第 1 个参数作为 continuation parameter
        record_from_solution = (x, p; k...) -> x[1]
    )

    opts = ContinuationPar(
        p_min = -1.0,
        p_max = 1.0,
        dsmax = 0.1,
        max_steps = 1000
    )

    br = continuation(prob, PALC(), opts)
    df = DataFrame(
    x = br.branch.x,
    p = br.branch.param
)
    CSV.write("data/bifurcation_data.csv", df)
    plot(br);
    savefig("data/bifurcation_diagram.svg")
    println("Bifurcation diagram saved as data/bifurcation_diagram.svg")
end