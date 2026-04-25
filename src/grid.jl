module Grid

using FastGaussQuadrature

export make_chebyshev_lobatto_nodes,
    make_chebyshev_diff_matrices,
    make_grid

"""
    make_chebyshev_lobatto_nodes(n)

生成标准区间 `[-1, 1]` 上的 Chebyshev--Lobatto 节点与权重。
内部节点优先使用 `FastGaussQuadrature` 生成。
"""
function make_chebyshev_lobatto_nodes(n::Integer)
    n >= 2 || throw(ArgumentError("n 必须至少为 2。"))

    xi_inner, _ = gausschebyshevu(n - 1)
    xi = vcat(-1.0, reverse(collect(xi_inner)), 1.0)

    w = fill(pi / n, n + 1)
    w[1] = pi / (2 * n)
    w[end] = pi / (2 * n)

    return xi, w
end

"""
    make_chebyshev_diff_matrices(xi, a, b)

根据 Chebyshev--Lobatto 节点构造物理区间 `[a, b]` 上的一阶、二阶谱微分矩阵。
这部分需要按谱方法公式显式构造，`FastGaussQuadrature` 不直接提供微分矩阵。
"""
function make_chebyshev_diff_matrices(xi::AbstractVector, a::Real, b::Real)
    a < b || throw(ArgumentError("要求区间满足 a < b。"))

    T = promote_type(eltype(xi), typeof(float(a)), typeof(float(b)))
    n = length(xi) - 1
    c = ones(T, n + 1)
    c[1] = 2
    c[end] = 2
    c .*= T[isodd(k) ? -1 : 1 for k in 0:n]

    Dxi = Matrix{T}(undef, n + 1, n + 1)
    for i in 1:n+1
        for j in 1:n+1
            if i == j
                Dxi[i, j] = zero(T)
            else
                Dxi[i, j] = (c[i] / c[j]) / (xi[i] - xi[j])
            end
        end
    end

    for i in 2:n
        Dxi[i, i] = -xi[i] / (2 * (1 - xi[i]^2))
    end
    Dxi[1, 1] = -(2 * n^2 + 1) / 6
    Dxi[end, end] = (2 * n^2 + 1) / 6

    scale = T(2) / (T(b) - T(a))
    D1 = scale .* Dxi
    D2 = D1 * D1
    return D1, D2
end

"""
    make_grid(a, b, n)

生成物理区间 `[a, b]` 上的配置网格。
返回 `NamedTuple`，便于脚本直接调用。
"""
function make_grid(a::Real, b::Real, n::Integer)
    a < b || throw(ArgumentError("要求区间满足 a < b。"))
    xi, w = make_chebyshev_lobatto_nodes(n)
    x = @. (a + b) / 2 + (b - a) / 2 * xi
    D1, D2 = make_chebyshev_diff_matrices(xi, a, b)

    return (
        a = float(a),
        b = float(b),
        n = Int(n),
        xi = collect(float.(xi)),
        x = collect(float.(x)),
        w = collect(float.(w)),
        D1 = D1,
        D2 = D2,
    )
end

end
