module Problems

export make_model_params,
    update_model_params,
    remake_problem,
    make_bvp_problem,
    constant_guess,
    stacked_guess,
    make_dirichlet_bc,
    make_robin_bc,
    flatten_state,
    reshape_state

"""
    make_model_params(; kwargs...)

显式构造模型参数包，返回 `NamedTuple`。
这样在 HQCD 脚本里可以统一把势函数系数、耦合常数、温度参数等收进 `p`。
"""
make_model_params(; kwargs...) = (; kwargs...)

"""
    update_model_params(p; kwargs...)

在原参数包基础上更新部分参数，返回新的 `NamedTuple`。
"""
function update_model_params(p; kwargs...)
    p === nothing && return (; kwargs...)
    return merge(p, (; kwargs...))
end

"""
    remake_problem(problem; p=problem.p, kwargs...)

在保留原问题结构的前提下替换参数或元信息。
"""
function remake_problem(problem; p=problem.p, field_names=problem.field_names)
    return (
        f! = problem.f!,
        bc_left! = problem.bc_left!,
        bc_right! = problem.bc_right!,
        grid = problem.grid,
        nfields = problem.nfields,
        p = p,
        field_names = field_names,
    )
end

"""
    make_bvp_problem(f!, bc_left!, bc_right!, grid; nfields, p=nothing, field_names=nothing)

构造边值问题描述。
- `f!` 签名：`f!(res, u, du, d2u, x, p)`
- `bc_left!` 与 `bc_right!` 签名相同
- `u`、`du`、`d2u` 都是长度为 `nfields` 的向量视图
- 左右边界各返回 `nfields` 个残差
- `field_names` 可选，用于提高多场脚本的可读性
"""
function make_bvp_problem(f!, bc_left!, bc_right!, grid; nfields::Integer, p=nothing, field_names=nothing)
    nfields > 0 || throw(ArgumentError("nfields 必须为正整数。"))
    if field_names !== nothing
        length(field_names) == nfields || throw(DimensionMismatch("field_names 的长度必须等于 nfields。"))
    end

    return (
        f! = f!,
        bc_left! = bc_left!,
        bc_right! = bc_right!,
        grid = grid,
        nfields = Int(nfields),
        p = p,
        field_names = field_names,
    )
end

"""
    constant_guess(problem; value=0.0)

生成常数初值矩阵，尺寸为 `(nfields, n + 1)`。
"""
function constant_guess(problem; value::Real=0.0)
    npoints = problem.grid.n + 1
    T = eltype(problem.grid.x)
    return fill(T(value), problem.nfields, npoints)
end

"""
    stacked_guess(columns...)

把多个场的节点值按行堆叠成初值矩阵。

示例：
`stacked_guess(phi0, chi0)` 会返回一个 `2 × npoints` 矩阵。
"""
function stacked_guess(columns::AbstractVector...)
    length(columns) > 0 || throw(ArgumentError("至少需要一个场的初值向量。"))
    npoints = length(columns[1])
    all(length(col) == npoints for col in columns) ||
        throw(DimensionMismatch("所有场的初值向量长度必须一致。"))
    return reduce(vcat, permutedims.(collect.(columns)))
end

"""
    make_dirichlet_bc(target)

生成 Dirichlet 边界条件函数。
`target` 是长度为 `nfields` 的目标值向量。
"""
function make_dirichlet_bc(target::AbstractVector)
    target_vec = collect(float.(target))
    return function (res, u, du, d2u, x, p)
        res .= u .- target_vec
        return res
    end
end

"""
    make_robin_bc(alpha, beta, gamma)

生成 Robin 边界条件函数，形式为
`alpha .* u + beta .* du - gamma = 0`。

这对 HQCD 中 UV / IR 两端采用不同类型边界条件很方便。
"""
function make_robin_bc(alpha::AbstractVector, beta::AbstractVector, gamma::AbstractVector)
    alpha_vec = collect(float.(alpha))
    beta_vec = collect(float.(beta))
    gamma_vec = collect(float.(gamma))
    n = length(alpha_vec)
    length(beta_vec) == n || throw(DimensionMismatch("beta 的长度必须与 alpha 一致。"))
    length(gamma_vec) == n || throw(DimensionMismatch("gamma 的长度必须与 alpha 一致。"))

    return function (res, u, du, d2u, x, p)
        @. res = alpha_vec * u + beta_vec * du - gamma_vec
        return res
    end
end

flatten_state(U::AbstractMatrix) = vec(U)

function reshape_state(problem, u::AbstractVector)
    npoints = problem.grid.n + 1
    expected = problem.nfields * npoints
    length(u) == expected || throw(DimensionMismatch("状态向量长度应为 $(expected)。"))
    return reshape(u, problem.nfields, npoints)
end

end
