using LinearAlgebra
using NonlinearSolve

"""
    residual_vector(problem, u)

组装整个非线性边值问题的残差向量。
残差布局为：左边界、内部节点、右边界。
"""
function residual_vector(problem, u::AbstractVector)
    U = reshape_state(problem, u)
    nfields = problem.nfields
    npoints = problem.grid.n + 1

    dU = U * transpose(problem.grid.D1)
    d2U = U * transpose(problem.grid.D2)
    res = zeros(eltype(u), nfields * npoints)

    offset = 0
    left_block = view(res, offset + 1:offset + nfields)
    problem.bc_left!(left_block, view(U, :, 1), view(dU, :, 1), view(d2U, :, 1), problem.grid.x[1], problem.p)
    offset += nfields

    for j in 2:npoints-1
        block = view(res, offset + 1:offset + nfields)
        problem.f!(block, view(U, :, j), view(dU, :, j), view(d2U, :, j), problem.grid.x[j], problem.p)
        offset += nfields
    end

    right_block = view(res, offset + 1:offset + nfields)
    problem.bc_right!(right_block, view(U, :, npoints), view(dU, :, npoints), view(d2U, :, npoints), problem.grid.x[npoints], problem.p)

    return res
end

"""
    solve_bvp(problem, guess; alg=NewtonRaphson(), kwargs...)

用 `NonlinearSolve` 求解配置法离散后的非线性代数方程组。
"""
function solve_bvp(problem, guess; alg=NewtonRaphson(), kwargs...)
    u0 = guess isa AbstractMatrix ? flatten_state(guess) : copy(guess)
    npoints = problem.grid.n + 1
    length(u0) == problem.nfields * npoints ||
        throw(DimensionMismatch("初值长度必须等于 nfields * (n + 1)。"))
    nres = problem.nfields * npoints

    function f!(res, u, p)
        res .= residual_vector(problem, u)
        return res
    end

    prob = NonlinearProblem(
        NonlinearFunction(f!; resid_prototype=zeros(eltype(u0), nres)),
        u0,
        problem.p,
    )
    sol = NonlinearSolve.solve(prob, alg; kwargs...)

    U = reshape_state(problem, sol.u)
    dU = U * transpose(problem.grid.D1)
    d2U = U * transpose(problem.grid.D2)
    res = residual_vector(problem, sol.u)
    residual_norm = norm(res, Inf)
    retcode_text = sprint(show, sol.retcode)
    abstol = get(kwargs, :abstol, 1e-10)
    converged = occursin("Success", retcode_text) || residual_norm <= max(10 * abstol, 1e-10)

    return (
        converged = converged,
        retcode = sol.retcode,
        grid = problem.grid,
        u = copy(U),
        du = copy(dU),
        d2u = copy(d2U),
        residual = res,
        residual_norm = residual_norm,
        solution = sol,
        field_names = problem.field_names,
        params = problem.p,
    )
end

"""
    continuation_solve(problem, param_name, param_values, guess; update_problem=nothing, kwargs...)

最简单的自然 continuation。
沿给定参数序列逐点求解，并把上一步解当作下一步初值。

- `param_name` 是参数名，例如 `:lambda`
- `param_values` 是扫描序列
- `update_problem` 可选，用于在每一步自定义如何重建 problem
"""
function continuation_solve(problem, param_name::Symbol, param_values, guess; update_problem=nothing, kwargs...)
    values = collect(param_values)
    length(values) > 0 || throw(ArgumentError("param_values 不能为空。"))

    current_guess = guess isa AbstractMatrix ? copy(guess) : reshape_state(problem, guess)
    results = Vector{Any}(undef, length(values))

    for (i, value) in enumerate(values)
        single_param = NamedTuple{(param_name,)}((value,))
        new_params = problem.p === nothing ? single_param : merge(problem.p, single_param)

        step_problem = if update_problem === nothing
            remake_problem(problem; p=new_params)
        else
            update_problem(problem, value, new_params)
        end

        result = solve_bvp(step_problem, current_guess; kwargs...)
        results[i] = (
            param_name = param_name,
            param_value = value,
            result = result,
        )

        result.converged || throw(ErrorException("continuation 在 $(param_name) = $(value) 处失败。"))
        current_guess = result.u
    end

    return (
        param_name = param_name,
        param_values = values,
        branches = results,
    )
end