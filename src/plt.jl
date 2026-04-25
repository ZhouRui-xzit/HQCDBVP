module plt

using CSV
using DataFrames
using Plots

export result_field_names,
    make_solution_dataframe,
    save_solution_csv,
    plot_solution,
    save_solution_plot

"""
    result_field_names(result; field_names=nothing)

返回结果中各场的显示名称。
若未显式提供，则自动生成 `field1`, `field2`, ...。
"""
function result_field_names(result; field_names=nothing)
    if field_names !== nothing
        return string.(field_names)
    end
    if haskey(result, :field_names) && result.field_names !== nothing
        return string.(result.field_names)
    end
    nfields = size(result.u, 1)
    return ["field$(i)" for i in 1:nfields]
end

"""
    make_solution_dataframe(result; field_names=nothing, include_du=true, include_d2u=false)

把求解结果整理成 `DataFrame`，便于保存或后处理。
"""
function make_solution_dataframe(result; field_names=nothing, include_du=true, include_d2u=false)
    names_ = result_field_names(result; field_names=field_names)
    df = DataFrame(x = result.grid.x)

    for (i, name) in enumerate(names_)
        df[!, Symbol(name)] = vec(result.u[i, :])
        if include_du
            df[!, Symbol(name * "_d")] = vec(result.du[i, :])
        end
        if include_d2u
            df[!, Symbol(name * "_dd")] = vec(result.d2u[i, :])
        end
    end

    return df
end

"""
    save_solution_csv(result, path; kwargs...)

把结果保存为 CSV。
"""
function save_solution_csv(result, path::AbstractString; kwargs...)
    df = make_solution_dataframe(result; kwargs...)
    CSV.write(path, df)
    return path
end

"""
    plot_solution(result; field_names=nothing, quantity=:u, title=nothing, xlabel="x", ylabel=nothing)

绘制场函数、导数或二阶导数。
- `quantity=:u` 绘制场本身
- `quantity=:du` 绘制一阶导
- `quantity=:d2u` 绘制二阶导
"""
function plot_solution(result; field_names=nothing, quantity=:u, title=nothing, xlabel="x", ylabel=nothing)
    names_ = result_field_names(result; field_names=field_names)

    values = if quantity == :u
        result.u
    elseif quantity == :du
        result.du
    elseif quantity == :d2u
        result.d2u
    else
        throw(ArgumentError("quantity 只能是 :u, :du 或 :d2u。"))
    end

    default_ylabel = quantity == :u ? "field" : quantity == :du ? "d(field)/dx" : "d2(field)/dx2"
    plt = plot(; xlabel=xlabel, ylabel=something(ylabel, default_ylabel), title=title, legend=:best)

    for (i, name) in enumerate(names_)
        plot!(plt, result.grid.x, vec(values[i, :]); lw=2, label=name)
    end

    return plt
end

"""
    save_solution_plot(result, path; kwargs...)

保存结果图像。
"""
function save_solution_plot(result, path::AbstractString; kwargs...)
    plt = plot_solution(result; kwargs...)
    savefig(plt, path)
    return path
end

end
