# HQCDBVP.jl

`HQCDBVP.jl` 是一个面向全息 QCD 以及相关径向耦合常微分方程组的轻量级 Julia 边值问题工具集。

当前仓库版本为 **v0.1**。

## v0.1 的目标范围

这个版本刻意保持小而透明，主要覆盖以下工作流：

- 有限区间上的 Chebyshev-Lobatto 配置法
- 一阶、二阶谱微分矩阵
- 多场耦合 ODE 系统
- 用户显式给出的 UV / IR 边界残差
- 基于 `NonlinearSolve.jl` 的非线性代数方程组求解
- 基于参数扫描的简单 natural continuation
- 解结果的绘图与 CSV 导出

目前代码采用“普通函数文件 + `include`”的组织方式，而不是注册包形式。主入口为：

```julia
include("../src/hqcdbvp.jl")
```

## 设计取舍

`v0.1` 暂时 **不自动处理** 以下内容：

- 正则奇点或更一般的奇异端点
- UV 渐近展开自动生成
- horizon 正则条件自动生成
- pseudo-arclength continuation
- 多区间谱元
- 自适应阶数
- PDE 支持

因此默认假设是：你已经将原问题改写成数值上正则的 ODE 系统，并且可以直接写出左右边界的残差形式。

## 文件结构

- `src/hqcdbvp.jl`：总入口，统一 `using` 和 `include`
- `src/grid.jl`：节点、权重与谱微分矩阵
- `src/problem.jl`：问题构造、参数更新、初值与边界辅助函数
- `src/solver.jl`：残差组装、非线性求解、natural continuation
- `src/plotting.jl`：绘图与 CSV 导出
- `scripts/`：可直接运行的示例与测试脚本

## 核心接口

### 1. 配置网格

```julia
grid = make_grid(a, b, n)
```

这会在区间 `[a, b]` 上生成 Chebyshev-Lobatto 配置网格。
节点优先借助 `FastGaussQuadrature.jl` 生成，而谱微分矩阵按显式公式构造。

### 2. 构造问题

```julia
problem = make_bvp_problem(f!, bc_left!, bc_right!, grid;
    nfields=2,
    p=make_model_params(...),
    field_names=[:phi, :chi],
)
```

残差函数签名为：

```julia
f!(res, u, du, d2u, x, p)
bc_left!(res, u, du, d2u, x, p)
bc_right!(res, u, du, d2u, x, p)
```

其中：

- `u`、`du`、`d2u` 是单个配置点上的长度为 `nfields` 的向量视图
- `res` 也是长度为 `nfields` 的向量，需要原地写入
- `p` 是模型参数包，推荐用 `NamedTuple`

### 3. 模型参数

```julia
p = make_model_params(lambda=1.0, zh=1.2)
p2 = update_model_params(p; lambda=1.5)
```

### 4. 初值

单场常数初值：

```julia
guess = constant_guess(problem; value=0.0)
```

多场堆叠初值：

```julia
guess = stacked_guess(phi0, chi0, g0)
```

### 5. 求解

```julia
result = solve_bvp(problem, guess; abstol=1e-12, reltol=1e-12, maxiters=200)
```

返回值是一个 `NamedTuple`，主要字段包括：

- `converged`
- `retcode`
- `u`
- `du`
- `d2u`
- `residual`
- `residual_norm`
- `params`
- `field_names`

### 6. 边界条件辅助函数

Dirichlet：

```julia
uv_bc! = make_dirichlet_bc([0.0, 1.0])
```

Robin：

```julia
ir_bc! = make_robin_bc([0.0, 0.0], [1.0, 1.0], [1.0, -1.0])
```

它对应的形式是：

```text
alpha .* u + beta .* du - gamma = 0
```

这对 HQCD 中 UV / IR 两端采用不同类型边界条件时比较方便。

### 7. Natural continuation

```julia
scan = continuation_solve(problem, :lambda, 0.0:0.2:2.0, guess;
    abstol=1e-12,
    reltol=1e-12,
    maxiters=200,
)
```

这会沿给定参数序列逐点求解，并自动把上一步解作为下一步初值。

### 8. 绘图与导出

```julia
plt = plot_solution(result; field_names=[:phi, :chi], quantity=:u)
save_solution_plot(result, "solution.png"; field_names=[:phi, :chi])
save_solution_csv(result, "solution.csv"; field_names=[:phi, :chi])
```

## 示例脚本

可以直接运行：

```powershell
julia scripts/smoke_test.jl
julia scripts/coupled_two_field_example.jl
julia scripts/hqcd_uv_ir_mixed_bc.jl
julia scripts/hqcd_three_field_template.jl
julia scripts/continuation_demo.jl
julia scripts/plotting_demo.jl
```

这些脚本分别覆盖：

- 单场 smoke test
- 双场耦合示例
- UV / IR 混合边界示例
- 三场 HQCD 风格模板
- simple continuation 示例
- 绘图与 CSV 导出演示

## 依赖

脚本默认假设当前 Julia 环境已经安装以下包：

- `FastGaussQuadrature`
- `NonlinearSolve`
- `Plots`
- `LaTeXStrings`
- `DataFrames`
- `CSV`

## 版本

当前发布目标版本：**v0.1**。