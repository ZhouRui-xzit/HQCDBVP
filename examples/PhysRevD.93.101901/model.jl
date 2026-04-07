include("../../src/hqcdbvp.jl")
include("constant.jl")

function dilaton(z, p)
    if p.dilaton_mode == :positive
        return p.mu0 * z^2
    elseif p.dilaton_mode == :interpolating
        return -p.mu1 * z^2 + (p.mu1 + p.mu0) * z^2 * tanh(p.mu2 * z^2)
    else
        throw(ArgumentError("unsupported dilaton mode: $(p.dilaton_mode)"))
    end
end

function dilaton_prime(z, p)
    if p.dilaton_mode == :positive
        return 2 * p.mu0 * z
    elseif p.dilaton_mode == :interpolating
        tanh_term = tanh(p.mu2 * z^2)
        sech2_term = sech(p.mu2 * z^2)^2
        return -2 * p.mu1 * z + (p.mu1 + p.mu0) * (2 * z * tanh_term + 2 * p.mu2 * z^3 * sech2_term)
    else
        throw(ArgumentError("unsupported dilaton mode: $(p.dilaton_mode)"))
    end
end

blackening_u(u) = 1 - u^4
blackening_u_prime(u) = -4 * u^3

function dVdchi(chi, p)
    return -3 * chi + 3 * p.v3 * chi^2 + 4 * p.v4 * chi^3
end

function alpha_source(p)
    return p.mq * FIG1_ZETA * p.zh
end

function uv_log_coefficient(p)
    α = alpha_source(p)
    return α * (2 * α^2 * p.v4 - p.mu1 * p.zh^2)
end

function uv_log_chi_term(u, p)
    u == 0 && return 0.0
    return uv_log_coefficient(p) * u^3 * log(u)
end

function chi_from_y(y, u, p)
    return alpha_source(p) * u + u^3 * y + uv_log_chi_term(u, p)
end

function chi_u_from_y(y, y_u, u, p)
    if u == 0
        return alpha_source(p)
    end
    blog = uv_log_coefficient(p) * u^2 * (3 * log(u) + 1)
    return alpha_source(p) + 3 * u^2 * y + u^3 * y_u + blog
end

function chi_uu_from_y(y, y_u, y_uu, u)
    return 6 * u * y + 6 * u^2 * y_u + u^3 * y_uu
end

function chi_uu_from_y(y, y_u, y_uu, u, p)
    if u == 0
        return 0.0
    end
    blog = uv_log_coefficient(p) * u * (6 * log(u) + 5)
    return 6 * u * y + 6 * u^2 * y_u + u^3 * y_uu + blog
end

function fig1_bulk_equation!(res, uvec, duvec, d2uvec, u, p)
    y = uvec[1]
    y_u = duvec[1]
    y_uu = d2uvec[1]

    chi = chi_from_y(y, u, p)
    chi_u = chi_u_from_y(y, y_u, u, p)
    chi_uu = chi_uu_from_y(y, y_u, y_uu, u, p)

    z = p.zh * u
    f = blackening_u(u)
    f_u = blackening_u_prime(u)
    as_u = -1 / u
    phi_u = p.zh * dilaton_prime(z, p)
    potential_prefactor = 1 / u^2

    res[1] = f * chi_uu + (3 * as_u * f - phi_u * f + f_u) * chi_u - potential_prefactor * dVdchi(chi, p)
    return res
end

function fig1_uv_bc!(res, uvec, duvec, d2uvec, u, p)
    # chi = alpha*u + u^3*y with y = beta + O(u^2), so y_u(0) = 0.
    res[1] = duvec[1]
    return res
end

function fig1_ir_bc!(res, uvec, duvec, d2uvec, u, p)
    y = uvec[1]
    y_u = duvec[1]
    chi = chi_from_y(y, u, p)
    chi_u = chi_u_from_y(y, y_u, u, p)

    # Horizon regularity in u-variable: 4 * chi_u(1) + dV/dchi = 0.
    res[1] = 4 * chi_u + dVdchi(chi, p)
    return res
end

function make_fig1_problem(case, temperature; nz::Integer=case.nz)
    T = float(temperature)
    zh = zh_from_temperature(T)
    grid = make_grid(0.0, 1.0, nz)
    params = merge(case, (; zh=zh, temperature=T))
    return make_bvp_problem(
        fig1_bulk_equation!,
        fig1_uv_bc!,
        fig1_ir_bc!,
        grid;
        nfields=1,
        p=params,
        field_names=[:y],
    )
end
