include("constant.jl")

blackening_u(u) = 1 - u^4
blackening_u_prime(u) = -4 * u^3

dilaton_u_prime(u, p) = 2 * p.mu_g^2 * p.zh^2 * u
bulk_mass5_squared(u, p) = -3 - p.mu_c^2 * p.zh^2 * u^2

function uv_source_u(p)
    return p.mu * ZETA_3COLOR * p.zh
end

function uv_source_s(p)
    return p.ms * ZETA_3COLOR * p.zh
end

function uv_linear_u_coefficient(p)
    return -p.mu * p.ms * p.gamma * ZETA_3COLOR^2 * p.zh^2 / (2 * sqrt(2))
end

function uv_linear_s_coefficient(p)
    return -p.mu^2 * p.gamma * ZETA_3COLOR^2 * p.zh^2 / (2 * sqrt(2))
end

function uv_log_u_coefficient(p)
    raw = p.mu * ZETA_3COLOR * (
        -p.ms^2 * p.gamma^2 * ZETA_3COLOR^2 -
        p.mu^2 * p.gamma^2 * ZETA_3COLOR^2 +
        8 * p.mu^2 * ZETA_3COLOR^2 * p.lambda +
        16 * p.mu_g^2 -
        8 * p.mu_c^2
    ) / 16
    return raw * p.zh^3
end

function uv_log_s_coefficient(p)
    raw = (
        -p.ms * p.mu^2 * p.gamma^2 * ZETA_3COLOR^3 +
        4 * p.ms^3 * ZETA_3COLOR^3 * p.lambda +
        8 * p.ms * ZETA_3COLOR * p.mu_g^2 -
        4 * p.ms * ZETA_3COLOR * p.mu_c^2
    ) / 8
    return raw * p.zh^3
end

function uv_log_chi_term(u, coeff)
    u == 0 && return 0.0
    return coeff * u^3 * log(u)
end

function chi_from_y(y, u, source, quadratic, log_coeff)
    return source * u + quadratic * u^2 + u^3 * y + uv_log_chi_term(u, log_coeff)
end

function chi_u_from_y(y, y_u, u, source, quadratic, log_coeff)
    if u == 0
        return source
    end
    blog = log_coeff * u^2 * (3 * log(u) + 1)
    return source + 2 * quadratic * u + 3 * u^2 * y + u^3 * y_u + blog
end

function chi_uu_from_y(y, y_u, y_uu, u, quadratic, log_coeff)
    if u == 0
        return 2 * quadratic
    end
    blog = log_coeff * u * (6 * log(u) + 5)
    return 2 * quadratic + 6 * u * y + 6 * u^2 * y_u + u^3 * y_uu + blog
end

function two_plus_one_bulk_equation!(res, uvec, duvec, d2uvec, u, p)
    yu, ys = uvec
    yu_u, ys_u = duvec
    yu_uu, ys_uu = d2uvec

    srcu = uv_source_u(p)
    srcs = uv_source_s(p)
    quadu = uv_linear_u_coefficient(p)
    quads = uv_linear_s_coefficient(p)
    logu = uv_log_u_coefficient(p)
    logs = uv_log_s_coefficient(p)

    chiu = chi_from_y(yu, u, srcu, quadu, logu)
    chis = chi_from_y(ys, u, srcs, quads, logs)
    chiu_u = chi_u_from_y(yu, yu_u, u, srcu, quadu, logu)
    chis_u = chi_u_from_y(ys, ys_u, u, srcs, quads, logs)
    chiu_uu = chi_uu_from_y(yu, yu_u, yu_uu, u, quadu, logu)
    chis_uu = chi_uu_from_y(ys, ys_u, ys_uu, u, quads, logs)

    f = blackening_u(u)
    fp = blackening_u_prime(u)
    drift = fp + f * (-3 / u - dilaton_u_prime(u, p))
    m5 = bulk_mass5_squared(u, p)
    inv_u2 = 1 / u^2
    det_prefactor = p.gamma / (2 * sqrt(2))

    res[1] = f * chiu_uu + drift * chiu_u -
             inv_u2 * (m5 * chiu + p.lambda * chiu^3 + det_prefactor * chiu * chis)
    res[2] = f * chis_uu + drift * chis_u -
             inv_u2 * (m5 * chis + p.lambda * chis^3 + det_prefactor * chiu^2)
    return res
end

function two_plus_one_uv_bc!(res, uvec, duvec, d2uvec, u, p)
    res[1] = duvec[1]
    res[2] = duvec[2]
    return res
end

function two_plus_one_ir_bc!(res, uvec, duvec, d2uvec, u, p)
    yu, ys = uvec
    yu_u, ys_u = duvec
    srcu = uv_source_u(p)
    srcs = uv_source_s(p)
    quadu = uv_linear_u_coefficient(p)
    quads = uv_linear_s_coefficient(p)
    logu = uv_log_u_coefficient(p)
    logs = uv_log_s_coefficient(p)
    chiu = chi_from_y(yu, u, srcu, quadu, logu)
    chis = chi_from_y(ys, u, srcs, quads, logs)
    chiu_u = chi_u_from_y(yu, yu_u, u, srcu, quadu, logu)
    chis_u = chi_u_from_y(ys, ys_u, u, srcs, quads, logs)

    res[1] = -12 * chiu - 4 * p.zh^2 * p.mu_c^2 * chiu +
             sqrt(2) * p.gamma * chis * chiu + 4 * p.lambda * chiu^3 +
             16 * chiu_u
    res[2] = -12 * chis - 4 * p.zh^2 * p.mu_c^2 * chis +
             sqrt(2) * p.gamma * chiu^2 + 4 * p.lambda * chis^3 +
             16 * chis_u
    return res
end

function make_2plus1_problem(case, temperature; nz::Integer=case.nz)
    T = float(temperature)
    zh = zh_from_temperature(T)
    grid = make_grid(0.0, 1.0, nz)
    params = merge(case, (; zh=zh, temperature=T))
    return make_bvp_problem(
        two_plus_one_bulk_equation!,
        two_plus_one_uv_bc!,
        two_plus_one_ir_bc!,
        grid;
        nfields=2,
        p=params,
        field_names=[:yu, :ys],
    )
end
