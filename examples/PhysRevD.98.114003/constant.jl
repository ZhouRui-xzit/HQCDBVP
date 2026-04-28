const ZETA_3COLOR = sqrt(3) / (2 * pi)
const GEV_PER_MEV = 1.0e-3

temperature_from_zh(zh::Real) = 1 / (pi * zh)
zh_from_temperature(T::Real) = 1 / (pi * T)

"""
    make_2plus1_case(name; kwargs...)

Parameter set for Phys. Rev. D 98, 114003.
Energy-like inputs are in GeV unless the keyword ends with `_mev`.
"""
function make_2plus1_case(
    name;
    mu_mev::Real=3.336,
    ms_mev::Real=95.0,
    mu_g::Real=0.440,
    mu_c::Real=1.180,
    gamma::Real=-22.6,
    lambda::Real=16.8,
    nz::Integer=96,
    yu_guess::Real=0.25,
    ys_guess::Real=0.55,
    uv_fit_points::Integer=10,
)
    return (
        name = String(name),
        mu = float(mu_mev) * GEV_PER_MEV,
        ms = float(ms_mev) * GEV_PER_MEV,
        mu_g = float(mu_g),
        mu_c = float(mu_c),
        gamma = float(gamma),
        lambda = float(lambda),
        nz = Int(nz),
        yu_guess = float(yu_guess),
        ys_guess = float(ys_guess),
        uv_fit_points = Int(uv_fit_points),
    )
end
