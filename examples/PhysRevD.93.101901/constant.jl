const FIG1_ZETA = sqrt(3) / (2 * pi)
const GEV_PER_MEV = 1.0e-3

"""
    make_fig1_case(name; kwargs...)

Build one parameter set for the two-flavor Fig. 1 chiral condensate scan.
All energy-like inputs are in GeV units unless explicitly marked as `*_mev`.
"""
function make_fig1_case(
    name;
    dilaton_mode::Symbol=:interpolating,
    mq_mev::Real=0.0,
    v3::Real=0.0,
    v4::Real=8.0,
    mu0::Real=0.43^2,
    mu1::Real=0.83^2,
    mu2::Real=0.176^2,
    nz::Integer=64,
    y_guess::Real=0.02,
    uv_fit_points::Integer=8,
    uv_fit_windows::Tuple=(6, 8, 10, 12),
)
    return (
        name = String(name),
        dilaton_mode = dilaton_mode,
        mq = float(mq_mev) * GEV_PER_MEV,
        v3 = float(v3),
        v4 = float(v4),
        mu0 = float(mu0),
        mu1 = float(mu1),
        mu2 = float(mu2),
        nz = Int(nz),
        y_guess = float(y_guess),
        uv_fit_points = Int(uv_fit_points),
        uv_fit_windows = uv_fit_windows,
    )
end

temperature_from_zh(zh::Real) = 1 / (pi * zh)
zh_from_temperature(T::Real) = 1 / (pi * T)
