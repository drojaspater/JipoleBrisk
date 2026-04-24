
include("../metrics.jl")

mutable struct GlobalParams
    a::Float64
    Rout::Float64
    cstartx::MVector{4,Float64}
    cstopx::MVector{4,Float64}
end

f::Float64 = 1.8
ch_mu = Float64[]
ch_I = Float64[]
ch_delta = Float64[]





function thindisk_region(Xi::MVec4, Xf::MVec4)::Bool
    """
    Checks if the geodesic segment crosses the thin disk region.
    Parameters:
    @Xi: Starting position vector in internal coordinates.
    @Xf: Ending position vector in internal coordinates.
    """
    _, th_i = bl_coord(Xi)
    r_f, th_f = bl_coord(Xf)
    
    midplane::Bool = (sign(th_i - π/2) != sign(th_f - π/2))
    em_region::Bool = (r_f > r_isco && r_f < Rout)
    return midplane && em_region
end
function radiating_region(X, Rh::Float64)
    return false
end


function GetTDBoundaryCondition(X::SVector{4, Float64}, Kcon::SVector{4, Float64}, a::Float64, Rh::Float64)
    r, _ = bl_coord(X)

    if(r > Rh)
        Temp::Float64 = 0.
        omega::Float64 = 0.
        Temp, omega = thindisk_vals(r, a)

        Ucon::MVec4 = zero(MVec4)
        Ucov::MVec4 = zero(MVec4)
        Bcon::MVec4 = zero(MVec4)
        Bcov::MVec4 = zero(MVec4)
        Ucon, Ucov, Bcon, Bcov = get_model_fourv(X, a)


        mu::Float64 = abs(cos(get_bk_angle(Kcon, Ucov, Bcon, Bcov)))
    
        nu::Float64 = get_fluid_nu(Kcon, Ucov)

        if(nu <= 0. || any(isnan(nu)) || any(isinf(nu)))
            println("At X = $X\n Kcon = $Kcon")
            println("Ucov = $Ucov")
            println("Kcon $Kcon")
            error("Frequency must be positive, got nu = $nu")
        end

        I = fbbpolemis!(nu, Temp, mu)
    else
        I = 0.0
    end
    return I
end

function thindisk_vals(r::Float64, a::Float64)
    """
    Computes the temperature and angular frequency for the thin disk model.
    Parameters:
    @r: Radial coordinate in KS/BL coordinates.
    """
    b::Float64 = 1. - 3. /r + 2. *a/r^(3/2)
    kc = krolikc(r, a)
    d::Float64 = r *r - 2. * r + a * a
    lc::Float64 = (r_isco * r_isco - 2. * a * sqrt(r_isco) + a * a) / (sqrt(r_isco) - 2. * sqrt(r_isco) + a)
    hc::Float64 = (2. * r - a * lc) / d
    ar::Float64 = (r * r + a * a)^2 - a * a * d * sin(π/2.)^2
    om::Float64 = 2. * a * r / ar

    #Start the disk at r_isco, the marginally stable orbit which N-K take as an inner boundary condition.
    #End it eventually.
    if(r > r_isco)
        omega = max(1. / (r^(3/2) + a), om)
    else
        omega = max((lc + a * hc) / (r * r + 2. * r * (1. + hc)), om)
    end

    if(r > r_isco && r < Rout)
        Temp = T0 * (kc / b / r^3)^(1. /4.)
    else
        Temp = T0 / 1.e5
    end
    return Temp, omega
end

function krolikc(r::Float64, a::Float64)
    """
    Computes the Krolik & Hawley (2002) disk model for a given radius and spin.
    Parameters:
    @r: Radial coordinate in KS/BL coordinates.
    @a: Spin parameter of the black hole.
    """
    y::Float64 = sqrt(r)
    yms::Float64 = sqrt(r_isco)
    y1::Float64 = 2. * cos(1. / 3. * (acos(a) - π))
    y2::Float64 = 2. * cos(1. / 3. * (acos(a) + π))
    y3::Float64 = -2. * cos(1. / 3. * acos(a))
    arg1::Float64 = 3. * a / (2. * y)
    arg2::Float64 = 3. * (y1 - a)^2 / (y * y1 * (y1 - y2) * (y1 - y3))
    arg3::Float64 = 3. * (y2 - a)^2 / (y * y2 * (y2 - y1) * (y2 - y3))
    arg4::Float64 = 3. * (y3 - a)^2 / (y * y3 * (y3 - y1) * (y3 - y2))

    return 1. - yms / y - arg1 * log(y / yms) - arg2 * log((y - y1) / (yms - y1)) - arg3 * log((y - y2) / (yms - y2)) - arg4 * log((y - y3) / (yms - y3))
end

function fbbpolemis!(nu::Float64, Temp::Float64, cosne::Float64)
    """
    Computes the emissivity for the thin disk model.
    Parameters:
    @nu: Frequency in internal coordinates.
    @Temp: Temperature in internal coordinates.
    @mu: Cosine of the angle between the photon direction and the magnetic field.
    """
    I = f^(-4.) * bnu(nu, Temp * f)

    
    if(nu <= 0. || any(isnan(nu)) || any(isinf(nu)))
        error("Frequency must be positive, got nu = $nu")
    end
    interpI::Float64 = 0.0
    interpDel::Float64 = 0.0
    interpI, interpDel = interp_chandra(cosne)

    I *= interpI/(nu * nu * nu)

    return I
end

function bnu(nu::Float64, Temp::Float64)
    """
    Computes the Planck function for a given frequency and temperature.
    Parameters:
    @nu: Frequency.
    @TTemp: Temperature
    """
    return 2 * HPL * nu^3 / (CL^2) / (exp(HPL * nu / (KBOL * Temp)) - 1)
end



function interp_chandra(mu::Float64)
    """
    Interpolates the Chandra emissivity and absorption coefficient.
    Parameters:
    @mu: Cosine of the angle between the photon direction and the magnetic field.
    """
    indx::Int64 = 1
    weight::Float64, indx = get_weight!(ch_mu, mu, indx)
    i = (1. - weight) * ch_I[indx] + weight * ch_I[indx + 1]
    del = (1. - weight) * ch_delta[indx] + weight * ch_delta[indx + 1]
    return i, del
end


function get_weight!(xx::Vector{Float64}, x::Float64, jlo::Int64)
    """
    Computes the weight for interpolation based on the given x value and the xx array.

    Parameters:
    @xx: Vector of x values for interpolation.
    @x: The x value for which to compute the weight.
    @jlo: Reference to the index of the lower bound for interpolation.
    """
    while(xx[jlo] < x)
        jlo += 1
    end
    jlo -= 1
    return (x - xx[jlo]) / (xx[jlo + 1] - xx[jlo]), jlo
end


function load_chandra_tab24()
    """
    Loads the Chandra emissivity and absorption coefficient data from a file.
    The data is expected to be in the format:
    mu I delta
    where mu is the cosine of the angle, I is the emissivity, and delta is the absorption coefficient.
    """
    fname::String = joinpath(@__DIR__, "../../Tables/ch24_vals.txt")
    println("Current path is $(pwd())")
    if !isfile(fname)
        @error "Error reading file $fname!"
        error("File not found: $fname")
    end
    vals::IO = open(fname, "r")
    npts::Int = 21  # Number of points in the table
    ch_mu::Vector{Float64} = zeros(npts)
    ch_I::Vector{Float64} = zeros(npts)
    ch_delta::Vector{Float64} = zeros(npts)
    for i in 1:npts
        line = readline(vals)
        vals_split = split(line)
        ch_mu[i] = parse(Float64, vals_split[1])
        ch_I[i] = parse(Float64, vals_split[2])
        ch_delta[i] = parse(Float64, vals_split[3])
    end
    close(vals)
    @info "Chandra table loaded successfully from $fname"
    return ch_mu, ch_I, ch_delta
end
ch_mu, ch_I, ch_delta = load_chandra_tab24()

@inline function get_model_fourv(X::MVec4, a::Float64)
    gcov::MMat4 = gcov_func(X, a)
    r, _ = bl_coord(X)
    _, omega = thindisk_vals(r, a)
    Ucon::MVec4 = MVec4(undef)
    Ucov::MVec4 = MVec4(undef)
    Bcon::MVec4 = MVec4(undef)
    Bcov::MVec4 = MVec4(undef)
    Ucon[1] = sqrt(-1. / (gcov[1, 1] + 2. * gcov[1, 4] * omega + gcov[4, 4] * omega * omega))
    Ucon[2] = 0.0
    Ucon[3] = 0.0
    Ucon[4] = omega * Ucon[1]
    Ucov = flip_index(Ucon, gcov)
    Bcon = calc_polvec(X, a)
    Bcov = flip_index(Bcon, gcov)
    return Ucon, Ucov, Bcon, Bcov
end

function calc_polvec(X::MVec4, a::Float64)
    """
    Calculates the polarization vector in Kerr-Schild coordinates.
    Parameters:
    @X: Position vector in internal coordinates.
    @Kcon: Covariant 4-vector of the photon in internal coordinates.
    @a: Spin parameter of the black hole.
    @fourf: Output vector for the polarization vector.
    """
    fourf_bl::MVec4 = zeros(MVec4)
    fourf_bl[3] = 1.0
    fourf_ks::MVec4 = zeros(MVec4)
    # Transform to KS and then to eKS
    fourf_ks = bl_to_ks(X, fourf_bl, a)
    fourf = vec_to_ks(X, fourf_ks)
    # Normalize the polarization vector
    gcov::MMat4 = gcov_func(X, a)
    fourf_cov::MVec4 = zeros(MVec4)

    fourf_cov = flip_index(fourf, gcov)

    normf = sqrt(fourf[1] * fourf_cov[1] + 
                 fourf[2] * fourf_cov[2] + 
                 fourf[3] * fourf_cov[3] + 
                 fourf[4] * fourf_cov[4])
    
    for i in 1:4
        fourf[i] /= normf
    end

    return fourf
end

