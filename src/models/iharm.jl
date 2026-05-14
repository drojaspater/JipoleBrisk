using HDF5
using Printf

# --- Constants for Primitives ---
const VALID_PRIMS = ["RHO", "UU", "U1", "U2", "U3", "B1", "B2", "B3"]
const USE_GEODESIC_SIGMACUT = true
const M_unit = 3.e26
const RHO_unit = M_unit / L_unit^3  # Density unit in g/cm^3
const U_unit = RHO_unit * CL^2  # Internal energy density unit in erg
const B_unit = CL * sqrt(4 * π * RHO_unit)  # Magnetic field unit in Gauss


# --- Data Storage Structure ---
"""
    IharmData

A struct to hold the data from an iharm3d simulation dump.
The data is stored in a dictionary where keys are the primitive names
(e.g., "RHO") and values are 3D arrays.
"""
# struct IharmData
#     primitives::Dict{String, Array{Float64, 3}}
# end

struct IharmData
    t::Float64
    RHO::Array{Float64,3}
    UU::Array{Float64,3}
    U1::Array{Float64,3}
    U2::Array{Float64,3}
    U3::Array{Float64,3}
    B1::Array{Float64,3}
    B2::Array{Float64,3}
    B3::Array{Float64,3}
    ne::Array{Float64,3}
    b::Array{Float64,3}
    θe::Array{Float64,3}
    sigma::Array{Float64,3}
    beta::Array{Float64,3}
    dθedRhi::Array{Float64,3}
end



using HDF5
using Printf
using StaticArrays

# --- Constants ---
const METRIC_MKS = 0
const METRIC_BHACMKS = 1
const METRIC_FMKS = 2
const METRIC_MKS3 = 3
const METRIC_EKS = 4

const METRIC_MINKOWSKI = 5
const METRIC_EMINKOWSKI = 6

const ELECTRONS_TFLUID = 3
const USE_FIXED_TPTE = false
const USE_MIXED_TPTE = true 


# Holds the physics/metric parameters (formerly C globals)
mutable struct GlobalParams
    metric::Int
    ELECTRONS::Int
    RADIATION::Int
    
    # Physics
    gam::Float64
    game::Float64
    gamp::Float64
    Te_unit::Float64
    Thetae_unit::Float64
    
    # Weights
    mu_i::Float64
    mu_e::Float64
    mu_tot::Float64
    Ne_factor::Float64

    # Radiation units
    M_unit::Float64
    T_unit::Float64
    L_unit::Float64
    MBH::Float64
    tp_over_te::Float64

    # Metric Params
    a::Float64
    hslope::Float64
    Rin::Float64
    Rout::Float64
    poly_xt::Float64
    poly_alpha::Float64
    mks_smooth::Float64
    poly_norm::Float64
    
    # MKS3 specific
    mks3R0::Float64
    mks3H0::Float64
    mks3MY1::Float64
    mks3MY2::Float64
    mks3MP0::Float64

    #Grid Specific
    N1::Int64
    N2::Int64
    N3::Int64
    dx::MVector{4,Float64}
    startx::MVector{4,Float64}
    stopx::MVector{4,Float64}
    cstartx::MVector{4,Float64}
    cstopx::MVector{4,Float64}

    rmin_geo::Float64
    rmax_geo::Float64
end

# Default constructor for Params
function GlobalParams()
    return GlobalParams(0, 0, 0, 
                        5/3, 4/3, 5/3, 1.0, 1.0, # gam defaults
                        1.0, 1.0, 1.0, 1.0,      # weights defaults
                        1.0, 1.0, 1.0, 1.0, 3.0, # units defaults
                        0.0, 1.0, 0.0, 100.0,    # metric generic defaults
                        0.82, 14.0, 0.5, 1.0,    # fmks defaults
                        0.0, 0.0, 0.0, 0.0, 0.0, # mks3 defaults
                        0, 0, 0,                  # grid sizes
                        MVector{4,Float64}(0.0,0.0,0.0,0.0), # dx
                        MVector{4,Float64}(0.0,0.0,0.0,0.0), # startx
                        MVector{4,Float64}(0.0,0.0,0.0,0.0), # stopx
                        MVector{4,Float64}(0.0,0.0,0.0,0.0), # cstartx
                        MVector{4,Float64}(0.0,0.0,0.0,0.0), # cstopx
                        1.0, 100.0) # rmin_geo, rmax_geo
end


function read_header(filename::String)
    println("Initializing grid from: $filename")
    
    params = GlobalParams()
    cstopx_2 = 0.0

    h5open(filename, "r") do file
        header = file["header"]
        if haskey(header, "has_electrons")
            params.ELECTRONS = read(header, "has_electrons")
        else
            params.ELECTRONS = 0
        end

        if haskey(header, "has_radiation")
            params.RADIATION = read(header, "has_radiation")
        else
            params.RADIATION = 0
        end

        if haskey(header, "has_derefine_poles")
            error("Dump includes flag 'has_derefine_poles' and is therefore non-standard and not well-defined")
        end

        if haskey(header, "weights")
            weights = header["weights"]
            params.mu_i = read(weights, "mu_i")
            params.mu_e = read(weights, "mu_e")
            params.mu_tot = read(weights, "mu_tot")
            @printf(stderr, "Loaded molecular weights (mu_i, mu_e, mu_tot): %g %g %g\n", params.mu_i, params.mu_e, params.mu_tot)
            params.Ne_factor = 1.0 / params.mu_e
            params.ELECTRONS = ELECTRONS_TFLUID
        end

        metric_name_str = read(header, "metric")
        if isa(metric_name_str, Vector{String}) || isa(metric_name_str, Vector{UInt8})
             # Sometimes HDF5 strings come as arrays, handle if needed. 
             # Usually standard read works. Assuming String here.
        end
        
        use_eKS_internal = 0

        if startswith(metric_name_str, "MKS")
            params.metric = METRIC_MKS
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "BHAC_MKS")
            params.metric = METRIC_BHACMKS
            cstopx_2 = π
        elseif startswith(metric_name_str, "MMKS") || startswith(metric_name_str, "FMKS")
            params.metric = METRIC_FMKS
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "MKS3")
            use_eKS_internal = 1
            params.metric = METRIC_MKS3
            cstopx_2 = 1.0
        elseif startswith(metric_name_str, "EKS")
            params.metric = METRIC_EKS
            cstopx_2 = π
        else
            error("File is in unknown metric $metric_name_str. Cannot continue.")
        end

        # 5. Grid Dimensions & Gamma
        params.N1 = read(header, "n1")
        params.N2 = read(header, "n2")
        params.N3 = read(header, "n3")
        params.gam = read(header, "gam")

        if haskey(header, "gam_e")
            @printf(stderr, "custom electron model loaded from dump file...\n")
            params.game = read(header, "gam_e")
            params.gamp = read(header, "gam_p")
        end
        params.Te_unit = params.Thetae_unit
        
        if !USE_FIXED_TPTE && !USE_MIXED_TPTE
            if params.ELECTRONS != 1
                error("! no electron temperature model specified! Cannot continue")
            end
            params.ELECTRONS = 1
            params.Thetae_unit = MP/ME
        elseif params.ELECTRONS == ELECTRONS_TFLUID
            @printf(stderr, "Using Ressler/Athena electrons with mixed tp_over_te and\n")
            @printf(stderr, "trat_small = %g, trat_large = %g, and beta_crit = %g\n", trat_small, trat_large, beta_crit)
        elseif USE_FIXED_TPTE && !USE_MIXED_TPTE
            params.ELECTRONS = 0
            @printf(stderr, "Using fixed tp_over_te ratio = %g\n", tp_over_te)
            params.Thetae_unit = 2.0/3.0 * MP/ME / (2.0 + tp_over_te)
        elseif USE_MIXED_TPTE && !USE_FIXED_TPTE
            params.ELECTRONS = 2
            @printf(stderr, "Using mixed tp_over_te with trat_small = %g, trat_large = %g, and beta_crit = %g\n", 
              trat_small, trat_large, beta_crit)
        else
            error("Unknown electron model $(params.ELECTRONS)! Cannot continue.")
        end
        
        params.Te_unit = params.Thetae_unit

        # 7. Radiation Units
        if params.RADIATION == 1
            @printf(stderr, "custom radiation field tracking information loaded...\n")
            units = header["units"]
            params.M_unit = read(units, "M_unit")
            params.T_unit = read(units, "T_unit")
            params.L_unit = read(units, "L_unit")
            params.Thetae_unit = read(units, "Thetae_unit")
            params.MBH = read(units, "Mbh")
            params.tp_over_te = read(units, "tp_over_te")
        end

        # 8. Geometry Details
        geom = header["geom"]
        params.startx[2] = read(geom, "startx1")
        params.startx[3] = read(geom, "startx2")
        params.startx[4] = read(geom, "startx3")
        params.dx[2] = read(geom, "dx1")
        params.dx[3] = read(geom, "dx2")
        params.dx[4] = read(geom, "dx3")

        # Metric specific sub-groups
        local mks_group
        if params.metric == METRIC_MKS
            mks_group = geom["mks"]
            @printf(stderr, "Using Modified Kerr-Schild coordinates MKS\n")
        elseif params.metric == METRIC_BHACMKS
            mks_group = geom["bhac_mks"]
            @printf(stderr, "Using BHAC-style Modified Kerr-Schild coordinates BHAC_MKS\n")
        elseif params.metric == METRIC_FMKS
            mks_group = geom["mmks"]
            @printf(stderr, "Using Funky Modified Kerr-Schild coordinates FMKS\n")
        elseif params.metric == METRIC_MKS3
            mks_group = geom["mks3"]
            @printf(stderr, "Using logarithmic KS coordinates internally\n")
        elseif params.metric == METRIC_EKS
            mks_group = geom["eks"]
            @printf(stderr, "Using Kerr-Schild coordinates with exponential radial coordinate\n")
        end


        # Read Metric Parameters
        if params.metric == METRIC_MKS3
            params.a = read(mks_group, "a")
            params.mks3R0 = read(mks_group, "R0")
            params.mks3H0 = read(mks_group, "H0")
            params.mks3MY1 = read(mks_group, "MY1")
            params.mks3MY2 = read(mks_group, "MY2")
            params.mks3MP0 = read(mks_group, "MP0")
            params.Rout = 100.0
        elseif params.metric == METRIC_EKS
            params.a = read(mks_group, "a")
            params.Rin = read(mks_group, "r_in")
            params.Rout = read(mks_group, "r_out")
            @printf(stderr, "eKS parameters a: %f Rin: %f Rout: %f\n", params.a, params.Rin, params.Rout)
        else
            # Common MKS-like parameters
            params.a = read(mks_group, "a")
            params.hslope = read(mks_group, "hslope")
            
            # Handle Rin/Rout case sensitivity or existence
            if haskey(mks_group, "Rin")
                params.Rin = read(mks_group, "Rin")
                params.Rout = read(mks_group, "Rout")
            else
                params.Rin = read(mks_group, "r_in")
                params.Rout = read(mks_group, "r_out")
            end
            @printf(stderr, "MKS parameters a: %f hslope: %f Rin: %f Rout: %f\n", params.a, params.hslope, params.Rin, params.Rout)

            if params.metric == METRIC_FMKS
                params.poly_xt = read(mks_group, "poly_xt")
                params.poly_alpha = read(mks_group, "poly_alpha")
                params.mks_smooth = read(mks_group, "mks_smooth")
                
                # Math Translation: 0.5*M_PI*1./(1. + 1./(poly_alpha + 1.)*1./pow(poly_xt, poly_alpha));
                params.poly_norm = 0.5 * π * 1.0 / (1.0 + 1.0 / (params.poly_alpha + 1.0) * 1.0 / (params.poly_xt^params.poly_alpha))
                
                @printf(stderr, "FMKS parameters poly_xt: %f poly_alpha: %f mks_smooth: %f poly_norm: %f\n", 
                    params.poly_xt, params.poly_alpha, params.mks_smooth, params.poly_norm)
            end
        end

    end # HDF5 file closes here

    # 9. Final Grid Calculations
    params.rmax_geo = min(params.rmax_geo, params.Rout)
    params.rmin_geo = max(params.rmin_geo, params.Rin)

    params.stopx = MVector{4, Float64}(
        1.0,
        params.startx[2] + params.N1 * params.dx[2],
        params.startx[3] + params.N2 * params.dx[3],
        params.startx[4] + params.N3 * params.dx[4]
    )

    params.cstartx = MVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
    params.cstopx  = MVector{4, Float64}(0.0, 0.0, cstopx_2, 2*π) 

    # Special logic for MKS/logarithms if needed
    if params.metric != METRIC_BHACMKS && params.metric != METRIC_EKS
        params.cstopx[2] = log(params.Rout)
    end
    
    @printf(stderr, "Grid start (startx): %.15e, %.15e, %.15e stop (stopx): %.15e, %.15e, %.15e\n",
        params.startx[2], params.startx[3], params.startx[4], params.stopx[2], params.stopx[3], params.stopx[4])
    @printf(stderr, "grid dx: %.15e, %.15e, %.15e\n", params.dx[2], params.dx[3], params.dx[4])
    # Construct and return the Grid struct and the Params

    return params
end

function _read_single_primitive(file_handle, prim_name::String)
    if "prims" in keys(file_handle)
        prims = read(file_handle["prims"])
        prim_idx = findfirst(isequal(uppercase(prim_name)), VALID_PRIMS)
        # Returns 'nothing' if the primitive is not found
        return prim_idx === nothing ? nothing : permutedims(prims[prim_idx, :, :, :], (3, 2, 1))
    else
        error("Dataset 'prims' not found in the HDF5 file.")
    end
end


function load_data(filename::String, trat_large::Float64)
    println("Loading data from '$filename' into 'iharm' module...")
    !isfile(filename) && error("File not found: $filename")

    # Temporary variables to store primitives
    rho = uu = u1 = u2 = u3 = b1 = b2 = b3 = nothing

    #this is stupid, change this afterwards
    data_array = Vector{IharmData}(undef, 1)

    
    #Rescale mdot
    rescale_factor = 1.;
    target_mdot = 0; #TODO: THIS HAS TO BE READ FROM INPUT OR PLACED SOMEWHERE ELSE
    if(target_mdot > 0)
        println("Resetting M_unit to match target_mdot = $target_mdot")
        current_mdot = Mdot_dump/MdotEdd_dump
        println("... is now $(M_unit * abs(target_mdot / current_mdot))")
        rescale_factor = abs(target_mdot / current_mdot)
        M_unit *= rescale_factor
    end


    h5open(filename, "r") do file
        t = read(file, "t")
        Threads.@threads for prim_name in VALID_PRIMS
            data_3d = _read_single_primitive(file, prim_name)
            if data_3d !== nothing
                data_3d = Float64.(data_3d)  # ensure Float64

                # Assign to the correct struct field
                if prim_name == "RHO"
                    rho = data_3d
                    if(size(rho,1) != params.N1 || size(rho,2) != params.N2 || size(rho,3) != params.N3)
                        println("N1 = $(size(rho,1)), N2 = $(size(rho,2)), N3 = $(size(rho,3))")
                        error("Data dimensions do not match expected grid size N1,N2,N3")
                    end
                elseif prim_name == "UU"
                    uu = data_3d
                elseif prim_name == "U1"
                    u1 = data_3d
                elseif prim_name == "U2"
                    u2 = data_3d
                elseif prim_name == "U3"
                    u3 = data_3d
                elseif prim_name == "B1"
                    b1 = data_3d
                elseif prim_name == "B2"
                    b2 = data_3d
                elseif prim_name == "B3"
                    b3 = data_3d
                end
            end
        end
        data_array[1] = IharmData(t, rho, uu, u1, u2, u3, b1, b2, b3, zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)), zeros(size(rho)))            
    end

    #println("Primitives successfully loaded. Dimensions: ", size(rho))
    #println("Calculating physical quantities...")
    Threads.@threads for i in 1:(params.N1)
        for j in 1:(params.N2)
            X::MVec4 = zeros(MVec4)
            ijktoX(i-1, j-1, 0, X)
            gcov::MMat4 = zeros(MMat4)
            gcon::MMat4 = zeros(MMat4)
            gcov_func!(X, params.a, gcov)
            gcon_func!(gcov, gcon)
            g = gdet_zone(i-1, j-1, 0)

            for k in 1:(params.N3)
                ijktoX(i-1, j-1, k, X)

                Ufields = (data_array[1].U1, data_array[1].U2, data_array[1].U3)
                UdotU = 0.0
                for l in 1:(NDIM -1)       # l = 1,2,3 corresponds to U1,U2,U3
                    for m in 1:(NDIM -1)
                        UdotU += gcov[l+1, m+1] * Ufields[l][i,j,k] * Ufields[m][i,j,k]
                    end
                end

                ufac = sqrt(-1. / gcon[1,1] * (1. + abs(UdotU)))
                ucon::MVec4 = MVec4(undef)
                ucon[1] = -ufac * gcon[1,1]

                for μ in 1:(NDIM-1)
                    ucon[μ + 1] = Ufields[μ][i,j,k] - ufac * gcon[1, μ+1]
                end
                
                ucov::MVec4 = MVec4(undef)
                flip_index!(ucov, ucon, gcov)
                udotB = 0.0
                Bfields = (data_array[1].B1, data_array[1].B2, data_array[1].B3)
                for l in 1:(NDIM -1)
                    udotB += ucov[l+1] * Bfields[l][i,j,k]
                end

                bcon::MVec4 = MVec4(undef)
                bcon[1] = udotB
                for μ in 1:(NDIM-1)
                    bcon[μ+1] = (Bfields[μ][i,j,k] + ucon[μ+1] * udotB) / ucon[1]
                end
                bcov = MVec4(undef)
                flip_index!(bcov, bcon, gcov)

                bsq = 0.0
                for l in 1:NDIM
                    bsq += bcov[l] * bcon[l]
                end
                data_array[1].b[i,j,k] = sqrt(bsq) * B_unit  # Magnetic field strength

            end
        end
    end
    init_physical_quantities(data_array, 1, rescale_factor, trat_large)  # Initialize physical quantities for the first dataset

    # Check if all fields were loaded
    fields = [rho, uu, u1, u2, u3, b1, b2, b3]
    if any(x -> x === nothing, fields)
        @warn "Some primitives were missing in file '$filename'."
    else
        println("All primitives successfully loaded. Dimensions: ", size(rho))
    end

    #update the dump path for slow light if needed
    if(SLOW_LIGHT)
        params_slowlight.current_dumps_path = update_dump_path()
    end
    return data_array[1]
end


function init_physical_quantities(data, n::Int64, rescale_factor::Float64, trat_large::Float64)
    #println("Using mixed tp_over_te with trat_small = $(trat_small), trat_large = $(trat_large), and beta_crit = $(beta_crit)")
    
    # Pre-compute constants
    rescale_factor_sqrt = sqrt(rescale_factor)
    rho_factor = RHO_unit / (MP + ME) * params.Ne_factor
    gam_minus_1 = params.gam - 1.0
    beta_crit_sq = beta_crit * beta_crit
    θe_factor = (MP / ME) * (params.game - 1.0) * (params.gamp - 1.0)
    game_minus_1 = params.game - 1.0
    gamp_minus_1 = params.gamp - 1.0
    B_unit_inv = 1.0 / B_unit
    
    # Get array references once (helps with type stability)
    ne_arr = data[n].ne
    b_arr = data[n].b
    θe_arr = data[n].θe
    sigma_arr = data[n].sigma
    beta_arr = data[n].beta
    RHO_arr = data[n].RHO
    UU_arr = data[n].UU
    dθedRhi = data[n].dθedRhi

    
    @inbounds Threads.@threads for i in 1:params.N1
        for j in 1:params.N2
            for k in 1:params.N3
                rho_ijk = RHO_arr[i, j, k]
                uu_ijk = UU_arr[i, j, k]
                b_ijk = b_arr[i, j, k]
                
                ne_arr[i, j, k] = rho_ijk * rho_factor
                
                b_ijk *= rescale_factor_sqrt
                b_arr[i, j, k] = b_ijk
                
                bsq_normalized = b_ijk * B_unit_inv
                bsq = bsq_normalized * bsq_normalized
                
                sigma_m = bsq / rho_ijk
                beta_m = uu_ijk * gam_minus_1 / (0.5 * bsq)
                
                betasq = beta_m * beta_m / beta_crit_sq
                betasq_plus_1_inv = 1.0 / (1.0 + betasq)
                trat = trat_large * betasq * betasq_plus_1_inv + trat_small * betasq_plus_1_inv                
                θe_unit = θe_factor / (game_minus_1 * trat + gamp_minus_1)
                θe_val = θe_unit * uu_ijk / rho_ijk

                dtratdRhi = betasq * betasq_plus_1_inv
                dθe_unit_dRhi = - θe_factor * game_minus_1/((game_minus_1 * trat + gamp_minus_1)^2) * dtratdRhi
                dθe_dRhi = dθe_unit_dRhi * uu_ijk / rho_ijk 
                
                if θe_val > 1.0e-3
                    θe_arr[i, j, k] = θe_val
                    dθedRhi[i, j, k] = dθe_dRhi
                else
                    θe_arr[i, j, k] = 1.0e-3
                    dθedRhi[i, j, k] = 0.0
                end

                sigma_arr[i, j, k] = sigma_m > SMALL ? sigma_m : SMALL
                beta_arr[i, j, k] = beta_m > SMALL ? beta_m : SMALL
            end
        end
    end
end

function get_model_sigma(X, data)
    if (X_in_domain(X) == 0)
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, data)

    return interp_scalar_time(X, data[nA].sigma, data[nB].sigma, tfac)
end


function get_sigma_smoothfac(sigma)
    sigma_above = sigma_cut
    if(sigma_cut_high > 0.0)
        sigma_above = sigma_cut_high
    end
    if(sigma < sigma_cut)
        return 1.0
    end
    if(sigma > sigma_above)
        return 0.0
    end
    dsig = sigma_above - sigma_cut
    return cos(π/2. /dsig * (sigma - sigma_cut))
end



function get_model_ne(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    sigma_smoothfac = 1.0;

    if(USE_GEODESIC_SIGMACUT)
        sigma = get_model_sigma(X, data);
        if(sigma > sigma_cut)
            return 0.0
        end
        sigma_smoothfac = get_sigma_smoothfac(sigma)
    end
    nA, nB, tfac = set_tinterp_ns(X, data)
    return interp_scalar_time(X, data[nA].ne, data[nB].ne, tfac) * sigma_smoothfac
end

function set_tinterp_ns(X, data)
    if (SLOW_LIGHT)
        nA = 0
        nB = 0
        if(X[1] < data[2].t)
            nA = 1
            nB = 2
        else
            nA = 2
            nB = 3
        end
        tinterp = 1. - (X[1] - data[nA].t)/(data[nB].t - data[nA].t)
        return nA, nB, tinterp
    else
        return 1, 1, 0.0
    end
end


function get_model_thetae(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, data)
    return interp_scalar_time(X, data[nA].θe, data[nB].θe, tfac)
end

function get_model_thetae_deriv(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, data)
    return interp_scalar_time(X, data[nA].dθedRhi, data[nB].dθedRhi, tfac)
end

function get_model_b(X, data)
    if(X_in_domain(X) == 0)
        return 0.0
    end
    nA, nB, tfac = set_tinterp_ns(X, data)
    return interp_scalar_time(X, data[nA].b, data[nB].b, tfac)
end

@inline function get_model_fourv(data, X, Kcon, Ucon, Ucov, Bcon, Bcov, bhspin)
    gcov = gcov_func(X, bhspin)
    gcon = gcon_func(gcov)

    if(X_in_domain(X) == 0)
        Ucov[1] = -1. /sqrt(-gcon[1, 1])
        Ucov[2] = 0.0
        Ucov[3] = 0.0
        Ucov[4] = 0.0
        Ucon[1] = 0.0
        Ucon[2] = 0.0
        Ucon[3] = 0.0
        Ucon[4] = 0.0

        for μ in 1:NDIM
            Ucon[1] += Ucov[μ] * gcon[1, μ]
            Ucon[2] += Ucov[μ] * gcon[2, μ]
            Ucon[3] += Ucov[μ] * gcon[3, μ]
            Ucon[4] += Ucov[μ] * gcon[4, μ]
            Bcon[μ] = 0.0
            Bcov[μ] = 0.0
        end
        return
    end
    Vcon = similar(X)
    nA, nB, tfac = set_tinterp_ns(X, data)
    Vcon[2] = interp_scalar_time(X, data[nA].U1, data[nB].U1, tfac);
    Vcon[3] = interp_scalar_time(X, data[nA].U2, data[nB].U2, tfac);
    Vcon[4] = interp_scalar_time(X, data[nA].U3, data[nB].U3, tfac);

    VdotV = 0.0
    for μ in 2:NDIM
        for ν in 2:NDIM
            VdotV += gcov[μ, ν] * Vcon[μ] * Vcon[ν]
        end
    end

    Vfac = sqrt(-1. /gcon[1, 1] * (1. + abs(VdotV)))
    Ucon[1] = -Vfac * gcon[1, 1]
    for μ in 2:NDIM
        Ucon[μ] =  Vcon[μ] - Vfac * gcon[1, μ]
    end

    Ucov_local = flip_index(Ucon, gcov)

    #Now set Bcon and get Bcov by lowering it

    Bcon1 = interp_scalar_time(X, data[nA].B1, data[nB].B1, tfac);
    Bcon2 = interp_scalar_time(X, data[nA].B2, data[nB].B2, tfac);
    Bcon3 = interp_scalar_time(X, data[nA].B3, data[nB].B3, tfac);

    Bcon[1] = (Ucov_local[2] * Bcon1 + Ucov_local[3] * Bcon2 + Ucov_local[4] * Bcon3)
    Bcon[2] = (Bcon1 + Ucon[2] * Bcon[1]) / Ucon[1]
    Bcon[3] = (Bcon2 + Ucon[3] * Bcon[1]) / Ucon[1]
    Bcon[4] = (Bcon3 + Ucon[4] * Bcon[1]) / Ucon[1]

    Bcov_local = flip_index(Bcon, gcov)

    for μ in 1:NDIM
        Ucov[μ] = Ucov_local[μ]
        Bcov[μ] = Bcov_local[μ]
    end
end



function radiating_region(X, Rh::Float64)
    """
    Checks if the position is within the radiating region.
    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, th = bl_coord(X)
    return (r > (params.rmin_geo) && r < params.rmax_geo && th > th_beg && th < (π - th_beg))
end
