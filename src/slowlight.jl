
const NFILES = 3
#for slow_light
mutable struct OfImg
    nstep::Int
    Intensity::Float64
    tau::Float64
    tauF::Float64
    N_coord::MMatrix{4, 4, ComplexF64}
end

#for slow_light
function Base.zero(::Type{OfImg})
    OfImg(
        0,
        0.0,
        0.0,
        0.0,
        MMatrix{4,4,ComplexF64}(zeros(ComplexF64,4,4))
    )
end

mutable struct OfSlowLight
    dump_max::Int64
    nloaded::Int64
    tA::Float64
    tB::Float64
    tf::Float64
    current_dumps_path::String
end

using Printf
function update_dump_path()
    dump_idx = params_slowlight.nloaded
    params_slowlight.nloaded += 1

    return Printf.format(Printf.Format(all_dumps_path), dump_idx)
end

function get_specific_dump_time(dump_idx::Int64)
    dump_path = Printf.format(Printf.Format(all_dumps_path), dump_idx)
    t::Float64 = 0.0
    h5open(dump_path, "r") do file
        t = read(file, "t")
    end
    return t
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

function update_data!(simulation_data::Vector{IharmData})
    # Save the reference to the oldest data object before it gets overwritten.
    # We do this so we can reuse its allocated memory for the new dump!
    oldest_data = simulation_data[1]

    #Shift the timeline down (what was middle is now oldest, newest is now middle)
    simulation_data[1] = simulation_data[2]
    simulation_data[2] = simulation_data[3]

    #Move the old memory block to the "newest" slot so it can be overwritten
    simulation_data[3] = oldest_data
    
    simulation_data[3] = load_data(params_slowlight.current_dumps_path, trat_large)

    #Update the global slowlight time parameters
    params_slowlight.tA = simulation_data[1].t
    params_slowlight.tB = simulation_data[2].t

    @info "Loaded data" dump_path=params_slowlight.current_dumps_path tA=params_slowlight.tA tB=params_slowlight.tB
end