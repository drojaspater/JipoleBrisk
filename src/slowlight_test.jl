
using DelimitedFiles

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
    ImageCadence::Int64
    tA::Float64
    tB::Float64
    tf::Float64
    current_dumps_path::String
end

using Printf

#####################
const dump_paths = readlines("dump_list.txt")

function update_dump_path()
    dump_idx = params_slowlight.nloaded
    params_slowlight.nloaded += 1
    return dump_paths[dump_idx]
end
#####################3
#function update_dump_path()
#    dump_idx = params_slowlight.nloaded
#    params_slowlight.nloaded += 1 #Originalmente +1, estoy saltando de a dos snapshots 
#
#    return Printf.format(Printf.Format(all_dumps_path), dump_idx)
#end

function get_specific_dump_time(dump_idx::Int64)
    dump_path = Printf.format(Printf.Format(all_dumps_path), dump_idx)
    t::Float64 = 0.0
    h5open(dump_path, "r") do file
        t = read(file, "t")
    end
    return t
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
    
    # Avanza al siguiente dump de la lista antes de cargar
    params_slowlight.current_dumps_path = update_dump_path()
    simulation_data[3] = load_data(params_slowlight.current_dumps_path, trat_large)

    #Update the global slowlight time parameters
    params_slowlight.tA = simulation_data[1].t
    params_slowlight.tB = simulation_data[2].t

    @info "Loaded data" dump_path=params_slowlight.current_dumps_path tA=params_slowlight.tA tB=params_slowlight.tB
end



function process_slowlight_images!(
    params_slowlight, simulation_data, all_geodesics, nsteps, 
    params, t0, tgeof, tgeoi, pixels_x, pixels_y, freq
)
    
    last_img_target = params_slowlight.tA - tgeof
    nimgs_concurrently = round(Int, 2 + abs(t0) / params_slowlight.ImageCadence)
    
    MovieArray = [zero(OfImg) for _ in 1:pixels_x, _ in 1:pixels_y, _ in 1:nimgs_concurrently]
    target_times = zeros(Float64, nimgs_concurrently)
    valid_images = zeros(Float64, nimgs_concurrently)
    
    println("First Image will be produced at $last_img_target")
    nimg = 1
    nopenimgs = 1
    output = "Image.%05d.txt"

    while true
        while (last_img_target + t0 < params_slowlight.tB)
            target_times[nimg] = last_img_target
            if (last_img_target + tgeoi < params_slowlight.tf - params.rmax_geo)
                valid_images[nimg] = 1
                nopenimgs += 1
                for i in 1:pixels_x
                    for j in 1:pixels_y
                        MovieArray[i, j, nimg].nstep = nsteps[i, j]
                        MovieArray[i, j, nimg].Intensity = 0.0
                        MovieArray[i, j, nimg].tau = 0.0
                        MovieArray[i, j, nimg].tauF = 0.0
                    end
                end
                nimg += 1
                if nimg > nimgs_concurrently
                    nimg = 1
                end
            end
            last_img_target += params_slowlight.ImageCadence
        end

        for k in 1:nimgs_concurrently
            if valid_images[k] == 0
                continue
            end
            do_output = true

            p = Progress(
                pixels_x * pixels_y; 
                desc = "Rendering frame slice $k... out of $nimgs_concurrently", 
                showspeed = true, 
                barlen = 30
            )

            Threads.@threads for i in 1:pixels_x
                for j in 1:pixels_y
                    Xi = MVec4(undef)
                    Kconi = MVec4(undef)
                    Xf = MVec4(undef)
                    Kconf = MVec4(undef)
                    Xhalf = MVec4(undef)
                    Kconhalf = MVec4(undef)
                    traj = all_geodesics[i, j]
                    nstep = copy(MovieArray[i,j,k].nstep)
                    
                    while (nstep > 2)
                        for a in 1:NDIM
                            Xi[a] = traj[nstep].X[a]
                            Xhalf[a] = traj[nstep].Xhalf[a]
                            Xf[a] = traj[nstep - 1].X[a]
                            Kconi[a] = traj[nstep].Kcon[a]
                            Kconhalf[a] = traj[nstep].Kconhalf[a]
                            Kconf[a] = traj[nstep - 1].Kcon[a]
                        end
                        Xi[1] += target_times[k] + 1e-5
                        Xhalf[1] += target_times[k] + 1.e-5
                        Xf[1] += target_times[k] + 1.e-5

                        if (Xi[1] < params_slowlight.tA)
                            Xf[1] += params_slowlight.tA - Xi[1]
                            Xhalf[1] += params_slowlight.tA - Xi[1]
                            Xi[1] = params_slowlight.tA
                        end
                        if (Xi[1] >= params_slowlight.tB)
                            if (Xf[1] >= params_slowlight.tf)
                                Xi[1] += params_slowlight.tf - Xf[1]
                                Xhalf[1] += params_slowlight.tf - Xf[1]
                                Xf[1] = params_slowlight.tf
                            else
                                break
                            end
                        end
                        
                        ji, ki = get_jk(Xi, Kconi, freq, params.a, simulation_data)
                        jf, kf = get_jk(Xf, Kconf, freq, params.a, simulation_data)

                        MovieArray[i,j,k].Intensity = approximate_solve(MovieArray[i,j,k].Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)
                        
                        nstep -= 1
                    end
                    MovieArray[i,j,k].nstep = copy(nstep)
                    if (nstep != 2)
                        do_output = false
                    end
                    ProgressMeter.next!(p)
                end
            end
            finish!(p)
            
            if (do_output)
                Image_out = map(x -> x.Intensity, MovieArray[:,:,k]) .* freq^3
                
                file_name = Printf.format(Printf.Format(output), target_times[k])
                writedlm(file_name, Image_out)
                println("Saving image $(file_name)")
                
                valid_images[k] = 0
                nopenimgs -= 1
            end 
        end
        
        if (nopenimgs <= 1)
            break
        end
        update_data!(simulation_data)
    end
end
