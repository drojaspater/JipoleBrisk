using DelimitedFiles

const MODEL = "iharm"
const MBH = 6.2e9
const SLOW_LIGHT = true
include("../src/main.jl");

#path to folder with the arxiv name properly indexed
#dump_filepath = "/home/pedro/kharma/iharm3d_out/tmp.00028.h5"
#dump_filepath = "/home/pedro/sample_dump_SANE_a+0.94_MKS_0900.h5"
const all_dumps_path = "/home/pedro/kharma/iharm3d_out/tmp.%05d.h5"

const dump_max = 10

params_slowlight = OfSlowLight(dump_max, 0, 0.0, 0.0, 0.0, "")
params_slowlight.current_dumps_path = update_dump_path()

trat_large = 20. 
const trat_small = 1. 
const beta_crit = 1.0 
const th_beg = 1.74e-2 
const sigma_cut = 1.0 
const sigma_cut_high = -1.0;

const params = read_header(params_slowlight.current_dumps_path);
const simulation_data = Vector{IharmData}(undef, 3)

# everytime you load a file slow_light mode will automatically advance to the next one
simulation_data[1] = load_data(params_slowlight.current_dumps_path, trat_large)
simulation_data[2] = load_data(params_slowlight.current_dumps_path, trat_large)
simulation_data[3] = load_data(params_slowlight.current_dumps_path, trat_large)

params_slowlight.tA = simulation_data[1].t;
params_slowlight.tB = simulation_data[2].t;

params_slowlight.tf = get_specific_dump_time(params_slowlight.dump_max);

# Observer distance in gravitational radii (Rg)
const ro = 1000.0

# Inclination angle (deg) — angle between the observer and the BH spin axis
const th = 163.0

# Azimuthal angle (deg) — rotation around the system
const phi = 0.0

# Image resolution — total geodesics traced = res^2
const res = 128
const pixels_x = 128
const pixels_y = 128

# Distance to the source (in parsecs, converted to code units)
const SourceD = 16.9e6 * PC

# Radius where ray integration stops
const Rstop = 100.0

# Event horizon radius for a Kerr black hole
const Rh = 1 + sqrt(1. - params.a * params.a)

# Observing frequency (Hz), e.g. 230 GHz for EHT-like images
const freq = 230e9

# Image plane size (in Rg), scaled from physical distance
const DXsize = SourceD / L_unit / MUAS_PER_RAD * 160
const DYsize = SourceD / L_unit / MUAS_PER_RAD * 160

# Field of view (radians)
const fovx = DXsize / ro
const fovy = DYsize / ro

# Image offsets (can be used to shift the camera)
const xoff = 0.0
const yoff = 0.0

# Calculate the camera position in native coordinates
Xcamera = MVec4(camera_position(ro, th, phi, params.a, params.Rout))

# Unitless frequency 
const freq_unitless = freq * HPL / (ME * CL * CL)

# Array that will hold the Intensity value for each pixel
Image = zeros(Float64, pixels_x, pixels_y)
midplane_crossings = zeros(Int, pixels_x, pixels_y)
nsteps = zeros(Int, pixels_x, pixels_y)

# Number of threads used in the calculation
const nthreads = Threads.nthreads() + 1

println("Allocating workspaces for $nthreads threads...")
# Number of maximum steps in the geodesic calculation
const maxnstep = 15000

# Allocating the scratchpad vector for each thread
#thread_trajs = [Vector{OfTrajM}(undef, maxnstep) for _ in 1:nthreads]
#for t in 1:nthreads
#    for k in 1:maxnstep
#        thread_trajs[t][k] = OfTrajM(
#            0.0, 
#            MVec4(undef), MVec4(undef), MVec4(undef), MVec4(undef)
#        )
#    end
#end

# Update thread allocation to use the new immutable OfTrajS
dummy_svec = @SVector zeros(4)
dummy_traj = OfTrajS(0.0, dummy_svec, dummy_svec, dummy_svec, dummy_svec)

thread_trajs = [Vector{OfTrajS}(undef, maxnstep) for _ in 1:nthreads]
for t in 1:nthreads
    for k in 1:maxnstep
        thread_trajs[t][k] = dummy_traj
    end
end

# This will hold the exact number of steps for each pixel.
const all_geodesics = Matrix{Vector{OfTrajS}}(undef, pixels_x, pixels_y)

# Allocate an array to hold the minimum time found by EACH thread
# Initialized to 0.0 because your photon times will be negative
thread_t0 = zeros(Float64, nthreads)

# tgeoi needs to find the maximum (closest to zero), so initialize very negative
thread_tgeoi = fill(-1e100, nthreads) 

# tgeof needs to find the minimum (most negative), so initialize at zero
thread_tgeof = zeros(Float64, nthreads)

p = Progress(
    pixels_x * pixels_y; 
    desc = "Raytracing Image...", 
    showspeed = true, 
    barlen = 30
)

println("Tracing Geodesics...")
Threads.@threads for i in 0:(pixels_x - 1)
    tid = Threads.threadid() 
    
    for j in 0:(pixels_y - 1)
        nstep, midplane_crossings[i+1,j+1] = get_pixel(
            thread_trajs[tid], i, j, Xcamera, 
            fovx, fovy, freq_unitless, 
            pixels_x, pixels_y, params.a, 
            Rh, params.Rout, Rstop, xoff, yoff
        ) 
        nsteps[i+1, j+1] = nstep
        # Save to permanent storage
        #all_geodesics[i + 1, j + 1] = deepcopy(thread_trajs[tid][1:nstep])

        all_geodesics[i + 1, j + 1] = thread_trajs[tid][1:nstep]

        final_step_time = thread_trajs[tid][nstep].X[1]
        if final_step_time < thread_t0[tid]
            thread_t0[tid] = final_step_time
        end

        # tgeoi and tgeof calculations following how ipole does it
        pixel_tgeoi = 1.0
        pixel_tgeof = 1.0
        for k in 1:nstep
            X = thread_trajs[tid][k].X
            K = thread_trajs[tid][k].Kcon
        
            log_r = X[2]          # must be log(r)
            t_coord = X[1]
            k_r = K[2]            # must match Kcon[1]
        
            if pixel_tgeoi > 0.0 && log_r < log(100.0)
                pixel_tgeoi = t_coord
            end
        
            if pixel_tgeof > 0.0 && log_r > log(100.0) && k_r < 0.0
                pixel_tgeof = t_coord
            end
        end
        final_step_time = thread_trajs[tid][nstep].X[1]
        
        if pixel_tgeoi < 0.0 && pixel_tgeoi > thread_tgeoi[tid]
            thread_tgeoi[tid] = pixel_tgeoi
        end
        if pixel_tgeof < 0.0 && pixel_tgeof < thread_tgeof[tid]
            thread_tgeof[tid] = pixel_tgeof
        elseif pixel_tgeof > 0.0 && final_step_time < thread_tgeof[tid]
            thread_tgeof[tid] = final_step_time
        end
        
        ProgressMeter.next!(
                p; 
                showvalues = [
                    (:thread_id, tid), 
                    (:pixel, "($i, $j)"), 
                    (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")
            ]
        )
    end
end

Image *= freq^3
finish!(p);

t0 = minimum(thread_t0)
tgeof = minimum(thread_tgeof) # The most negative time in the active zone (Oldest file needed)
tgeoi = maximum(thread_tgeoi) # The least negative time in the active zone (Newest file needed)

println("Calculated t0 (absolute longest time): $t0")
println("Calculated tgeof (oldest active time): $tgeof")
println("Calculated tgeoi (newest active time): $tgeoi")

# Eliminate arrays from RAM
thread_to = nothing
thread_tgeoi = nothing
thread_tgeof = nothing
thread_trajs = nothing
GC.gc()

function process_slowlight_images!(
    params_slowlight, simulation_data, all_geodesics, nsteps, 
    params, t0, tgeof, tgeoi, pixels_x, pixels_y, freq
)
    # 1. Initialize local variables
    ImageCadence = 10 
    last_img_target = params_slowlight.tA - tgeof
    nimgs_concurrently = round(Int, 2 + abs(t0) / ImageCadence)
    
    MovieArray = [zero(OfImg) for _ in 1:pixels_x, _ in 1:pixels_y, _ in 1:nimgs_concurrently]
    target_times = zeros(Float64, nimgs_concurrently)
    valid_images = zeros(Float64, nimgs_concurrently)
    
    println("First Image will be produced at $last_img_target")
    nimg = 1
    nopenimgs = 1
    output = "Image.%05d.txt"

    # 2. The main processing loop
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
            last_img_target += ImageCadence
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
                
                if (k == 1)
                    println("0,0 $(Image_out[1,1])")
                    println("0,1 $(Image_out[1,2])")
                    println("1,0 $(Image_out[2,1])")
                    println("1,1 $(Image_out[2,2])")
                end
                
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

println("Starting image processing loop...")
process_slowlight_images!(
    params_slowlight, 
    simulation_data, 
    all_geodesics, 
    nsteps, 
    params, 
    t0, 
    tgeof, 
    tgeoi, 
    pixels_x, 
    pixels_y, 
    freq
)
println("Done!")