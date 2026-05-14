const MODEL = "iharm"
const MBH = 6.2e9
const SLOW_LIGHT = true
include("../src/main.jl");

#path to folder with the arxiv name properly indexed
#dump_filepath = "/home/pedro/kharma/iharm3d_out/tmp.00028.h5"
#dump_filepath = "/home/pedro/sample_dump_SANE_a+0.94_MKS_0900.h5"
const all_dumps_path = "/home/pedro/kharma/iharm3d_out/tmp.%05d.h5"

const dump_max = 10
const ImageCadence = 10
params_slowlight = OfSlowLight(dump_max, ImageCadence, 0, 0.0, 0.0, 0.0, "")
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