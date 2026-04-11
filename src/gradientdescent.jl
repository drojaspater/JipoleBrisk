using ImageFiltering
using ProgressMeter



function cost_func(ImageObs, ImageTest)
    """
    Normalized Mean Squared Error (NMSE) between observed and test images.
    """
    if length(ImageObs) != length(ImageTest)
        println("Length of ImageObs: $(length(ImageObs))")
        println("Length of ImageTest: $(length(ImageTest))")
        throw(ArgumentError("ImageObs and ImageTest must have the same length."))
    end

    # numerator: sum of squared differences
    numerator = sum((ImageTest .- ImageObs).^2)

    # denominator: sum of squared reference values
    denominator = sum(ImageObs.^2)

    nmse = numerator / denominator
    return nmse
end

function GradientofCostFunction(ImageObs, ImageTest, dI_dθo, dI_da)
    if size(ImageObs) != size(ImageTest)
        throw(ArgumentError("ImageObs and ImageTest must have the same shape"))
    end

    ΔI = ImageTest .- ImageObs
    denom = sum(ImageObs.^2)

    grad_θo = 2 * sum(ΔI .* dI_dθo) / denom
    grad_a  = 2 * sum(ΔI .* dI_da)  / denom

    return grad_θo, grad_a
end

function FiniteDifferencesθ(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h, bhspin, Rout, Rstop, data = nothing, xoff = 0.0, yoff = 0.0)
    """
    Finite differences method to calculate the intensity at each pixel in the image.
    Uses threaded per-pixel integration to minimize memory overhead.
    """
    
    # --- Pre-calculations ---
    θh = th + h
    θl = th - h

    # Calculate the fov at each dim x and y 
    fovx = DXsize / ro
    fovy = DYsize / ro

    # Unitless frequency for geodesics
    freq_unitless = freq * HPL / (ME * CL * CL) 

    # Calculate the camera positions in native coordinates
    Xcamh = MVec4(camera_position(ro, θh, phi, bhspin, Rout))
    Xcaml = MVec4(camera_position(ro, θl, phi, bhspin, Rout))
    Xcamc = MVec4(camera_position(ro, th, phi, bhspin, Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    # Note: Retained for logging, even if not explicitly passed to integrate_emission! in the new pattern
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
    println("scale_factor = $scale_factor")

    # --- Internal Helper Function (The New Algorithm) ---
    function trace_image(Xcamera, description)
        println("Calculating $description...")
        
        # Initialize Image buffer
        local_img = zeros(Float64, pixels_x, pixels_y)
        
        # Setup Progress Meter
        p = Progress(
            pixels_x * pixels_y; 
            desc = "Raytracing $description...", 
            showspeed = true, 
            barlen = 30
        )
        
        # Raytracing Loop
        Threads.@threads for i in 0:(pixels_x - 1)
            tid = Threads.threadid()
            for j in 0:(pixels_y - 1)
                # Initialize trajectory vector for this pixel
                traj = Vector{OfTraj}()
                sizehint!(traj, maxnstep)
                
                # Calculate Geodesic
                nstep = get_pixel(traj, i, j, Xcamera, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rout, Rstop, xoff, yoff) 
                
                # Resize and Integrate
                resize!(traj, nstep)
                integrate_emission!(traj, nstep, local_img, i + 1, j + 1, freq, bhspin, data)
                
                # Update Progress
                # ProgressMeter.next!(
                #     p; 
                #     showvalues = [
                #         (:thread_id, tid), 
                #         (:pixel, "($i, $j)"), 
                #         (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")
                #     ]
                # )
            end
        end
        finish!(p)
        
        # Apply Scaling
        local_img *= freq^3
        return local_img
    end

    # --- Execution ---
    
    # 1. Calculate High Trajectory/Image
    Imageh = trace_image(Xcamh, "High (+h)")
    
    # 2. Calculate Low Trajectory/Image
    Imagel = trace_image(Xcaml, "Low (-h)")

    # 3. Calculate Finite Difference
    dI_dθo = (Imageh - Imagel) / (2 * h)

    # 4. Calculate Central Image
    Imagec = trace_image(Xcamc, "Central")

    return dI_dθo, Imagec
end


# function FiniteDifferencesTrat(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h_trat, bhspin, Rout, Rstop, dump_filepath)
#     """
#     Finite differences method for trat_large (Rhigh).
#     Optimized to compute geodesic once per pixel, integrate against all temp models, 
#     and discard trajectory immediately to save memory.
#     """

#     println("Initializing Physics Data for Finite Differences...")

#     Trat_h = trat_large + h_trat
#     Trat_l = trat_large - h_trat

#     data_h = load_data(dump_filepath, Trat_h)
#     data_l = load_data(dump_filepath, Trat_l)
#     data_c = load_data(dump_filepath, trat_large)

#     fovx = DXsize / ro
#     fovy = DYsize / ro
#     Xcam = MVec4(camera_position(ro, th, phi, bhspin, Rout))

#     freq_unitless = freq * HPL / (ME * CL * CL) 

#     scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)

#     Imageh = zeros(Float64, pixels_x, pixels_y)
#     Imagel = zeros(Float64, pixels_x, pixels_y)
#     Imagec = zeros(Float64, pixels_x, pixels_y)

#     println("Calculating Trajectories and integrating High/Low/Central simultaneously...")

#     p = Progress(pixels_x * pixels_y; desc = "Computing Pixels...", showspeed = true, barlen = 30)

#     Threads.@threads for i in 0:(pixels_x - 1)
        
#         for j in 0:(pixels_y - 1)
#             traj = Vector{OfTraj}()
#             sizehint!(traj, maxnstep)

#             nstep = get_pixel(traj, i, j, Xcam, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rout, Rstop, 0.0, 0.0)
            
#             resize!(traj, nstep)
            
#             # 1. High Trat
#             integrate_emission!(traj, nstep, Imageh, i + 1, j + 1, freq, bhspin, data_h)
            
#             # 2. Low Trat
#             integrate_emission!(traj, nstep, Imagel, i + 1, j + 1, freq, bhspin, data_l)

#             # 3. Central Trat
#             integrate_emission!(traj, nstep, Imagec, i + 1, j + 1, freq, bhspin, data_c)

#         end
#     end
#     finish!(p)

#     Imageh .*= freq^3
#     Imagel .*= freq^3
#     Imagec .*= freq^3

#     dI_dRhigh = (Imageh - Imagel) / (2 * h_trat)

#     data_h = nothing
#     data_l = nothing
#     data_c = nothing

#     return dI_dRhigh, Imagec
# end

function FiniteDifferencesTrat(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep, h_trat, bhspin, Rout, Rstop, dump_filepath)
    """
    Finite differences method for trat_large.
    Optimized for Limited RAM: Aggressive GC and Buffer Reuse.
    """

    # --- Pre-calculations ---
    fovx = DXsize / ro
    fovy = DYsize / ro
    Xcam = MVec4(camera_position(ro, th, phi, bhspin, Rout))
    freq_unitless = freq * HPL / (ME * CL * CL) 
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)

    # --- Internal Helper Function ---
    function trace_variant(target_trat, description)
        println("\n=== Processing $description ===")
        
        # 1. Load Data
        println("Loading simulation data for Trat = $target_trat...")
        local_data = load_data(dump_filepath, target_trat)
        
        # 2. Initialize Image Buffer
        local_img = zeros(Float64, pixels_x, pixels_y)

        # 4. Raytracing Loop
        p = Progress(pixels_x * pixels_y; desc = "Raytracing $description...", showspeed = true, barlen = 30)
        
        Threads.@threads for i in 0:(pixels_x - 1)
            # Get the pre-allocated vector for this specific thread
            id = Threads.threadid()            
            for j in 0:(pixels_y - 1)
                traj = Vector{OfTraj}()
                sizehint!(traj, maxnstep)
                
                # Calculate Geodesic (fills the reused traj vector)
                nstep = get_pixel(traj, i, j, Xcam, maxnstep, fovx, fovy, freq_unitless, pixels_x, pixels_y, bhspin, Rh, Rout, Rstop, 0.0, 0.0) 
                
                # Resize logic (if get_pixel doesn't set length perfectly)
                # Note: If get_pixel pushes to traj, nstep is likely just length(traj)
                if length(traj) != nstep
                    resize!(traj, nstep)
                end
                
                # Integrate
                integrate_emission!(traj, nstep, local_img, i + 1, j + 1, freq, bhspin, local_data)
                
                # ProgressMeter.next!(p) # Thread-safe in newer versions
            end
        end
        finish!(p)
        
        # 5. Clean up & Force GC
        # This is the most critical step for your kernel issues
        local_data = nothing
        thread_trajs = nothing
        
        println("Forcing Garbage Collection for $description...")
        GC.gc() # Force full garbage collection
        GC.gc() # Run twice to ensure older generations are swept
        
        # Apply scaling
        local_img .*= freq^3
        return local_img
    end

    # --- Execution: Sequential Passes ---

    # 1. Calculate High Image (+h)
    Imageh = trace_variant(trat_large + h_trat, "High (+h)")

    # 2. Calculate Low Image (-h)
    Imagel = trace_variant(trat_large - h_trat, "Low (-h)")

    # 3. Calculate Derivative
    # Create the derivative matrix
    dI_dRhigh = (Imageh .- Imagel) ./ (2 * h_trat)

    # 4. Free High/Low images immediately
    # We don't need them anymore, so free them before the Central pass
    Imageh = nothing
    Imagel = nothing
    println("Clearing High/Low buffers...")
    GC.gc()

    # 5. Calculate Central Image
    Imagec = trace_variant(trat_large, "Central")

    return dI_dRhigh, Imagec
end


function FiniteDifferences_a(ro, th, phi, DXsize, DYsize, pixels_x, pixels_y, SourceD, freq, maxnstep,h, bhspin, Rout, Rstop)
    """
    Finite differences method to calculate the intensity at each pixel in the image.
    This function calculates the geodesics and the intensity at each pixel using finite differences.
    """
    ah = bhspin + h
    al = bhspin - h

    # Calculate the fov at each dim x and y 
    fovx = DXsize / ro
    fovy = DYsize / ro

    # Calculate the camera position in native coordinates
    Xcamh = MVec4(camera_position(ro, th, phi, ah, Rout))
    Xcaml = MVec4(camera_position(ro, th, phi, al, Rout))

    # Scales the intensity of each pixel by the real size of each pixel
    scale_factor = CalculateScaleFactor(DXsize, DYsize, pixels_x, pixels_y, SourceD, L_unit)
    # integrate_emission_flag setted to false signals that the output of the function will be the trajectory and not the Image
    trajectoryh = CalculateGeodesics(Xcamh, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, ah, Rout, Rstop);
    trajectoryl = CalculateGeodesics(Xcaml, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, al, Rout, Rstop);
    
    # Integrate the emission along the geodesics
    Imageh = IpoleGeoIntensityIntegration(trajectoryh, freq, pixels_x, pixels_y, scale_factor, ah)
    Imagel = IpoleGeoIntensityIntegration(trajectoryl, freq, pixels_x, pixels_y, scale_factor, al)

    #deallocate trajectories to save memory
    trajectoryh = nothing
    trajectoryl = nothing

    dI_da = (Imageh - Imagel) / (2 * h)  # Finite difference approximation

    #calculate the image at the central value
    Xcam = MVec4(camera_position(ro, th, phi, bhspin, Rout))
    trajectory = CalculateGeodesics(Xcam, fovx, fovy, freq, maxnstep, pixels_x, pixels_y, bhspin, Rout, Rstop);
    Imagec = IpoleGeoIntensityIntegration(trajectory, freq, pixels_x, pixels_y, scale_factor, bhspin);
    trajectory = nothing

    return dI_da, Imagec
end

function armijo_line_search!(cost_func, x, grad, direction, bounds, scales,
                             args...; α=1e-4, β=0.5, initial_step=1.0, max_steps=10)
    """
    Improved Armijo line search with proper bounds handling and more aggressive stepping
    """
    step_size = initial_step
    x_new = similar(x)
    
    # Initial cost and directional derivative
    f0 = cost_func(x, args...)
    df0 = dot(grad, direction)
    println("grad = $grad, direction = $direction")
    println("  Line search: f0=$f0, df0=$df0, initial_step=$initial_step")

    #Tell if the direction is increasing or decreasing
    if direction[1] > 0
        println("  Direction for θo is increasing, testing higher θo values")
    else
        println("  Direction for θo is decreasing, testing lower θo values")
    end
    
    # If direction is not a descent direction, return current point
    if df0 >= 0
        @warn "Not a descent direction, df0 = $df0"
        #return x, f0, 0.0
    end
    
    best_x = copy(x)
    best_f = f0
    best_step = 0.0
    
    for i in 1:max_steps
        # Propose new point
        x_new .= x .+ step_size .* direction
        
        # Project onto bounds
        any_bounded = false
        for j in eachindex(x_new)
            old_val = x_new[j]
            x_new[j] = clamp(x_new[j], bounds[j][1], bounds[j][2])
            if x_new[j] != old_val
                any_bounded = true
            end
        end
        
        if any_bounded
            println("  Step $i: \e[31mHit bounds\e[0m")
            if x_new[2] == bounds[2][1] || x_new[2] == bounds[2][2]
                println(" Wanted to try Rhigh = $((x[2] .+ step_size .* direction[2] )* scales[2]), but hit bounds at $(x_new[2] * scales[2])")
                step_size = (x_new[2] - x[2]) / direction[2]
                println("")
            end
            if x_new[1] == bounds[1][1] || x_new[1] == bounds[1][2]
                println(" Wanted to try θo = $((x[1] .+ step_size .* direction[1] )* scales[1]), but hit bounds at $(x_new[1] * scales[1])")
                step_size = (x_new[1] - x[1]) / direction[1]
            end
            #continue
        end
        
        # Check if step is too small. Sometimes we get stuck in a loop of tiny steps
        step_norm = norm(x_new - x)
        if step_norm < 1e-20
            println(" \e[31mStep $i: Step norm too small ($step_norm), breaking\e[0m")
            println(" x_new = $x_new, x = $x, step_size = $step_size")
            break
        end
        
        # Evaluate cost at new point
        f_new = cost_func(x_new, args...)
        println("  Step $i: step_size=$step_size, f_new=$f_new, improvement=$(f0-f_new), Rhigh tested = $(x_new[2] * scales[2]), θo tested = $(x_new[1] * scales[1])")
        
        # Keep track of best point found so far
        if f_new < best_f
            best_x .= x_new
            best_f = f_new
            best_step = step_size
        end
        
        # Check Armijo condition. If necessary, change alpha and beta to be more or less strict
        #armijo_threshold = f0 + α * step_size * df0
        armijo_threshold = f0 + α * step_size * df0
        if f_new <= armijo_threshold
            println("  \e[32mArmijo condition satisfied!\e[0m")
            println(" New cost function value: $f_new")
            return x_new, f_new, step_size, true
        end
        
        println("Reducing step size by beta factor: $β, Armijo Condition: $armijo_threshold, $f0, step_size=$step_size, df0=$df0")
        step_size *= β
        
        # If step becomes too small, return best point found
        if abs(x_new[1] - x[1]) < 1e-2 && abs(x_new[2] * scales[2] - x[2] * scales[2]) < 1e-2
            println("  \e[31mBoth parameter changes below 1e-2 threshold, returning best point found\e[0m")
            return best_x, best_f, best_step, false
        end
    end
    
    # If no Armijo condition met but we found improvements, return best
    if best_f < f0
        println("\e[31m  No Armijo satisfaction, but found improvement: $(f0 - best_f)\e[0m")
        return best_x, best_f, best_step, false
    end
    
    # If no improvement found at all, returning first try (escape local minima)
    println("\e[31mLine search failed to find any improvement\e[0m")
    return x, f0, best_step, false
end


function true_conjugate_gradient_optimization(Iobs, ro, θoi, ai, freq, nx, ny, nmaxstep, 
                                            fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
                                            cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
                                            max_iterations=200, cg_restart_freq=20,
                                            optimize_param::Symbol=:both)
    """
    True conjugate gradient optimization with proper scaling and convergence criteria
    
    Parameters:
    @Iobs: Observed image intensities (2D array) - This is the one we want to fit
    @ro: Observer radial coordinate
    @θoi: Initial guess for inclination angle in degrees
    @ai: Initial guess for spin parameter (0 to 0.994)
    @freq: Observed frequency, usually 230 GHz
    @nx, @ny: Image dimensions. The resolution of the image
    @nmaxstep: Maximum number of integration steps for geodesics
    @fovx, @fovy: Field of view in radians
    @Rout: Outer radius for geodesic integration
    @Rstop: Stopping radius for geodesic integration
    @σ_pixels: Standard deviation for Gaussian filter applied to intensities and gradients
    @cost_tol: Tolerance for cost function convergence
    @param_tol: Tolerance for parameter change convergence
    @grad_tol: Tolerance for gradient norm convergence
    @max_iterations: Maximum number of optimization iterations
    @cg_restart_freq: Frequency of conjugate gradient restarts
    @optimize_param: Symbol indicating which parameter(s) to optimize (:both, :theta, :spin)

    """
    # Pre-allocate trajectory arrays for each thread (this is going to be used in the autodiff function)
    num_threads = Threads.nthreads()
    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads)
    for tid in 1:num_threads
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    # You can optimize either both parameters, only θo, or only a, it depends on the optimize_param argument. Make sure it is one of the three options.
    if !(optimize_param in [:both, :theta, :spin])
        throw(ArgumentError("optimize_param must be :both, :theta, or :spin"))
    end
    
    # Parameter scaling factors (normalize to similar ranges)
    θo_scale = 60.0
    a_scale = 0.6
    scales = [θo_scale, a_scale]
    
    # Initialize with scaled parameters
    x_scaled = [θoi / θo_scale, ai / a_scale]
    bounds_scaled = [(0.1 / θo_scale, 90.0 / θo_scale), 
                     (0.0 / a_scale, 0.994 / a_scale)]
    
    # Determine which parameters to optimize
    optimize_theta = optimize_param in [:both, :theta]
    optimize_spin = optimize_param in [:both, :spin]
    
    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing a: $optimize_spin")
    
    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_da = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)
    
    function compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0)
        # Convert back to physical parameters
        θo_val = x_scaled_val[1] * θo_scale
        a_val = x_scaled_val[2] * a_scale
        println("Running AutoDiffGeoTrajEulerMethod with θo = $θo_val, a = $a_val and applying σ_pixels = $σ_pixels filter")
        # Compute intensities and derivatives
        Threads.@threads for i in 0:(nx-1)
            for j in 0:(ny-1)
                tid = Threads.threadid()
                dI_dθo_out = Ref{Float64}()
                intensity_out = Ref{Float64}()
                dI_da_out = Ref{Float64}()
                
                AutoDiffGeoTrajEulerMethod!(thread_trajs[tid],dI_dθo_out, intensity_out, dI_da_out,
                    ro, θo_val, a_val, nx, ny, nmaxstep, i, j, freq, fovx, fovy, Rout, Rstop)
                
                I_calc[i+1, j+1] = intensity_out[]
                dI_da[i+1, j+1] = dI_da_out[]
                dI_dθo[i+1, j+1] = dI_dθo_out[]
            end
        end
        I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
        dI_dθo = imfilter(dI_dθo, Kernel.gaussian(σ_pixels))
        dI_da = imfilter(dI_da, Kernel.gaussian(σ_pixels))
        cost = cost_func(Iobs, I_calc)
        grad_θo, grad_a = GradientofCostFunction(Iobs, I_calc, dI_dθo, dI_da)
        
        grad_scaled = [grad_θo * θo_scale, grad_a * a_scale]
        
        # Zero out gradients if evolving only theta or only spin. 
        if !optimize_theta
            grad_scaled[1] = 0.0
        end
        if !optimize_spin
            grad_scaled[2] = 0.0
        end
        
        return cost, grad_scaled
    end
    
    function constrained_armijo_line_search!(cost_func, compute_gradients!, x, grad, direction, bounds, scales, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_spin
            constrained_direction[2] = 0.0
        end
        
        return armijo_line_search!(cost_func, compute_gradients!, x, grad, constrained_direction, bounds, scales, args...; kwargs...)
    end
        
    function check_convergence(cost, grad, cost_history, iteration)
        cost_converged = cost < cost_tol
        
        grad_norm = 0.0
        if optimize_theta
            grad_norm += grad[1]^2
        end
        if optimize_spin
            grad_norm += grad[2]^2
        end
        grad_norm = sqrt(grad_norm)
        grad_converged = false # For now, I'm not using grad convergence, maybe implement this later

        # Relative cost improvement over recent iterations
        rel_improvement_converged = false
        if iteration > 10 && length(cost_history) > 10
            recent_improvement = (cost_history[end-9] - cost) / abs(cost_history[end-9])
            rel_improvement_converged = recent_improvement < param_tol
        end
        
        println("  Convergence check: cost=$cost, grad_norm=$grad_norm")
        if iteration > 10
            recent_improvement = length(cost_history) > 10 ? (cost_history[end-9] - cost) / abs(cost_history[end-9]) : Inf
            println("  Recent relative improvement: $recent_improvement")
        end
        println("  Cost converged: $cost_converged, Grad converged: $grad_converged, Stagnant: $rel_improvement_converged")
        
        return cost_converged || grad_converged || rel_improvement_converged
    end

    last_x_computed = nothing
    last_cost = nothing
    last_grad = nothing

    function cached_compute_cost_and_gradients(x_scaled_val, σ_pixels=0.0)
        if last_x_computed !== nothing && last_x_computed ≈ x_scaled_val
            println("Using cached computation for x = $x_scaled_val")
            return last_cost, last_grad
        end
        
        cost, grad = compute_cost_and_gradients(x_scaled_val, σ_pixels)
        last_x_computed = copy(x_scaled_val)
        last_cost = cost
        last_grad = copy(grad)
        return cost, grad
    end
    

    # Initial evaluation
    cost, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        a_final = x_scaled[2] * a_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, a_final, [cost], 1
    end
    # Initialize CG variables
    direction = -copy(grad)
    # Zero out direction for fixed parameters
    if !optimize_theta
        direction[1] = 0.0
    end
    if !optimize_spin
        direction[2] = 0.0
    end
    
    costs = Float64[]
    push!(costs, cost)
    θos = Float64[]
    push!(θos, x_scaled[1] * θo_scale)
    as = Float64[]
    push!(as, x_scaled[2] * a_scale)
    θo_phys = x_scaled[1] * θo_scale
    a_phys = x_scaled[2] * a_scale

    println("Initial cost: $cost, Initial θo: $θo_phys, Initial a: $a_phys")
    println("Initial gradient norm: $(norm(grad))")
    
    x_old = copy(x_scaled)
    step_size = 0.0
    aggressive_initial_step = 0.0
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")
        
        # Line search with constrained direction
        if (iteration == 1)
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
        else
            if(step_size < 0.05/(a_scale * direction[2] ) && optimize_spin)
                step_size = 0.05/(a_scale * direction[2] )
                println("Step size for a is too small, resetting to $step_size")
            end

            if(step_size < 3.0/(θo_scale * direction[1] ) && optimize_theta)
                step_size = 3.0/(θo_scale * direction[1] )
                println("Step size for θo is too small, resetting to $step_size")
            end

            aggressive_initial_step = step_size
            aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))

        end

        println("Trying aggressive initial step: $aggressive_initial_step, set at iteration $iteration")
        println("Cost before line search: $cost")
        cost_comparison = copy(cost)
        x_new, cost_new, step_size, success = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[2],
            x_scaled, grad, direction, bounds_scaled, scales,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )
        
        # Check for sufficient decrease
        println("Line search completed: new cost = $cost_new, step size = $step_size")
        println("Initial cost after line search: $cost")
        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("Cost improvement: $absolute_improvement (relative: $relative_improvement)")
        
        if absolute_improvement <= 1e-16 && iteration > 1
            println("No significant improvement in line search, trying steepest descent")
            direction .= -grad
            # Zero out direction for fixed parameters
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            
            x_new, cost_new, step_size, success = constrained_armijo_line_search!(
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[1],
                (x_val, args...) -> cached_compute_cost_and_gradients(x_val, σ_pixels)[2],
                x_scaled, grad, direction, bounds_scaled, scales,
                α=1e-5, β=0.5, initial_step=step_size * 10
            )
            
            absolute_improvement = cost - cost_new
            if absolute_improvement <= 1e-16
                @warn "No improvement possible, stopping optimization"
                break
            end
        end
        
        if optimize_theta
            x_scaled[1] = x_new[1]
        end
        if optimize_spin
            x_scaled[2] = x_new[2]
        end
        
        param_change = norm(x_new - x_old)
        x_old .= x_scaled
        cost = copy(cost_new)
        push!(costs, cost)
        
        θo_phys = x_scaled[1] * θo_scale
        a_phys = x_scaled[2] * a_scale
        push!(θos, θo_phys)
        push!(as, a_phys)
        println("Updated: cost = $cost, θo = $θo_phys, a = $a_phys, step = $step_size")
        println("Parameter change magnitude: $param_change")
        
        converged = check_convergence(cost, grad, costs, iteration)
        
        relative_param_change = param_change / max(norm(x_scaled), 1e-10)
        if relative_param_change < param_tol
            println("Relative parameter change too small ($relative_param_change), may have converged")
            converged = true
        end
        if converged
            println("Converged! Final θo = $θo_phys, Final a = $a_phys")
            return θos, as, costs, max_iterations
        end
        
        grad_old = copy(grad)
        _, grad = cached_compute_cost_and_gradients(x_scaled, σ_pixels)
        
        if iteration % cg_restart_freq == 0 || norm(grad) > 10 * norm(grad_old)
            direction .= -grad
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_spin
                direction[2] = 0.0
            end
            println("CG restart at iteration $iteration")
        else
            # Polak-Ribière formula
            grad_diff = grad - grad_old
            beta_denom = dot(grad_old, grad_old)
            
            if beta_denom > 1e-16
                beta = max(0.0, dot(grad, grad_diff) / beta_denom)
                
                beta = min(beta, 2.0)
                
                direction .= -grad .+ beta .* direction
                
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_spin
                    direction[2] = 0.0
                end
                
                active_grad_norm = 0.0
                active_dir_norm = 0.0
                active_dot = 0.0
                
                if optimize_theta
                    active_grad_norm += grad[1]^2
                    active_dir_norm += direction[1]^2
                    active_dot += direction[1] * grad[1]
                end
                if optimize_spin
                    active_grad_norm += grad[2]^2
                    active_dir_norm += direction[2]^2
                    active_dot += direction[2] * grad[2]
                end
                
                active_grad_norm = sqrt(active_grad_norm)
                active_dir_norm = sqrt(active_dir_norm)
                
                if active_dot >= -1e-10 * active_dir_norm * active_grad_norm
                    direction .= -grad
                    if !optimize_theta
                        direction[1] = 0.0
                    end
                    if !optimize_spin
                        direction[2] = 0.0
                    end
                    println("Reset to steepest descent (not descent direction)")
                end
                
                println("CG update: beta = $beta")
            else
                direction .= -grad
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_spin
                    direction[2] = 0.0
                end
                println("Beta = $beta, Reset to steepest descent (small denominator)")
            end
        end
    end
    
    @warn "Maximum iterations reached without convergence"
    θo_final = x_scaled[1] * θo_scale
    a_final = x_scaled[2] * a_scale
    return θos, as, costs, max_iterations
end





function true_conjugate_gradient_optimization_GRMHD(Iobs, ro, θoi, Rhighi, freq, nx, ny, nmaxstep, 
                                            fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
                                            cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
                                            max_iterations=200, cg_restart_freq=20,
                                            optimize_param::Symbol=:both, dump_filepath = nothing, sensemode = "AD")
    """
    True conjugate gradient optimization with proper scaling and convergence criteria
    
    Parameters:
    @Iobs: Observed image intensities (2D array) - This is the one we want to fit
    @ro: Observer radial coordinate
    @θoi: Initial guess for inclination angle in degrees
    @ai: Initial guess for spin parameter (0 to 0.994)
    @freq: Observed frequency, usually 230 GHz
    @nx, @ny: Image dimensions. The resolution of the image
    @nmaxstep: Maximum number of integration steps for geodesics
    @fovx, @fovy: Field of view in radians
    @Rout: Outer radius for geodesic integration
    @Rstop: Stopping radius for geodesic integration
    @σ_pixels: Standard deviation for Gaussian filter applied to intensities and gradients
    @cost_tol: Tolerance for cost function convergence
    @param_tol: Tolerance for parameter change convergence
    @grad_tol: Tolerance for gradient norm convergence
    @max_iterations: Maximum number of optimization iterations
    @cg_restart_freq: Frequency of conjugate gradient restarts
    @optimize_param: Symbol indicating which parameter(s) to optimize (:both, :theta, :spin)

    """
    # Pre-allocate trajectory arrays for each thread (this is going to be used in the autodiff function)
    num_threads = Threads.nthreads()
    simulation_data = load_data(dump_filepath, Rhighi);

    thread_trajs = Vector{Vector{OfTraj}}(undef, num_threads+1)
    for tid in 1:(num_threads+1)
        thread_trajs[tid] = Vector{OfTraj}()
        sizehint!(thread_trajs[tid], nmaxstep)
    end

    # You can optimize either both parameters, only θo, or only a, it depends on the optimize_param argument. Make sure it is one of the three options.
    if !(optimize_param in [:both, :theta, :Rhigh])
        throw(ArgumentError("optimize_param must be :both, :theta, or :Rhigh"))
    end

    if !(sensemode in ["AD", "FD"])
        throw(ArgumentError("sensemode arg must be either 'AD' or 'FD'"))
    end

    sucess::Bool = false
    
    # Parameter scaling factors (normalize to similar ranges)
    θo_scale = 100.0
    Rhigh_scale = 200.0
    scales = [θo_scale, Rhigh_scale]
    
    # Initialize with scaled parameters
    x_scaled = [θoi / θo_scale, Rhighi / Rhigh_scale]
    bounds_scaled = [(0.1 / θo_scale, 175.0 / θo_scale), 
                     (0.0 / Rhigh_scale, 100.0 / Rhigh_scale)]
    
    # Determine which parameters to optimize
    optimize_theta = optimize_param in [:both, :theta]
    optimize_Rhigh = optimize_param in [:both, :Rhigh]
    
    println("Optimization mode: $optimize_param")
    println("Optimizing θo: $optimize_theta, Optimizing Rhigh: $optimize_Rhigh")
    
    dI_dθo = Matrix{Float64}(undef, nx, ny)
    dI_dRhigh = Matrix{Float64}(undef, nx, ny)
    I_calc = Matrix{Float64}(undef, nx, ny)
    
    function compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels=0.0, simulation_data = nothing, sensemode = "AD")
        # Convert back to physical parameters
        θo_val = x_scaled_val[1] * θo_scale
        Rhigh_val = x_scaled_val[2] * Rhigh_scale
        if optimize_Rhigh
            simulation_data = load_data(dump_filepath, Rhigh_val);
        end
        if(compute_gradients)
            println("\n Running AutoDiffGeoTrajEulerMethod with θo = \e[32m$θo_val\e[0m, Rhigh = \e[32m$Rhigh_val\e[0m and applying σ_pixels = \e[32m$σ_pixels\e[0m filter")
            if(sensemode == "AD")
                Threads.@threads for i in 0:(nx-1)
                    for j in 0:(ny-1)
                        tid = Threads.threadid()
                        dI_dθo_out = Ref{Float64}()
                        intensity_out = Ref{Float64}()
                        dI_dRhigh_out = Ref{Float64}()
                        
                        AutoDiffGeoTrajEulerMethod_GRMHD!(thread_trajs[tid], dI_dθo_out, intensity_out, dI_dRhigh_out,
                            ro, θo_val, phi, params.a, nx, ny, nmaxstep, i, j, freq, fovx, fovy, Rout, Rstop, simulation_data)
                        I_calc[i+1, j+1] = intensity_out[]
                        dI_dRhigh[i+1, j+1] = dI_dRhigh_out[]
                        dI_dθo[i+1, j+1] = dI_dθo_out[]
                    end
                end
            elseif(sensemode == "FD")
                println("Using Finite Differences to compute gradients")
                dI_dθo, I_calc = FiniteDifferencesθ(ro, θo_val, phi, DXsize, DYsize, nx, ny, SourceD, freq, nmaxstep, 1e-4, params.a, Rout, Rstop, simulation_data)
                if(optimize_Rhigh)
                    error("Finite Differences for Rhigh not implemented yet")
                    dI_dRhigh, _ = FiniteDifferences_Rhigh(ro, θo_val, phi, DXsize, DYsize, nx, ny, SourceD, freq, nmaxstep, 1e-4, params.a, Rout, Rstop, simulation_data)
                end
            else
                throw(ArgumentError("sensemode must be either 'AD' or 'FD'"))
            end
            I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
            dI_dθo = imfilter(dI_dθo, Kernel.gaussian(σ_pixels))
            dI_dRhigh = imfilter(dI_dRhigh, Kernel.gaussian(σ_pixels))
            cost = cost_func(Iobs, I_calc)
            grad_θo, grad_Rhigh = GradientofCostFunction(Iobs, I_calc, dI_dθo, dI_dRhigh)
            
            grad_scaled = [grad_θo * θo_scale, grad_Rhigh * Rhigh_scale]
            
            # Zero out gradients if evolving only theta or only rhigh. 
            if !optimize_theta
                grad_scaled[1] = 0.0
            end
            if !optimize_Rhigh
                grad_scaled[2] = 0.0
            end
            
            return cost, grad_scaled
        else
            println("\n Computing image with \e[32mθo = $θo_val\e[0m, \e[32mRhigh = $Rhigh_val\e[0m and applying σ_pixels = \e[32m$σ_pixels\e[0m filter")
            # Find camera in native coordinates
            Xcamera = MVec4(camera_position(ro, θo_val, phi, params.a, params.Rout))
            # Scales the intensity of each pixel by the real size of each pixel
            scale_factor = CalculateScaleFactor(DXsize, DYsize, nx, ny, SourceD, L_unit)
            Threads.@threads for i in 0:(nx - 1)
                tid = Threads.threadid()
                for j in 0:(ny - 1)
                    empty!(thread_trajs[tid])
                    nstep = get_pixel(thread_trajs[tid], i, j, Xcamera, nmaxstep, fovx, fovy, freq_unitless, nx, ny, params.a, Rh, params.Rout, Rstop, xoff, yoff) 
                    integrate_emission!(thread_trajs[tid], nstep, I_calc, i + 1, j + 1, freq, params.a, simulation_data)
                    empty!(thread_trajs[tid])
                end
            end   
        end
        I_calc *= freq^3; # in the autodiff method this is done inside the function
        I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
        cost = cost_func(Iobs, I_calc)
        return cost, zeros(2) # Return zero gradients when not computing them
    end
    
    function constrained_armijo_line_search!(cost_func, x, grad, direction, bounds, scales, args...; kwargs...)
        constrained_direction = copy(direction)
        if !optimize_theta
            constrained_direction[1] = 0.0
        end
        if !optimize_Rhigh
            constrained_direction[2] = 0.0
        end
        
        return armijo_line_search!(cost_func, x, grad, constrained_direction, bounds, scales, args...; kwargs...)
    end
        
    function check_convergence(cost, grad, cost_history, iteration)
        cost_converged = cost < cost_tol
        
        grad_norm = 0.0
        if optimize_theta
            grad_norm += grad[1]^2
        end
        if optimize_Rhigh
            grad_norm += grad[2]^2
        end
        grad_norm = sqrt(grad_norm)
        grad_converged = false # For now, I'm not using grad convergence, maybe implement this later

        # Relative cost improvement over recent iterations
        rel_improvement_converged = false
        if iteration > 10 && length(cost_history) > 10
            recent_improvement = (cost_history[end-9] - cost) / abs(cost_history[end-9])
            rel_improvement_converged = recent_improvement < param_tol
        end
        
        println("  Convergence check: cost=$cost, grad_norm=$grad_norm")
        if iteration > 10
            recent_improvement = length(cost_history) > 10 ? (cost_history[end-9] - cost) / abs(cost_history[end-9]) : Inf
            println("  Recent relative improvement: $recent_improvement")
        end
        println("  Cost converged: $cost_converged, Grad converged: $grad_converged, Stagnant: $rel_improvement_converged")
        
        return cost_converged || grad_converged || rel_improvement_converged
    end

    last_x_computed = nothing
    last_cost = nothing
    last_grad = nothing

    function cached_compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels=0.0, simulation_data = nothing, sensemode = "AD")
        c, g = compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels, simulation_data, sensemode)
        
        last_x_computed = copy(x_scaled_val)
        last_cost = c
        last_grad = copy(g)
        
        return c, g
    end
    

    # Initial evaluation
    cost, grad = cached_compute_cost_and_gradients(x_scaled, true, σ_pixels, simulation_data, sensemode)
    initial_cost = cost

    if check_convergence(cost, grad, [cost], 0)
        θo_final = x_scaled[1] * θo_scale
        Rhigh_final = x_scaled[2] * Rhigh_scale
        println("Initial solution already satisfies tolerance")
        return θo_final, Rhigh_final, [cost], 1
    end
    # Initialize CG variables
    direction = -copy(grad)
    # Zero out direction for fixed parameters
    if !optimize_theta
        direction[1] = 0.0
    end
    if !optimize_Rhigh
        direction[2] = 0.0
    end
    
    costs = Float64[]
    push!(costs, cost)
    θos = Float64[]
    push!(θos, x_scaled[1] * θo_scale)
    Rhighs = Float64[]
    push!(Rhighs, x_scaled[2] * Rhigh_scale)
    θo_phys = x_scaled[1] * θo_scale
    Rhigh_phys = x_scaled[2] * Rhigh_scale

    println("Initial cost: $cost, Initial θo: $θo_phys, Initial Rhigh: $Rhigh_phys")
    println("Initial gradient norm: $(norm(grad))")
    
    x_old = copy(x_scaled)
    step_size = 0.0
    aggressive_initial_step = 0.0
    direction_old = copy(direction)
    grad_old = copy(grad)
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")
        
        # Line search with constrained direction
        if (iteration == 1)
            #aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
            #aggressive_initial_step = 0.3/norm(grad)
            # Define the maximum physical jump you'll allow in the first step
            max_dθo_phys = 20.0   # degrees
            max_dRhigh_phys = 10.0 # Rhigh units
            
            # Convert to scaled limits
            max_dθo_scaled = max_dθo_phys / θo_scale
            max_dRhigh_scaled = max_dRhigh_phys / Rhigh_scale
            
            # Calculate the step size required to hit those limits
            step_limit_θo = (optimize_theta && abs(direction[1]) > 1e-16) ? (max_dθo_scaled / abs(direction[1])) : Inf
            step_limit_Rhigh = (optimize_Rhigh && abs(direction[2]) > 1e-16) ? (max_dRhigh_scaled / abs(direction[2])) : Inf
            
            # Pick the step size that keeps both parameters within their max jump
            aggressive_initial_step = max(step_limit_θo, step_limit_Rhigh)
        else
            if(optimize_Rhigh && abs(direction[2]) > 1e-16)
                min_step_Rhigh = 5.0 / (Rhigh_scale * abs(direction[2]))
                if step_size < min_step_Rhigh
                    step_size = min_step_Rhigh
                    println("Step size for Rhigh is too small, resetting to $step_size")
                end
            end

            if(optimize_theta && abs(direction[1]) > 1e-16)
                min_step_theta = 5.0 / (θo_scale * abs(direction[1]))
                if step_size < min_step_theta
                    step_size = min_step_theta
                    println("Step size for θo is too small, resetting to $step_size")
                end
            end
            if(sucess)
                # Calculate directional derivatives (dot products)
                dir_deriv_old = dot(grad_old, direction_old)
                dir_deriv_new = dot(grad, direction)
                
                # Prevent division by zero and ensure we only scale if it makes sense
                if abs(dir_deriv_new) > 1e-40
                    correction_factor = dir_deriv_old / dir_deriv_new
                    
                    # Clamp it so it doesn't explode if the new derivative is extremely flat
                    correction_factor = clamp(correction_factor, 0.1, 5.0) 
                    
                    aggressive_initial_step = step_size * correction_factor
                    println("Line search successful. CG scaled step size by $correction_factor -> new guess: $aggressive_initial_step")
                else
                    aggressive_initial_step = step_size * 1.5 # Fallback expansion
                end
            else
                println("Line search failed to find a better point, resetting aggressive initial step")
                #aggressive_initial_step = max(3.0, 0.3 / max(norm(grad), 1e-12))
                #aggressive_initial_step = 0.3/norm(grad)
                # Define the maximum physical jump you'll allow in the first step
                max_dθo_phys = 10.0   # degrees
                max_dRhigh_phys = 10.0 # Rhigh units
                
                # Convert to scaled limits
                max_dθo_scaled = max_dθo_phys / θo_scale
                max_dRhigh_scaled = max_dRhigh_phys / Rhigh_scale
                
                # Calculate the step size required to hit those limits
                step_limit_θo = (optimize_theta && abs(direction[1]) > 1e-16) ? (max_dθo_scaled / abs(direction[1])) : Inf
                step_limit_Rhigh = (optimize_Rhigh && abs(direction[2]) > 1e-16) ? (max_dRhigh_scaled / abs(direction[2])) : Inf
                
                # Pick the step size that keeps both parameters within their max jump
                aggressive_initial_step = max(step_limit_θo, step_limit_Rhigh)
            end
        end

        cost_comparison = copy(cost)
        x_new, cost_new, step_size, sucess = constrained_armijo_line_search!(
            (x_val, args...) -> cached_compute_cost_and_gradients(x_val, false, σ_pixels, simulation_data, sensemode)[1],
            x_scaled, grad, direction, bounds_scaled, scales,
            α=1e-5, β=0.5, initial_step=aggressive_initial_step, max_steps=15
        )
        
        # Check for sufficient decrease
        absolute_improvement = cost_comparison - cost_new
        relative_improvement = absolute_improvement / max(abs(cost_comparison), 1e-16)
        println("\e[34mCost improvement: $absolute_improvement (relative: $relative_improvement)\e[0m")
        
        if absolute_improvement <= 1e-16 && iteration > 1
            println("Stagnation detected — probing directions")

            x_trial = copy(x_scaled)
            best_cost = cost
            best_x = copy(x_scaled)

            θ_step = 5.0 / θo_scale
            R_step = 5.0 / Rhigh_scale

            max_rounds = 10
            shrink = 0.8

            found_escape = false

            for round in 1:max_rounds
                println("Probe round $round with θ_step=$(θ_step*θo_scale), R_step=$(R_step*Rhigh_scale)")

                directions = Vector{Vector{Float64}}()

                if optimize_theta
                    push!(directions, [ θ_step, 0.0])
                    push!(directions, [-θ_step, 0.0])
                end

                if optimize_Rhigh
                    push!(directions, [0.0,  R_step])
                    push!(directions, [0.0, -R_step])
                end

                for d in directions

                    x_trial .= x_scaled .+ d

                    x_trial[1] = clamp(x_trial[1], bounds_scaled[1]...)
                    x_trial[2] = clamp(x_trial[2], bounds_scaled[2]...)

                    trial_cost, _ = cached_compute_cost_and_gradients(
                        x_trial, false, σ_pixels, simulation_data, sensemode
                    )

                    println("  Probe direction $d → cost = $trial_cost")

                    if trial_cost < best_cost
                        best_cost = trial_cost
                        best_x .= x_trial
                        found_escape = true
                    end
                end

                if found_escape
                    break
                end

                θ_step *= shrink
                R_step *= shrink
            end

            if found_escape
                println("Escape found! Restarting CG")
                x_scaled .= best_x
                cost = best_cost
                #if cost is lesser than tolerance, we can consider it as an answer
                converged = check_convergence(cost, grad, costs, iteration)

                if converged
                    #print in green color
                    println("\e[32mConverged after escape! Final θo = $(x_scaled[1] * θo_scale), Final Rhigh = $(x_scaled[2] * Rhigh_scale)\e[0m")
                    println("\e[32mFinal cost: $cost\e[0m")
                    θo_phys = x_scaled[1] * θo_scale
                    Rhigh_phys = x_scaled[2] * Rhigh_scale
                    push!(costs, cost)

                    push!(θos, θo_phys)
                    push!(Rhighs, Rhigh_phys)
                    return θos, Rhighs, costs, max_iterations
                end

                # Update old tracking variables before calculating the new ones!
                grad_old .= grad
                direction_old .= direction

                _, grad = cached_compute_cost_and_gradients(
                    x_scaled, true, σ_pixels, simulation_data, sensemode
                )

                direction .= -grad
                continue
            else
                println("No escape found after $max_rounds rounds — stopping")
                break
            end
        end
        
        if optimize_theta
            x_scaled[1] = x_new[1]
        end
        if optimize_Rhigh
            x_scaled[2] = x_new[2]
        end
        
        param_change = norm(x_new - x_old)
        x_old .= x_scaled
        cost = copy(cost_new)
        push!(costs, cost)
        
        θo_phys = x_scaled[1] * θo_scale
        Rhigh_phys = x_scaled[2] * Rhigh_scale
        push!(θos, θo_phys)
        push!(Rhighs, Rhigh_phys)
        println("Updated: cost = $cost, θo = $θo_phys, Rhigh = $Rhigh_phys, step = $step_size")
        println("Parameter change magnitude: $param_change")
        
        converged = check_convergence(cost, grad, costs, iteration)
        
        relative_param_change = param_change / max(norm(x_scaled), 1e-10)
        if relative_param_change < param_tol
            println("Relative parameter change too small ($relative_param_change), may have converged")
            println("However, we won't stop")

            converged = false
        end
        if converged
            println("Converged! Final θo = $θo_phys, Final Rhigh = $Rhigh_phys")
            return θos, Rhighs, costs, iteration
        end
        
        grad_old = copy(grad)
        direction_old = copy(direction)
        _, grad = cached_compute_cost_and_gradients(x_scaled, true, σ_pixels, simulation_data, sensemode)
        
        if iteration % cg_restart_freq == 0 || norm(grad) > 10 * norm(grad_old)
            direction .= -grad
            if !optimize_theta
                direction[1] = 0.0
            end
            if !optimize_Rhigh
                direction[2] = 0.0
            end
            println("CG restart at iteration $iteration")
        else
            # Polak-Ribière formula
            grad_diff = grad - grad_old
            beta_denom = dot(grad_old, grad_old)
            
            if beta_denom > 1e-16
                beta = max(0.0, dot(grad, grad_diff) / beta_denom)
                
                beta = min(beta, 2.0)
                
                direction .= -grad .+ beta .* direction
                
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_Rhigh
                    direction[2] = 0.0
                end
                
                active_grad_norm = 0.0
                active_dir_norm = 0.0
                active_dot = 0.0
                
                if optimize_theta
                    active_grad_norm += grad[1]^2
                    active_dir_norm += direction[1]^2
                    active_dot += direction[1] * grad[1]
                end
                if optimize_Rhigh
                    active_grad_norm += grad[2]^2
                    active_dir_norm += direction[2]^2
                    active_dot += direction[2] * grad[2]
                end
                
                active_grad_norm = sqrt(active_grad_norm)
                active_dir_norm = sqrt(active_dir_norm)
                
                if active_dot >= -1e-10 * active_dir_norm * active_grad_norm
                    direction .= -grad
                    if !optimize_theta
                        direction[1] = 0.0
                    end
                    if !optimize_Rhigh
                        direction[2] = 0.0
                    end
                    println("Reset to steepest descent (not descent direction)")
                end
                
                println("CG update: beta = $beta")
            else
                direction .= -grad
                if !optimize_theta
                    direction[1] = 0.0
                end
                if !optimize_Rhigh
                    direction[2] = 0.0
                end
                println("Reset to steepest descent (small denominator)")
            end
        end
    end
    
    @warn "Maximum iterations reached without convergence"
    θo_final = x_scaled[1] * θo_scale
    Rhigh_final = x_scaled[2] * Rhigh_scale
    return θos, Rhighs, costs, max_iterations
end

# using Optim
# using NLSolversBase
# function ConjGradOptim(Iobs, ro, θoi, Rhighi, freq, nx, ny, nmaxstep, 
#                                             fovx, fovy, Rout, Rstop, σ_pixels=0.0; 
#                                             cost_tol=2e-11, param_tol=1e-8, grad_tol=1e-10,
#                                             max_iterations=200, cg_restart_freq=20, # cg_restart_freq is ignored by Optim, but kept for signature compatibility
#                                             optimize_param::Symbol=:both, dump_filepath = nothing, sensemode = "AD")

    
#     num_threads = Threads.nthreads()
#     simulation_data = load_data(dump_filepath, Rhighi);
#     MAX_THREAD_ID = 16
#     thread_trajs = Vector{Vector{OfTraj}}(undef, MAX_THREAD_ID)
#     for tid in 1:MAX_THREAD_ID
#         default_mvector = MVector{4, Float64}(0.0, 0.0, 0.0, 0.0)
#         thread_trajs[tid] = [
#             OfTraj(0.0, 
#                 default_mvector, default_mvector, default_mvector, default_mvector,
#                 default_mvector, default_mvector, default_mvector, default_mvector) 
#             for _ in 1:nmaxstep
#         ]
#     end

#     if !(optimize_param in [:both, :theta, :Rhigh])
#         throw(ArgumentError("optimize_param must be :both, :theta, or :Rhigh"))
#     end

#     if !(sensemode in ["AD", "FD"])
#         throw(ArgumentError("sensemode arg must be either 'AD' or 'FD'"))
#     end

#     # Parameter scaling factors (normalize to similar ranges)
#     θo_scale = 60.0
#     Rhigh_scale = 120.0
    
#     # Initialize with scaled parameters
#     x_scaled = [θoi / θo_scale, Rhighi / Rhigh_scale]
    
#     # Setup flat arrays for Optim's Fminbox
#     lower_bounds = [0.1 / θo_scale, 0.0 / Rhigh_scale]
#     upper_bounds = [179.9 / θo_scale, 200.0 / Rhigh_scale]
    
#     # Determine which parameters to optimize
#     # Determine which parameters to optimize
#     optimize_theta = optimize_param in [:both, :theta]
#     optimize_Rhigh = optimize_param in [:both, :Rhigh]
    
#     # NEW: Create a map of active parameters
#     active_indices = Int[]
#     if optimize_theta; push!(active_indices, 1); end
#     if optimize_Rhigh; push!(active_indices, 2); end

#     # Create arrays containing ONLY the parameters we want to move
#     active_x = x_scaled[active_indices]
#     active_lower = lower_bounds[active_indices]
#     active_upper = upper_bounds[active_indices]
    
#     println("Optimization mode: $optimize_param")
#     println("Optimizing θo: $optimize_theta, Optimizing Rhigh: $optimize_Rhigh")
    
#     dI_dθo = Matrix{Float64}(undef, nx, ny)
#     dI_dRhigh = Matrix{Float64}(undef, nx, ny)
#     I_calc = Matrix{Float64}(undef, nx, ny)

#     # --- 2. CORE EVALUATION FUNCTIONS (From your original code) ---
#     function compute_cost_and_gradients(x_scaled_val, compute_gradient, σ_pixels=0.0, simulation_data=nothing, sensemode="AD")
#         # Convert back to physical parameters
#         θo_val = x_scaled_val[1] * θo_scale
#         Rhigh_val = x_scaled_val[2] * Rhigh_scale
        
#         if optimize_Rhigh 
#             simulation_data = load_data(dump_filepath, Rhigh_val);
#         end
        
#         if compute_gradient
#             println("\n Running AutoDiffGeoTrajEulerMethod with θo = \e[32m$θo_val\e[0m, Rhigh = \e[32m$Rhigh_val\e[0m and applying σ_pixels = \e[32m$σ_pixels\e[0m filter")
#             if sensemode == "AD"
#                 Threads.@threads for i in 0:(nx-1)
#                     for j in 0:(ny-1)
#                         tid = Threads.threadid()
#                         dI_dθo_out = Ref{Float64}()
#                         intensity_out = Ref{Float64}()
#                         dI_dRhigh_out = Ref{Float64}()
                        
#                         AutoDiffGeoTrajEulerMethod_GRMHD!(thread_trajs[tid], dI_dθo_out, intensity_out, dI_dRhigh_out,
#                             ro, θo_val, phi, params.a, nx, ny, nmaxstep, i, j, freq, fovx, fovy, Rout, Rstop, simulation_data)
#                         I_calc[i+1, j+1] = intensity_out[]
#                         dI_dRhigh[i+1, j+1] = dI_dRhigh_out[]
#                         dI_dθo[i+1, j+1] = dI_dθo_out[]
#                     end
#                 end
#             elseif sensemode == "FD"
#                 println("Using Finite Differences to compute gradients")
#                 dI_dθo, I_calc = FiniteDifferencesθ(ro, θo_val, phi, DXsize, DYsize, nx, ny, SourceD, freq, nmaxstep, 1e-4, params.a, Rout, Rstop, simulation_data)
#                 if optimize_Rhigh
#                     dI_dRhigh, _ = FiniteDifferences_Rhigh(ro, θo_val, phi, DXsize, DYsize, nx, ny, SourceD, freq, nmaxstep, 1e-4, params.a, Rout, Rstop, simulation_data)
#                 end
#             end
            
#             I_calc = imfilter(I_calc, Kernel.gaussian(σ_pixels))
#             dI_dθo = imfilter(dI_dθo, Kernel.gaussian(σ_pixels))
#             dI_dRhigh = imfilter(dI_dRhigh, Kernel.gaussian(σ_pixels))
            
#             cost = cost_func(Iobs, I_calc)
#             grad_θo, grad_Rhigh = GradientofCostFunction(Iobs, I_calc, dI_dθo, dI_dRhigh)
#             grad_scaled = [grad_θo * θo_scale, grad_Rhigh * Rhigh_scale]
            
#             return cost, grad_scaled
#         else 
#             println("\n Computing image with \e[32mθo = $θo_val\e[0m, \e[32mRhigh = $Rhigh_val\e[0m and applying σ_pixels = \e[32m$σ_pixels\e[0m filter")
#             Xcamera = MVec4(camera_position(ro, θo_val, phi, params.a, params.Rout))
#             scale_factor = CalculateScaleFactor(DXsize, DYsize, nx, ny, SourceD, L_unit)
            
#             Threads.@threads for i in 0:(nx - 1)
#                 tid = Threads.threadid()
#                 for j in 0:(ny - 1)
#                     sizehint!(thread_trajs[tid], nmaxstep)
#                     nstep = get_pixel(thread_trajs[tid], i, j, Xcamera, nmaxstep, fovx, fovy, freq_unitless, nx, ny, params.a, Rh, params.Rout, Rstop, xoff, yoff) 
                    
#                     resize!(thread_trajs[tid], nstep)
#                     integrate_emission!(thread_trajs[tid], nstep, I_calc, i + 1, j + 1, freq, params.a, simulation_data)
#                 end
#             end   
#             I_calc *= freq^3
#             imfilter(I_calc, Kernel.gaussian(σ_pixels))
#             cost = cost_func(Iobs, I_calc)

#             return cost, zeros(2)
#         end
#     end

#     # Keep caching to prevent redundant radiative transfer integrations
#     last_x_computed = nothing
#     last_cost = nothing
#     last_grad = nothing
#     last_had_gradients = false 

#     function cached_compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels=0.0, simulation_data=nothing, sensemode="AD")
#         if last_x_computed !== nothing && last_x_computed ≈ x_scaled_val
#             if !compute_gradients || last_had_gradients
#                 println("Using cached computation for x = $x_scaled_val (has_gradients: $last_had_gradients)")
#                 return last_cost, last_grad
#             end
#         end
        
#         cost, grad = compute_cost_and_gradients(x_scaled_val, compute_gradients, σ_pixels, simulation_data, sensemode)
#         last_x_computed = copy(x_scaled_val)
#         last_cost = cost
#         last_grad = copy(grad)
#         last_had_gradients = compute_gradients
#         return cost, grad
#     end


#     # --- 3. OPTIM.JL INTEGRATION ---
    
#     # History tracking arrays
#     costs = Float64[]
#     θos = Float64[]
#     Rhighs = Float64[]

#     # The fg! interface required by Optim.jl for joint cost/gradient evaluation
#     function fg!(F, G, x_opt)
#         x_full = copy(x_scaled) 
#         x_full[active_indices] .= x_opt
        
#         compute_grads = G !== nothing 
#         cost, grad_full = cached_compute_cost_and_gradients(x_full, compute_grads, σ_pixels, simulation_data, sensemode)

#         # NEW: Scale up the objective so the optimizer takes bigger steps!
#         cost_scale = 100000.0 

#         if G !== nothing
#             G .= grad_full[active_indices] .* cost_scale
#         end

#         if F !== nothing
#             return cost * cost_scale
#         end
#     end

#     function trace_callback(tr)
#         curr_opt_x = hasproperty(tr, :x) ? tr.x : tr.metadata["x"]
#         curr_cost = hasproperty(tr, :f_x) ? tr.f_x : tr.value
        
#         # Reconstruct full x for tracking history
#         curr_full_x = copy(x_scaled)
#         curr_full_x[active_indices] .= curr_opt_x
        
#         push!(costs, curr_cost)
#         push!(θos, curr_full_x[1] * θo_scale)
#         push!(Rhighs, curr_full_x[2] * Rhigh_scale)
        
#         return false 
#     end

   

#     # Wrap function for Optim, passing only the active initial array
#     objective = OnceDifferentiable(NLSolversBase.only_fg!(fg!), active_x)

#     println("Starting Optim.jl Fminbox(LBFGS())...")

#     # Run the optimization using the active subset
#     result = optimize(objective, active_lower, active_upper, active_x, 
#                       Fminbox(LBFGS()), 
#                       Optim.Options(
#                           iterations = max_iterations,
#                           f_reltol = cost_tol,
#                           g_tol = grad_tol,
#                           x_abstol = param_tol,
#                           show_trace = true,
#                           extended_trace = true,
#                           callback = trace_callback
#                       ))

#     # --- 4. EXTRACT RESULTS ---
#     println("\n--- Optimization Finished! ---")
#     println(result)
    
#     # Optional: If you want to ensure the very first point is in your history arrays,
#     # you can prepend them or just rely on the trace_callback output.
    
#     total_iters = Optim.iterations(result)
    
#     return θos, Rhighs, costs, total_iters
# end