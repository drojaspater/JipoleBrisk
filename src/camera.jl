export camera_position

include("../src/metrics.jl")


function root_find_ad_safe(x, cstartx, cstopx)
    x3_val = root_find(x, cstartx, cstopx)

    T = eltype(x)
    c1 = zero(T)
    c2 = log(x[2])
    c4 = x[4]

    x_eval = @SVector [c1, c2, x3_val, c4]

    eps = 1e-7
    
    x_eval_eps = @SVector [c1, c2, x3_val + eps, c4]
    
    slope = (theta_func(x_eval_eps) - theta_func(x_eval)) / eps

    return x3_val - (theta_func(x_eval) - x[3]) / slope
end

function root_find(x, cstartx, cstopx)
    """
    Finds the root of the theta function using a bisection method.
    Parameters:
    @x: Vector of position coordinates in internal coordinates.
    """
    th = x[3]
    T = eltype(x)
    
    # Cache the constant parts of the vector
    c1 = zero(T)
    c2 = log(x[2])
    c4 = x[4]

    # Track ONLY the scalar theta values
    xa3 = zero(T)
    xb3 = zero(T)

    if x[3] < π / 2.
        xa3 = cstartx[3]
        xb3 = (cstopx[3] - cstartx[3]) / 2 + SMALL
    else
        xa3 = (cstopx[3] - cstartx[3]) / 2 - SMALL
        xb3 = cstopx[3]
    end

    tol::Float64 = 1.e-9
    
    # Build SVectors on the fly for the function calls
    tha = theta_func(@SVector [c1, c2, xa3, c4])
    thb = theta_func(@SVector [c1, c2, xb3, c4])

    if abs(tha - th) < tol
        return xa3
    elseif abs(thb - th) < tol
        return xb3
    end

    xc3 = zero(T)
    
    for i in 1:1000
        xc3 = 0.5 * (xa3 + xb3)
        thc = theta_func(@SVector [c1, c2, xc3, c4])

        if (thc - th) * (thb - th) < 0.
            xa3 = xc3
        else
            xb3 = xc3
        end

        err = thc - th
        if abs(err) < tol
            break
        end
    end

    return xc3
end

function root_find_newton(x, cstartx, cstopx)
    """
    Finds the root of the theta function using Newton's method.
    Parameters:
    @x: Vector of position coordinates in internal coordinates.
    """
    # Target value we are trying to match
    th_target = x[3]

    # Initialize working vector
    xc = zeros(eltype(x), length(x))
    
    # Pre-fill constant components (assuming index 2 is log-radius and 4 is generic)
    xc[2] = log(x[2])
    xc[4] = x[4]

    # --- Initial Guess ---
    # We start at the midpoint of the search interval to be safe
    xc[3] = 0.5 * (cstartx[3] + cstopx[3])

    # Settings for Newton's Method
    tol::Float64 = 1.e-16
    max_iter::Int = 100
    epsilon::Float64 = 1.e-7 # Step size for finite difference derivative

    for i in 1:max_iter
        # 1. Evaluate the function at current guess
        # f(theta) = theta_func(theta) - th_target
        current_val = theta_func(xc)
        f_val = current_val - th_target

        # Check convergence
        if abs(f_val) < tol
            return xc[3]
        end

        # 2. Calculate Derivative using Finite Difference
        # f'(theta) ≈ (f(theta + eps) - f(theta)) / eps
        current_theta = xc[3]
        xc[3] = current_theta + epsilon
        val_plus = theta_func(xc)
        
        derivative = (val_plus - current_val) / epsilon

        # Reset xc[3] to current theta for the update step
        xc[3] = current_theta

        # Avoid division by zero
        if abs(derivative) < 1.e-14
            println("Warning: Derivative close to zero, Newton method failed.")
            break
        end

        # 3. Newton Update Step
        # x_new = x_old - f(x) / f'(x)
        xc[3] = xc[3] - (f_val / derivative)
        
        # Optional: Clamp the result to stay within bounds if necessary
        # xc[3] = clamp(xc[3], cstartx[3], cstopx[3])
    end

    return xc[3]
end


function camera_position(cam_dist::Float64, cam_theta_angle, cam_phi_angle::Float64, bhspin, Rout::Float64)
    """
    Computes the camera position in internal coordinates based on the distance and angles.
    Parameters:
    @cam_dist: Radial distance of the camera.
    @cam_theta_angle: Polar angle of the camera in degrees.
    @cam_phi_angle: Azimuthal angle of the camera in degrees.
    """

    if(MODEL == "analytic" || MODEL == "thin_disk")

        T = promote_type(typeof(cam_dist), typeof(cam_theta_angle), typeof(cam_phi_angle), typeof(bhspin))
        return @SVector [zero(T), log(cam_dist), cam_theta_angle / 180,  (cam_phi_angle / 180) * π]

    elseif (MODEL == "iharm")

        T = promote_type(typeof(cam_dist), typeof(cam_theta_angle), typeof(cam_phi_angle), typeof(bhspin))
        x = @SVector [zero(T), T(cam_dist), T(cam_theta_angle)/T(180) * T(π), T(cam_phi_angle)/T(180) * T(π)]
        x3_val = root_find_ad_safe(x, params.cstartx, params.cstopx)
        return @SVector [
            zero(T), 
            log(cam_dist), 
            x3_val, 
            (cam_phi_angle / 180) * π
        ]
    else
        error("Unknown MODEL type: $MODEL")
    end
end

