
using Statistics
using ForwardDiff: Dual


# A single wrapper that uses a type parameter (Var) to know which variable to differentiate
struct TransferWrapper{Var, TI, TX, TK, TXf, TKf, Tdl, Tf, Ta, Td}
    I::TI
    Xi::TX
    Ki::TK
    Xf::TXf
    Kf::TKf
    dl::Tdl
    freq::Tf
    bhspin::Ta
    data::Td
end

# Custom constructor so we don't have to write out the massive type signature manually
function TransferWrapper{Var}(I::TI, Xi::TX, Ki::TK, Xf::TXf, Kf::TKf, dl::Tdl, freq::Tf, bhspin::Ta, data::Td) where {Var, TI, TX, TK, TXf, TKf, Tdl, Tf, Ta, Td}
    return TransferWrapper{Var, TI, TX, TK, TXf, TKf, Tdl, Tf, Ta, Td}(I, Xi, Ki, Xf, Kf, dl, freq, bhspin, data)
end

# Define the forward calls for each variable (0=I, 1=Xi, 2=Ki, 3=Xf, 4=Kf)
(w::TransferWrapper{0})(I_new::Real) = transfer_step(I_new, w.Xi, w.Ki, w.Xf, w.Kf, w.dl, w.freq, w.bhspin, w.data)
(w::TransferWrapper{1})(Xi_new) = transfer_step(w.I, Xi_new, w.Ki, w.Xf, w.Kf, w.dl, w.freq, w.bhspin, w.data)
(w::TransferWrapper{2})(Ki_new) = transfer_step(w.I, w.Xi, Ki_new, w.Xf, w.Kf, w.dl, w.freq, w.bhspin, w.data)
(w::TransferWrapper{3})(Xf_new) = transfer_step(w.I, w.Xi, w.Ki, Xf_new, w.Kf, w.dl, w.freq, w.bhspin, w.data)
(w::TransferWrapper{4})(Kf_new) = transfer_step(w.I, w.Xi, w.Ki, w.Xf, Kf_new, w.dl, w.freq, w.bhspin, w.data)

# We also need one for the approximate_solve closure
struct ApproxSolveWrapper{TI, Tdl}
    I::TI
    dl::Tdl
end
(w::ApproxSolveWrapper)(v) = approximate_solve(w.I, v[1], v[2], v[3], v[4], w.dl)

struct RadTransferX{T1,T2,T3,T4,T5}
    Kconi::T1
    freq::T2
    Intensity::T3
    bhspin::T4
    data::T5
end
(f::RadTransferX)(x) = RadTransferDiff(x, f.Kconi, f.freq, f.Intensity, f.bhspin, f.data)

struct RadTransferK{T1,T2,T3,T4,T5}  
    Xi::T1
    freq::T2
    Intensity::T3
    bhspin::T4
    data::T5
end
(f::RadTransferK)(k) = RadTransferDiff(f.Xi, k, f.freq, f.Intensity, f.bhspin, f.data)

struct RadTransferA{T1,T2,T3,T4,T5}
    Xi::T1
    Kconi::T2  
    freq::T3
    Intensity::T4
    data::T5
end
(f::RadTransferA)(spin) = RadTransferDiff(f.Xi, f.Kconi, f.freq, f.Intensity, spin, f.data)

struct RadTransferI{T1,T2,T3,T4,T5}
    Xi::T1
    Kconi::T2
    freq::T3  
    bhspin::T4
    data::T5
end
(f::RadTransferI)(intens) = RadTransferDiff(f.Xi, f.Kconi, f.freq, intens, f.bhspin, f.data)


function Mom4ODE(X::AbstractVector, Kcon::AbstractVector, bhspin)
    T = eltype(Kcon)
    
    # Catch the returned SArray (notice we call the function without the "!")
    if(MODEL == "analytic" || MODEL == "thin_disk")
        lconn = get_connection_analytic(X, bhspin)
    elseif(MODEL == "iharm")
        lconn = get_connection_analytic(X, bhspin)
    else
        error("Unknown model: $MODEL")
    end
    
    result = zero(MVector{4, T})
    
    @inbounds for mu in 1:4
        for alpha in 1:4
            for beta in 1:4
                result[mu] -= lconn[mu, alpha, beta] * Kcon[alpha] * Kcon[beta]
            end
        end
    end
    return SVector(result)
end

function systemODEs_flat(XK)
    # Unpack explicitly. This is lightning fast and 0 allocations for SVectors.
    X = SVector{4}(XK[1], XK[2], XK[3], XK[4])
    K = SVector{4}(XK[5], XK[6], XK[7], XK[8])
    spin = XK[9]
    return Mom4ODE(X, K, spin)
end

function CalculateK(ro, θo, phi, i,j, nx, ny, fovx, fovy, bhspin, freq, Rout)
    Xcam = camera_position(ro, θo, phi, bhspin, Rout)
    T = eltype(Xcam)
    Kcon = MVector{4, T}(undef)
    X = MVector{4, T}(undef)
    init_XK!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin)
    return SVector(Kcon) * (freq * HPL / (ME * CL * CL))
end


function RadTransferDiff(Xi, Kconi, freq, Ii, bhspin, data)
    ji, ki = get_jk(Xi, Kconi, freq, bhspin, data, Val(false))
    return ji - ki * Ii
end


#I'm sure in this function I could use the same wrapper as used in the GRMHD function, but this is working relatively fast and I'm lazy
#and we're not gonna be using analytic models in the calculation.
function AutoDiffGeoTrajEulerMethod!(traj, dI_dθo_out::Base.RefValue{Float64}, intensity_out::Base.RefValue{Float64}, dI_da_out::Base.RefValue{Float64},ro::Float64, θo::Float64, phi::Float64, bhspin::Float64, nx::Int64, ny::Int64, nmaxstep::Int64,i::Int64,j::Int64,freq::Float64, fovx::Float64, fovy::Float64, Rout::Float64, Rstop::Float64, data = nothing)
    """
    Returns the intensity and the derivative of the intensity with respect to θo for pixel (i,j) using autodiff.
    """
    #First set up the initial position and momentum of the specific pixel (i,j)
    Xcam = MVec4(camera_position(ro, θo, phi, bhspin, Rout))
    Kcon = MVec4(undef)
    X = MVec4(undef)
    Rh = 1 + sqrt(1. - bhspin * bhspin);  # Radius of the horizon

    #Define X and Kcon
    init_XK!(X, Kcon, i,j, Xcam, nx, ny, fovx, fovy, bhspin)
    #Put Kcon in correct unitless
    Kcon .*= freq * HPL / (ME * CL * CL) 
    dl_unit::Float64 = L_unit * HPL / (ME * CL^2)  # Unit conversion factor for dl

    # half steps, used for polarization
    # Set the variables before (i) and after (f) the first step
    Xhalf = copy(X)
    Kconhalf = copy(Kcon)
    lconn = Tensor3D(undef)

    #Calculate the derivative of the initial positions and momentum with respect to θo
    #The derivative of K is calculated using finite differences
    #Define reference for the intensity integration part so that I don't have to reallocate each step
    jac = MMatrix{4, 9, Float64}(undef)
    if(MODEL == "analytic" || MODEL == "thin_disk")
        dX_dθo = ForwardDiff.derivative(x -> camera_position(ro, x, phi, bhspin, Rout), θo)
        dK_dθo = ForwardDiff.derivative(x -> CalculateK(ro, x, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, Rout), θo)
    end

    dX_da = ForwardDiff.derivative(x -> camera_position(ro, θo, phi, x, Rout), bhspin)
    dK_da = ForwardDiff.derivative(x -> CalculateK(ro, θo, phi, i, j, nx, ny, fovx, fovy, x, freq, Rout), bhspin)
    
    XK = MVector{9, Float64}(undef)
    XK[9] = bhspin

    push!(traj, OfTraj(
        0.0,
        SVector{4, Float64}(X),
        SVector{4, Float64}(Kcon),
        SVector{4, Float64}(Xhalf),
        SVector{4, Float64}(Kconhalf),
        SVector{4, Float64}(dX_dθo),
        SVector{4, Float64}(dK_dθo),
        SVector{4, Float64}(dX_da),
        SVector{4, Float64}(dK_da)
    ))

    step::Int64 = 1


    temp_dX_dθo = MVec4(undef)
    temp_dX_da = MVec4(undef)
    temp_dK_dθo = MVec4(undef) 
    temp_dK_da = MVec4(undef)

    temp_jac_dX_dθo = MVec4(undef)
    temp_jac_dK_dθo = MVec4(undef)
    temp_jac_dX_da = MVec4(undef)
    temp_jac_dK_da = MVec4(undef)
    cstartx = MVec4(0.0, log(Rh), 0.0, 0.0)
    cstopx = MVec4(0.0, log(Rout), 1.0, 2.0 * π)
    while (stop_backward_integration(X, Kcon, Rh, Rstop) == 0 && (step <= nmaxstep))
        @inbounds begin
            @inbounds for k = 1:4
                XK[k] = X[k]
                XK[k+4] = Kcon[k]
            end

            #Calculate the Jacobian of the system of ODEs with respect to the X^μ, K^μ and spin
            # jac is a 4×9 matrix:
            # Rows: Output variables (dK₁/dλ, dK₂/dλ, dK₃/dλ, dK₄/dλ)
            # Columns: Input variables (1:4 = X₁, X₂, X₃, X₄; 5:8 = K₁, K₂, K₃, K₄; 9 = bhspin)
            # Entry (i, j): ∂(ODE_i)/∂(var_j)
            # Table structure:
            #         | ∂(dK₁/dλ)/∂X₁ ... ∂(dK₁/dλ)/∂X₄ ∂(dK₁/dλ)/∂K₁ ... ∂(dK₁/dλ)/∂K₄ ∂(dK₁/dλ)/∂a |
            #         | ∂(dK₂/dλ)/∂X₁ ... ∂(dK₂/dλ)/∂X₄ ∂(dK₂/dλ)/∂K₁ ... ∂(dK₂/dλ)/∂K₄ ∂(dK₂/dλ)/∂a |
            #         | ∂(dK₃/dλ)/∂X₁ ... ∂(dK₃/dλ)/∂X₄ ∂(dK₃/dλ)/∂K₁ ... ∂(dK₃/dλ)/∂K₄ ∂(dK₃/dλ)/∂a |
            #         | ∂(dK₄/dλ)/∂X₁ ... ∂(dK₄/dλ)/∂X₄ ∂(dK₄/dλ)/∂K₁ ... ∂(dK₄/dλ)/∂K₄ ∂(dK₄/dλ)/∂a |
            ForwardDiff.jacobian!(jac, systemODEs_flat, XK)

            dl = stepsize(X, Kcon, cstartx, cstopx)
            scaled_dl = dl * dl_unit

            @. temp_dX_dθo = traj[step].dX_dθo - dl * traj[step].dK_dθo
            @. temp_dX_da = traj[step].dX_da - dl * traj[step].dK_da

            mul!(temp_jac_dX_dθo, view(jac, 1:4, 1:4), traj[step].dX_dθo)
            mul!(temp_jac_dK_dθo, view(jac, 1:4, 5:8), traj[step].dK_dθo)
            @. temp_dK_dθo = traj[step].dK_dθo - dl * (temp_jac_dX_dθo + temp_jac_dK_dθo)

            mul!(temp_jac_dX_da, view(jac, 1:4, 1:4), traj[step].dX_da)  
            mul!(temp_jac_dK_da, view(jac, 1:4, 5:8), traj[step].dK_da)
            # Handle the jacobian column separately to avoid broadcasting issues
            @. temp_dK_da = traj[step].dK_da - dl * (temp_jac_dX_da + temp_jac_dK_da)

            for k in 1:4
                temp_dK_da[k] = temp_dK_da[k] - dl * jac[k, 9]
            end

            push_photon!(X, Kcon, -dl,Xhalf, Kconhalf, lconn, bhspin)

            step += 1
            push!(traj, OfTraj(
                scaled_dl,
                SVector{4, Float64}(X),   
                SVector{4, Float64}(Kcon),   
                SVector{4, Float64}(Xhalf),   
                SVector{4, Float64}(Kconhalf),
                SVector{4, Float64}(temp_dX_dθo),
                SVector{4, Float64}(temp_dK_dθo),
                SVector{4, Float64}(temp_dX_da),
                SVector{4, Float64}(temp_dK_da)
            ))
        end
    end

    if (step > nmaxstep)
        @error("AutoDiffGeoTrajEulerMethod: Maximum number of steps reached without meeting geodesics stop condition.")
        error()
    end

    Intensity = 0.0
    dI_dθo = 0.0
    dI_da = 0.0
    jac_I_X = MVec4(undef)
    jac_I_K = MVec4(undef)

    Xi_S = traj[step].X
    Kconi_S = traj[step].Kcon

    ji, ki = get_jk(Xi_S, Kconi_S, freq, bhspin, data, Val(false))

    for nstep = step:-1:2
        Xi_S = traj[nstep].X
        Xf_S = traj[nstep - 1].X
        Kconi_S = traj[nstep].Kcon
        Kconf_S = traj[nstep - 1].Kcon

        if(MODEL == "thin_disk")
            if(thindisk_region(Xi_S, Xf_S))
                Intensity = GetTDBoundaryCondition(Xi_S, Kconi_S, bhspin, Rh)
            end
            continue
        end

        if !radiating_region(Xf_S, Rh)
            continue
        end

        rad_x = RadTransferX(Kconi_S, freq, Intensity, bhspin, data)
        rad_k = RadTransferK(Xi_S, freq, Intensity, bhspin, data)
        rad_a = RadTransferA(Xi_S, Kconi_S, freq, Intensity, data)
        rad_i = RadTransferI(Xi_S, Kconi_S, freq, bhspin, data)

        ForwardDiff.gradient!(jac_I_X, rad_x, Xi_S)
        ForwardDiff.gradient!(jac_I_K, rad_k, Kconi_S)
        jac_I_A = ForwardDiff.derivative(rad_a, bhspin)
        jac_I_I = ForwardDiff.derivative(rad_i, Intensity)
        

        dI_dθo = dI_dθo + (traj[nstep].dl) * (dot(jac_I_X, traj[nstep].dX_dθo) + dot(jac_I_K, traj[nstep].dK_dθo) + jac_I_I * dI_dθo)
        dI_da = dI_da + (traj[nstep].dl) * (dot(jac_I_X, traj[nstep].dX_da) + dot(jac_I_K, traj[nstep].dK_da) + jac_I_I * dI_da + jac_I_A)

        jf, kf = get_jk(Xf_S, Kconf_S, freq, bhspin, data, Val(false))
        Intensity = approximate_solve(Intensity, ji, ki, jf, kf, traj[nstep - 1].dl)
        if (isnan(Intensity) || isinf(Intensity))
            @error "NaN or Inf encountered in intensity calculation at pixel ($i, $j)"
            println("Intensity = $Intensity, ji = $ji, ki = $ki, jf = $jf, kf = $kf")
            print_vector("Kconf =", Kconf_S)
            print_vector("Kconi =", Kconi_S)
            error("NaN or Inf encountered in intensity calculation")
        end
        
        ji = jf
        ki = kf
    end

    dI_dθo_out[] = dI_dθo * freq^3
    intensity_out[] = Intensity * freq^3
    dI_da_out[] = dI_da * freq^3
    empty!(traj)
    return nothing
end


function transfer_step(I_prev, X_curr, K_curr, X_next, K_next, dl, freq, bhspin, data)
    ji, ki, _, _ = get_jk(X_curr, K_curr, freq, bhspin, data, Val(false))
    jf, kf, _, _ = get_jk(X_next, K_next, freq, bhspin, data, Val(false))

    return approximate_solve(I_prev, ji, ki, jf, kf, dl)
end

function AutoDiffGeoTrajEulerMethod_GRMHD!(traj, dI_dθo_out::Base.RefValue{Float64}, intensity_out::Base.RefValue{Float64}, dI_dRhigh_out::Base.RefValue{Float64},ro::Float64, θo::Float64, phi::Float64, bhspin::Float64, nx::Int64, ny::Int64, nmaxstep::Int64,i::Int64,j::Int64,freq::Float64, fovx::Float64, fovy::Float64, Rout::Float64, Rstop::Float64, data::T_data = nothing) where {T_data}
    """
    Returns the intensity and the derivative of the intensity with respect to θo for pixel (i,j) using autodiff.
    """
    # Emptying it in case of error from previous calls
    empty!(traj)

    Xcam = MVec4(camera_position(ro, θo, phi, bhspin, Rout))

    Kcon = MVec4(undef)
    X = MVec4(undef)
    Rh = 1 + sqrt(1. - bhspin * bhspin)  # Radius of the horizon

    init_XK!(X, Kcon, i, j, Xcam, nx, ny, fovx, fovy, bhspin)
    Kcon .*= freq * HPL / (ME * CL * CL) 
    dl_unit::Float64 = L_unit * HPL / (ME * CL^2)  

    Xhalf = copy(X)
    Kconhalf = copy(Kcon)
    lconn = Tensor3D(undef)

    jac = MMatrix{4, 9, Float64}(undef)
    dX_dθo = ForwardDiff.derivative(x -> camera_position(ro, x, phi, bhspin, Rout), θo)
    dK_dθo = ForwardDiff.derivative(x -> CalculateK(ro, x, phi, i, j, nx, ny, fovx, fovy, bhspin, freq, Rout), θo)
    XK = MVector{9, Float64}(undef)
    XK[9] = bhspin
    
    push!(traj, OfTraj(
        0.0,
        SVector{4, Float64}(X),
        SVector{4, Float64}(Kcon),
        SVector{4, Float64}(Xhalf),
        SVector{4, Float64}(Kconhalf),
        SVector{4, Float64}(dX_dθo),
        SVector{4, Float64}(dK_dθo),
        zero(SVector{4, Float64}),
        zero(SVector{4, Float64})
    ))

    step::Int64 = 1

    temp_dX_dθo = MVec4(undef)
    temp_dK_dθo = MVec4(undef) 
    temp_jac_dX_dθo = MVec4(undef)
    temp_jac_dK_dθo = MVec4(undef)

    while (stop_backward_integration(X, Kcon, Rh, Rstop) == 0 && (step <= nmaxstep))
        @inbounds begin
            @inbounds for k = 1:4
                XK[k] = X[k]
                XK[k+4] = Kcon[k]
            end

            jac_static = ForwardDiff.jacobian(systemODEs_flat, SVector(XK))
            jac .= jac_static

            dl = stepsize(X, Kcon, params.cstartx, params.cstopx)

            scaled_dl = dl * dl_unit
            
            @. temp_dX_dθo = traj[step].dX_dθo - dl * traj[step].dK_dθo

            mul!(temp_jac_dX_dθo, view(jac, 1:4, 1:4), traj[step].dX_dθo)
            mul!(temp_jac_dK_dθo, view(jac, 1:4, 5:8), traj[step].dK_dθo)
            @. temp_dK_dθo = traj[step].dK_dθo - dl * (temp_jac_dX_dθo + temp_jac_dK_dθo)

            push_photon!(X, Kcon, -dl, Xhalf, Kconhalf, lconn, bhspin)

            step += 1
            push!(traj, OfTraj(
                scaled_dl,
                SVector{4, Float64}(X),   
                SVector{4, Float64}(Kcon),   
                SVector{4, Float64}(Xhalf),   
                SVector{4, Float64}(Kconhalf),
                SVector{4, Float64}(temp_dX_dθo),
                SVector{4, Float64}(temp_dK_dθo),
                zero(SVector{4, Float64}),
                zero(SVector{4, Float64})
            ))
        end
    end

    if (step > nmaxstep)
        @error("AutoDiffGeoTrajEulerMethod: Maximum number of steps reached without meeting geodesics stop condition.")
        error()
    end
    

    Intensity = 0.0
    dI_dθo = 0.0
    dI_dRhigh = 0.0

    Xi_S = traj[step].X
    Kconi_S = traj[step].Kcon

    ji, ki, dji_dRhigh, dki_dRhigh = get_jk(Xi_S, Kconi_S, freq, bhspin, data, Val(true))

    for nstep = step:-1:2
        Xi_S = traj[nstep].X
        Xf_S = traj[nstep - 1].X
        Kconi_S = traj[nstep].Kcon
        Kconf_S = traj[nstep - 1].Kcon

        if(MODEL == "thin_disk")
            if(thindisk_region(Xi_S, Xf_S))
                Intensity = GetTDBoundaryCondition(Xi_S, Kconi_S, bhspin, Rh)
            end
            continue
        end

        if !radiating_region(Xf_S, Rh)
            continue
        end
        
        dl_step = traj[nstep - 1].dl
        local_I = Intensity
        
        tw0 = TransferWrapper{0}(local_I, Xi_S, Kconi_S, Xf_S, Kconf_S, dl_step, freq, bhspin, data)
        tw1 = TransferWrapper{1}(local_I, Xi_S, Kconi_S, Xf_S, Kconf_S, dl_step, freq, bhspin, data)
        tw2 = TransferWrapper{2}(local_I, Xi_S, Kconi_S, Xf_S, Kconf_S, dl_step, freq, bhspin, data)
        tw3 = TransferWrapper{3}(local_I, Xi_S, Kconi_S, Xf_S, Kconf_S, dl_step, freq, bhspin, data)
        tw4 = TransferWrapper{4}(local_I, Xi_S, Kconi_S, Xf_S, Kconf_S, dl_step, freq, bhspin, data)

        jac_I_I  = ForwardDiff.derivative(tw0, local_I)
        jac_I_Xi = ForwardDiff.gradient(tw1, Xi_S)
        jac_I_Ki = ForwardDiff.gradient(tw2, Kconi_S)
        jac_I_Xf = ForwardDiff.gradient(tw3, Xf_S)
        jac_I_Kf = ForwardDiff.gradient(tw4, Kconf_S)

        term_geom_i = dot(jac_I_Xi, traj[nstep].dX_dθo) + dot(jac_I_Ki, traj[nstep].dK_dθo)
        term_geom_f = dot(jac_I_Xf, traj[nstep - 1].dX_dθo) + dot(jac_I_Kf, traj[nstep - 1].dK_dθo)
        
        dI_dθo = (jac_I_I * dI_dθo) + term_geom_i + term_geom_f
        jf, kf, djf_dRhigh, dkf_dRhigh = get_jk(Xf_S, Kconf_S, freq, bhspin, data, Val(true))

        internal_wrapper = ApproxSolveWrapper(local_I, dl_step)
        internal_grads = ForwardDiff.gradient(internal_wrapper, SVector(ji, ki, jf, kf))
        
        dI_dji_solve, dI_dki_solve, dI_djf_solve, dI_dkf_solve = internal_grads
        dI_dRhigh = (jac_I_I * dI_dRhigh) + (dI_dji_solve * dji_dRhigh) + (dI_dki_solve * dki_dRhigh) + (dI_djf_solve * djf_dRhigh) + (dI_dkf_solve * dkf_dRhigh)
        
        Intensity = approximate_solve(Intensity, ji, ki, jf, kf, dl_step)
        
        if (isnan(Intensity) || isinf(Intensity))
            @error "NaN or Inf encountered in intensity calculation at pixel ($i, $j)"
            println("Intensity = $Intensity, ji = $ji, ki = $ki, jf = $jf, kf = $kf")
            print_vector("Kconf =", Kconf_S)
            print_vector("Kconi =", Kconi_S)
            error("NaN or Inf encountered in intensity calculation")
        end
        
        ji = jf
        ki = kf
        dji_dRhigh = djf_dRhigh
        dki_dRhigh = dkf_dRhigh
    end

    dI_dθo_out[] = dI_dθo * freq^3
    dI_dRhigh_out[] = dI_dRhigh * freq^3
    intensity_out[] = Intensity * freq^3
    empty!(traj)
    return nothing
end

