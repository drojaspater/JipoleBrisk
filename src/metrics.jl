using LinearAlgebra

function gdet_func(gcov)
    """
    Returns the determinant of the covariant metric tensor.

    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    F = lu(gcov)
    U = F.U

    if any(abs(U[i, i]) < 1e-14 for i in 1:size(U, 1))
        @warn "Singular matrix in gdet_func!"
        return -1.0
    end

    gdet = prod(diag(U))
    return sqrt(abs(gdet))
end



function gcov_func!(X::MVec4, bhspin::Float64, gcov, R0::Float64 = 0.0)
    """
    Returns g_{munu} at location specified by X.
    Adapted from ipole C code logic.
    """
    
    # Get Boyer-Lindquist coordinates (r, theta)
    # Assumes bl_coord is defined elsewhere
    r, th = bl_coord(X) 
    
    # Initialize metric to zero
    fill!(gcov, 0.0)

    
    #Minkowski (Spherical Polar)
    if params.metric == METRIC_MINKOWSKI
        gcov[1, 1] = -1.0
        gcov[2, 2] = 1.0
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return

    # E-Minkowski (Exponential Radial, Spherical Polar)
    elseif params.metric == METRIC_EMINKOWSKI
        gcov[1, 1] = -1.0
        gcov[2, 2] = r * r 
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return

    # FMKS (Funky Modified Kerr-Schild)
    elseif params.metric == METRIC_FMKS
        sth = sin(th)
        cth = cos(th)
        s2 = sth^2
        rho2 = r^2 + bhspin^2 * cth^2
    end

    #MKS 
    Gcov_ks = @MMatrix zeros(4, 4)
    
    cth = cos(th)
    sth = sin(th)
    s2 = sth^2
    rho2 = r^2 + bhspin^2 * cth^2
    
    Gcov_ks[1, 1] = -1.0 + 2.0 * r / rho2
    Gcov_ks[1, 2] = 2.0 * r / rho2
    Gcov_ks[1, 4] = -2.0 * bhspin * r * s2 / rho2
    
    Gcov_ks[2, 1] = Gcov_ks[1, 2]
    Gcov_ks[2, 2] = 1.0 + 2.0 * r / rho2
    Gcov_ks[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2)
    
    Gcov_ks[3, 3] = rho2
    
    Gcov_ks[4, 1] = Gcov_ks[1, 4]
    Gcov_ks[4, 2] = Gcov_ks[2, 4]
    Gcov_ks[4, 4] = s2 * (rho2 + bhspin^2 * s2 * (1.0 + 2.0 * r / rho2))


    dxdX = set_dxdX(X)
    
    fill!(gcov, 0.0)
    for mu in 1:4
        for nu in 1:4
            sum_val = 0.0
            for lam in 1:4
                for kap in 1:4
                    sum_val += Gcov_ks[lam, kap] * dxdX[lam, mu] * dxdX[kap, nu]
                end
            end
            gcov[mu, nu] = sum_val
        end
    end
end


function gcov_func(X::MVec4, bhspin::Float64, R0::Float64 = 0.0)
    """
    Returns g_{munu} at location specified by X.
    Adapted from ipole C code logic.
    """
    
    # Get Boyer-Lindquist coordinates (r, theta)
    # Assumes bl_coord is defined elsewhere
    r, th = bl_coord(X) 
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    gcov = @MMatrix zeros(T, 4, 4)
    
    # Initialize metric to zero
    fill!(gcov, 0.0)

    
    #Minkowski (Spherical Polar)
    if params.metric == METRIC_MINKOWSKI
        gcov[1, 1] = -1.0
        gcov[2, 2] = 1.0
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return gcov

    # E-Minkowski (Exponential Radial, Spherical Polar)
    elseif params.metric == METRIC_EMINKOWSKI
        gcov[1, 1] = -1.0
        gcov[2, 2] = r * r 
        gcov[3, 3] = r * r
        gcov[4, 4] = r * r * sin(th)^2
        return gcov

    end

    #MKS 
    Gcov_ks = @MMatrix zeros(4, 4)
    
    cth = cos(th)
    sth = sin(th)
    s2 = sth^2
    rho2 = r^2 + bhspin^2 * cth^2
    
    Gcov_ks[1, 1] = -1.0 + 2.0 * r / rho2
    Gcov_ks[1, 2] = 2.0 * r / rho2
    Gcov_ks[1, 4] = -2.0 * bhspin * r * s2 / rho2
    
    Gcov_ks[2, 1] = Gcov_ks[1, 2]
    Gcov_ks[2, 2] = 1.0 + 2.0 * r / rho2
    Gcov_ks[2, 4] = -bhspin * s2 * (1.0 + 2.0 * r / rho2)
    
    Gcov_ks[3, 3] = rho2
    
    Gcov_ks[4, 1] = Gcov_ks[1, 4]
    Gcov_ks[4, 2] = Gcov_ks[2, 4]
    Gcov_ks[4, 4] = s2 * (rho2 + bhspin^2 * s2 * (1.0 + 2.0 * r / rho2))


    dxdX = set_dxdX(X)
    
    fill!(gcov, 0.0)
    for mu in 1:4
        for nu in 1:4
            sum_val = 0.0
            for lam in 1:4
                for kap in 1:4
                    sum_val += Gcov_ks[lam, kap] * dxdX[lam, mu] * dxdX[kap, nu]
                end
            end
            gcov[mu, nu] = sum_val
        end
    end
    return gcov
end



function gcov_func_fd(X, bhspin, R0::Float64 = 0.0)
    """
    Returns covariant metric tensor in Kerr-Schild coordinates.

    Parameters:
    @X: Vector of position coordinates in internal coordinates.
    """
    r, th = bl_coord(X)
    T = promote_type(typeof(r), typeof(th), typeof(bhspin))
    gcov = @MMatrix zeros(T, 4, 4)
    Gcov_ks = @MMatrix zeros(T, 4, 4)
    gcov_ks(r, th, bhspin, Gcov_ks)



    dxdX = set_dxdX(X)
    for μ in 1:NDIM
        for ν in 1:NDIM
            for λ in 1:NDIM
                for κ in 1:NDIM
                    gcov[μ, ν] +=  Gcov_ks[λ, κ] * dxdX[λ, μ] * dxdX[κ, ν] 
                end
            end
        end
    end
    return gcov
end

function gcon_func!(gcov, gcon)
    """
    Returns contravariant metric tensor in Kerr-Schild coordinates through matrix inversion of the covariant tensor.
    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    gcon .= inv(gcov)
    if any(isnan.(gcon)) || any(isinf.(gcon))
        @error "Singular gcov encountered in gcon"
        print_matrix("gcov", gcov)
        print_matrix("gcon", gcon)
        error("Singular gcov encountered, cannot compute gcon.")
    end
end

function gcon_func(gcov)
    """
    Returns contravariant metric tensor in Kerr-Schild coordinates through matrix inversion of the covariant tensor.
    Parameters:
    @gcov: Covariant metric tensor in Kerr-Schild coordinates.
    """
    gcon = inv(gcov)
    if any(isnan.(gcon)) || any(isinf.(gcon))
        @error "Singular gcov encountered in gcon"
        print_matrix("gcov", gcov)
        print_matrix("gcon", gcon)
        error("Singular gcov encountered, cannot compute gcon.")
    end
    return gcon
end

function gcov_bl!(r,th, bhspin, gcov)
    """
    Computes the metric tensor in Boyer-Lindquist coordinates.
    Parameters:
    @r: Radial coordinate in Boyer-Lindquist coordinates.
    @th: Angular coordinate in Boyer-Lindquist coordinates.
    """

    sth = sin(th)
    if(sth < 1e-40)
        sth = 10^(-40)
    end
    cth = cos(th)
    s2 = sth * sth
    if(r < 1e-40)
        r = 10^(-40)
    end
    a2 = bhspin * bhspin
    r2 = r * r
    DD = (1.0 - 2.0 / r + a2 / r2)
    mu = 1.0 + a2 * cth * cth / r2

    gcov[1, 1] = -(1.0 - 2.0 / (r * mu))
    gcov[1, 4] = -2.0 * bhspin * s2 / (r * mu)
    gcov[4, 1] = gcov[1, 4]
    gcov[2, 2] = mu / (DD )
    gcov[3, 3] = r2 * mu
    gcov[4, 4] = r2 * sth * sth * (1.0 + a2 / r2 + 2.0 * a2 * s2 / (r2 * r * mu))

    #if any element of the diagonal is zero print variables
    if(gcov[1,1] == 0 || gcov[2,2] == 0 || gcov[3,3] == 0 || gcov[4,4] == 0)   
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $bhspin, r2 $r2, a2 $a2, mu $mu, DD $DD")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
    #if any (isnan.(gcov)) || any(isinf.(gcov))
    if any(isnan.(gcov)) || any(isinf.(gcov))
        @error "Singular gcov encountered in gcov_bl"
        println("sth $sth, cth $cth, r $r, a $bhspin, r2 $r2, a2 $a2, mu $mu, DD $DD")
        print_matrix("gcov", gcov)
        error("Singular gcov encountered, cannot compute gcov_bl.")
    end
end