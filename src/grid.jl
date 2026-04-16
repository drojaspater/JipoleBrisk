function Xtoijk_ghost(X)
    i_logical = trunc(Int, ((X[2] - params.startx[2]) / params.dx[2]) - 0.5 + 1000) - 1000
    j_logical = trunc(Int, ((X[3] - params.startx[3]) / params.dx[3]) - 0.5 + 1000) - 1000

    i = clamp(i_logical, 0, params.N1 - 2)
    j = clamp(j_logical, 0, params.N2 - 2)
    
    del2 = clamp((X[2] - ((i + 0.5) * params.dx[2] + params.startx[2])) / params.dx[2], 0.0, 1.0)
    del3 = clamp((X[3] - ((j + 0.5) * params.dx[3] + params.startx[3])) / params.dx[3], 0.0, 1.0)

    i += 1 
    j += 1 

    phi = rem(X[4], params.cstopx[4])
    if phi < 0.0
        phi += params.cstopx[4]
    end

    k_logical = trunc(Int, ((phi - params.startx[4]) / params.dx[4]) - 0.5 + 1000) - 1000
    del4 = clamp((phi - ((k_logical + 0.5) * params.dx[4] + params.startx[4])) / params.dx[4], 0.0, 1.0)

    k_logical_wrapped = mod(k_logical, params.N3) 
    k = k_logical_wrapped + 1

    return i, j, k, del2, del3, del4
end



function X_in_domain(X)
    if(X[2] < params.cstartx[2] || X[2] > params.cstopx[2] || X[3] < params.cstartx[3] || X[3] > params.cstopx[3])
        return 0
    end
    return 1
end

function ijktoX(i,j,k, X)
    X[2] = params.startx[2] + (i + 0.5) * params.dx[2]
    X[3] = params.startx[3] + (j + 0.5) * params.dx[3]
    X[4] = params.startx[4] + (k + 0.5) * params.dx[4]
    return
end



function interp_scalar(X, data)
    i, j, k, del2, del3, del4 = Xtoijk_ghost(X)

    (N1_data, N2_data, N3_data) = size(data) 

    ip1 = i + 1
    jp1 = j + 1

    kp1 = k + 1
    if kp1 > N3_data
        kp1 = 1  
    end

    b1 = 1.0 - del2
    b2 = 1.0 - del3

    # Interpolate in i and j (replace del[2], del[3], del[4] with local vars)
    interp = data[i, j, k]   * b1 * b2 +
             data[ip1, j, k] * del2 * b2 +
             data[i, jp1, k] * b1 * del3 +
             data[ip1, jp1, k] * del2 * del3

    interp = interp * (1.0 - del4) + (
             data[i, j, kp1]   * b1 * b2 +
             data[ip1, j, kp1] * del2 * b2 +
             data[i, jp1, kp1] * b1 * del3 +
             data[ip1, jp1, kp1] * del2 * del3
    ) * del4

    return interp
end

# function interp_scalar(X, data)
#     # del is required if Xtoijk_ghost! needs it to output i, j, k
#     # We won't use the values in 'del' for weighting, but we need the variable.
#     del = zeros(eltype(X), 4)

#     # Get the base indices (bottom-left-back corner of the voxel)
#     i, j, k = Xtoijk_ghost!(X, del)

#     (N1_data, N2_data, N3_data) = size(data) 

#     # Define the +1 indices
#     ip1 = i + 1
#     jp1 = j + 1

#     # --- k (phi) PERIODIC FIX ---
#     # Handle the periodic boundary for the 3rd dimension
#     kp1 = k + 1
#     if kp1 > N3_data
#         kp1 = 1  # Wrap around to the first cell
#     end
#     # --- END FIX ---

#     # Sum the values of the 8 surrounding corners
#     # Corner 1: (i, j, k)
#     sum_val = data[i, j, k] +
#               data[ip1, j, k] +
#               data[i, jp1, k] +
#               data[ip1, jp1, k]

#     # Corner 2: (i, j, kp1) - The periodic neighbor slice
#     sum_val += data[i, j, kp1] +
#                data[ip1, j, kp1] +
#                data[i, jp1, kp1] +
#                data[ip1, jp1, kp1]

#     # Return the average (Sum / 8)
#     return sum_val / 8.0
# end


function interp_scalar_time(X, dataA, dataB, tfac)
    vA = interp_scalar(X, dataA)
    if SLOW_LIGHT
        vB = interp_scalar(X, dataB)
        return (tfac) * vA + (1. -tfac) * vB
    end
    return vA
end



function gdet_zone(i, j, k)
    X2 = params.startx[2] + (i + 0.5) * params.dx[2]
    X3 = params.startx[3] + (j + 0.5) * params.dx[3]
    X4 = params.startx[4] + (k + 0.5) * params.dx[4]
    
    X = SVector(0.0, X2, X3, X4)

    rt = zeros(MVector{2, Float64})
    bl_coord!(rt, X)
    r = rt[1]
    th = rt[2]
    
    gcovKS = gcov_ks(r, th, params.a)
    
    dxdX = set_dxdX(X)

    gcov = zeros(MMatrix{4, 4, Float64})

    gcov = transpose(dxdX) * gcovKS * dxdX
    return gdet_func(gcov)
end
