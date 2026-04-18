using HDF5
startX::MVec4 = MVec4(undef)
stopX::MVec4 = MVec4(undef)

@inline function get_model_fourv(X::MVec4)
    """
    Calculates Ucon, Ucov, Bcon and Bcov from primitives at location X using interpolation.

    Parameters:
    @X: The point in spacetime in Native coordinates
    """
    gcov::MMat4 = gcov_func(X)
    gcon::MMat4 = gcon_func(gcov)

    Ucon = MVec4(undef)
    Ucov = MVec4(undef)

    #If we're outside of the logical domain, default to
    #normal observer velocity for Ucon/Ucov and default
    #Bcon/Bcov to zero.
    if (X_in_domain(X))
        Ucov[1] = -1. /sqrt(-gcon[1, 1])
        Ucov[2] = 0.0
        Ucov[3] = 0.0
        Ucov[4] = 0.0


        for μ in 1:NDIM
            Ucon[1] += Ucov[μ] * gcon[0, μ]
            Ucon[2] += Ucov[μ] * gcon[1, μ]
            Ucon[3] += Ucov[μ] * gcon[2, μ]
            Ucon[4] += Ucov[μ] * gcon[3, μ]
            Bcon[μ] = 0.0
            Bcov[μ] = 0.0
        end
    end
    Vcon = MVec4(undef)

    #Currently not in use, it should be accounted when considering SLOW_LIGHT
    nA, nB, tfac = set_tinterp_ns(X)

    Vcon[2] = interp_scalar_time()

end

function X_in_domain(X::MVec4)
    """
    Check if the point X is within the domain of the model.
    """
    
    if(X[2] < startx[2]|| X[2] >= stopx[2] || X[3] < startx[3] || X[3] >= stopx[3])
        return false
    end

    return true
end


function set_tinterp_ns(X::MVec4)
    """
    How far have we interpolated in time between two data points data[nA] and data[nB]

    Parameters:
    @X: The point in spacetime where we want to interpolate.

    Observations:
    - In slowlight mode, we perform linear interpolation in time. This function tells
    us how far we've progressed from data[nA]->t to data[nB]->t but "in reverse" as
    tinterp == 1 -> we're exactly on nA and tinterp == 0 -> we're exactly on nB. 
    
    - Currently, this function is just a placeholder, it must be implemented if SLOW_LIGHT is true.
    """

    return 0.0, 0, 0
end

function interp_scalar_time(X::MVec4, tfac::Float64)
    """
    Interpolates all four dims of scalar/vector fields

    Parameters:
    @X: The point in spacetime where we want to interpolate.
    @tfac: The time factor
    """
    vA = interp_scalar(X)

    #This is SLOWLIGHT specific, it should be implemented
    if(false)
        vB = interp_scalar(X)
        return tfac * vA + (1. - tfac) * vB
    end

    return vA
end

function interp_scalar(X::MVec4)
    """
    Interpolates the spatial dims at point X.
    
    Parameters
    @X: The point in spacetime in native coordinates where we want to interpolate.
    """

    del::MVec4 = MVec4(undef)
    b1::Float64, b2::Float64, interp::Float64 = 0.0, 0.0, 0.0

    i,j,k, del = Xtoijk(X)

    ip1 = i + 1
    jp1 = j + 1
    kp1 = k + 1

    b1 = 1 - del[1]
    b2 = 1 - del[2]

    error("This function is not implemented yet!")
end


using HDF5

function get_parfile(fname::String)
    """
    Reads the parameter file and returns the parameters as a dictionary.
    
    Parameters:
    @fname: The name of the parameter file.
    
    Returns:
    A dictionary containing the parameters.
    """

    attr_data::String = ""

    Status::Bool = true

    #Opening the HDF5 file with read-only access
    file_id = HDF5.h5f_open(fname, HDF5.H5F_ACC_RDONLY, HDF5.H5P_DEFAULT)
    # Check if the file has opened successfully
    if file_id <0
        HDF5.h5f_close(file_id)
        error("Failed to open the HDF5 file: $fname")
    end


    group_id = HDF5.h5g_open(file_id, "Input", HDF5.H5P_DEFAULT)
    if group_id < 0
        HDF5.H5Gclose(group_id)
        HDF5.h5f_close(file_id)
        error("Failed to open group 'Input' in the HDF5 file: $fname")
    end


    #Get the attribute
    attr_id = HDF5.h5a_open(group_id, "File", HDF5.H5P_DEFAULT)
    if attr_id < 0
        HDF5.h5a_close(attr_id)
        HDF5.h5g_close(group_id)
        HDF5.h5f_close(file_id)
        error("Failed to open attribute 'File' in the 'Input' group of the HDF5 file: $fname")
    end

    #Get the attribute data type
    attr_type = HDF5.h5a_get_type(attr_id)
    if attr_type < 0
        HDF5.h5a_close(attr_id)
        HDF5.h5g_close(group_id)
        HDF5.h5f_close(file_id)
        error("Failed to get the type of 'File' attribute in the 'Input' group of the HDF5 file: $fname")
    end

    # # Find the input group
    # group = file["Input"]
    # if group === nothing
    #     error("Failed to find 'Input' group in the HDF5 file: $fname")
    # end

    # # Find the file attributes
    # attr = H5Aopen(group_id, "File", H5P_DEFAULT);
    # if attr === nothing
    #     error("Failed to find 'File' attribute in the 'Input' group of the HDF5 file: $fname")
    # end

    # #Get attribute data type
    # attr_type = H5Aget_type(attr)
    # if attr_type < 0
    #     error("Failed to get the type of 'File' attribute in the 'Input' group of the HDF5 file: $fname")
    # end

    # # Check if attribute is a string
    # if(H5Tis_variable_string(attr))
    #     vldata::String = null
    #     status = H5Aread(attr, vldata)
    # end

end