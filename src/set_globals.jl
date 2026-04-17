println("Using model: $MODEL, change src/set_globals.jl to modify.")
# Mass of the black hole in solar masses
const L_unit = GNEWT * MBH * MSUN / CL^2  # Length unit in gravitational radius (Rg)
const T_unit = L_unit / CL  # Time unit in seconds
#const RHO_unit = M_unit / L_unit^3  # Density unit in g/cm^3