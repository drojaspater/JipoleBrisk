using SpecialFunctions

function get_nu_c(B)
    return EE * B/(2 * π * ME * CL)
end

function I_I(x)
    return 2.5651 * (1 + 1.92 * x^(-1. / 3.) +
        0.9977 * x^(-2. / 3.)) * exp(-1.8899 * x^(1. /
          3.));
end

function maxwell_juettner_dexter_I(Ne, ν, θe, B, θ)
    nus = 3.0 * EE * B * sin(θ) / 4.0 / π / ME / CL * θe * θe + 1.0
    x = ν / nus

    j = Ne * EE * EE * ν / 2. / sqrt(3) / CL / θe / θe * I_I(x) # [g/s^2/cm = ergs/s/cm^3]

    if isnan(j) || isinf(j)
        println("j nan in Dexter fit: j $j x $x nu $ν nus $nus Thetae $θe")
    end
    return j
end
function maxwell_juettner_leung_I(Ne, ν, θe, B, θ)
    K2 = max(besselk(2, 1. / θe), SMALL)
    nuc = EE * B / (2. * π * ME * CL)
    nus = (2. / 9.) * nuc * θe * θe * sin(θ)
    if (ν > 1.e12 * nus)
        return 0.0
    end

    x = ν / nus
    f = (x^(1. / 2.) + 2.0^(11. / 12.) * x^(1. / 6.))^2
    j = (sqrt(2.) * π * EE^2 * Ne * nus / (3. * CL * K2)) * f * exp(-x^(1. / 3.))

    if isnan(j) || isinf(j)
        println("j nan in Leung fit: j $j f $f x $x nu $ν nus $nus nuc $nuc K2 $K2 Thetae $θe")
    end
    return j
end
function maxwell_juettner_I(B, θ, θe, ν, ne)

    #TODO: For now, we are gonna return leung as it's non polarized, but original IPOLE code implementation has dexter params.dexter_fit = 1 for polarized transport
    return maxwell_juettner_leung_I(ne, ν, θe, B, θ)

    νc = get_nu_c(B)

    νs = (2. /9.) * νc * θe * θ^2

    X = ν/νs

    prefactor = ne * EE^2 * νc/(CL)

    term1 = sqrt(2.) * π / 27. * sin(θ)
    term2 = ((X^(1. /2.)) + 2.0^(11. /12.) * X^(1. /6.))^2
    term3 = exp(-X^(1. /3.))

    ans = prefactor * term1 * term2 * term3

    if isnan(ans) || isinf(ans)
        println("Invalid maxwell_juettner_I calculation:")
        println("B = $B, θ = $θ, θe = $θe, ν = $ν, ne = $ne")
        println("Computed values: νc = $νc, νs = $νs, X = $X, prefactor = $prefactor, term1 = $term1, term2 = $term2, term3 = $term3")
        error("Resulting intensity is NaN or Inf")
    end
    return ans
end

