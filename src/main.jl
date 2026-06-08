# ==============================================================================
# IMPORTACIÓN DE PAQUETES
# En Julia, "using" trae todas las funciones del paquete al espacio de nombres
# directamente, sin necesidad de usar prefijos.
# ==============================================================================

using Printf        # Permite imprimir con formato (como printf en C)
using Base.Threads  # Habilita el uso de múltiples hilos (paralelismo)
using LinearAlgebra # Operaciones de álgebra lineal (productos, normas, etc.)
using StaticArrays  # Vectores y matrices de tamaño fijo, almacenados en el stack
                    # (mucho más rápidos que los arrays normales de Julia para tamaños pequeños)
using ForwardDiff  # Diferenciación automática en modo forward (usado para calcular gradientes)

# ==============================================================================
# DEFINICIÓN DE TIPOS PERSONALIZADOS
# "const" define una constante global. En Julia, definir alias de tipos
# con "const" es una práctica común para escribir código más legible
# y reutilizable.
# ==============================================================================

# MVector es un vector mutable de tamaño fijo almacenado en el stack.
# MVec4 = vector mutable de 4 elementos Float64 (números de punto flotante de 64 bits)
const MVec4  = MVector{4,Float64}

# Versión genérica de MVec4: acepta cualquier tipo T (útil para ForwardDiff,
# que necesita trabajar con tipos duales, no solo Float64)
const TMVec4{T} = MVector{4,T} #T es un alias par aobjetos genericos, ej: T = Float64

# MMat4 = matriz mutable de 4x4 elementos Float64
const MMat4 = MMatrix{4,4,Float64}

# Versión genérica de MMat4. El "16" es el número total de elementos (4x4=16),
# requerido explícitamente por StaticArrays
const TMMat4{T} = MMatrix{4,4,T,16} #Si no esta declarado el tipo de objeto es necesario especificar el numero de elementos para guardarlo en la memoria 

# Tensor 3D mutable de 4x4x4 = 64 elementos Float64
# Representa objetos como los símbolos de Christoffel (conexión afín) en relatividad general
const Tensor3D = MArray{Tuple{4,4,4}, Float64, 3, 64}

# Versión genérica del tensor 3D
const TTensor3D{T} = MArray{Tuple{4,4,4}, T, 3, 64}

# ==============================================================================
# DEFINICIÓN DE STRUCTS (ESTRUCTURAS DE DATOS)
# En Julia, "struct" define un tipo compuesto. 
# - "mutable struct" permite modificar los campos después de crearlo.
# - "struct" sin mutable es inmutable (más eficiente en memoria).
# ==============================================================================

# OfTrajM: estructura mutable que almacena el estado de una trayectoria geodésica.
# Se usa cuando los valores necesitan modificarse durante la integración.
mutable struct OfTrajM
    dl::Float64         # Paso de integración a lo largo de la geodésica
    X::MVec4            # Posición 4D del fotón (t, r, θ, φ) en coordenadas de Boyer-Lindquist
    Kcon::MVec4         # 4-momento contravariante del fotón (dirección de propagación)
    Xhalf::MVec4        # Posición a mitad del paso (usado en el integrador RK2)
    Kconhalf::MVec4     # Momento a mitad del paso (RK2)
end

# OfTrajS: versión inmutable de OfTrajM usando SVector (vector estático inmutable).
# Es más eficiente cuando no se necesita modificar los valores.
struct OfTrajS
    dl::Float64
    X::SVector{4, Float64}
    Kcon::SVector{4, Float64}
    Xhalf::SVector{4, Float64}
    Kconhalf::SVector{4, Float64}
end

# OfTraj: estructura completa para diferenciación automática.
# Además de la trayectoria, almacena las derivadas de posición y momento
# con respecto a los parámetros del modelo:
# - θo: inclinación del observador
# - a: spin del agujero negro (parámetro de Kerr)
# Estos campos son los diferenciales dX/dP y dK/dP que Jipole calcula
# mediante diferenciación automática (AD).
struct OfTraj
    dl::Float64
    X::SVector{4, Float64}
    Kcon::SVector{4, Float64}
    Xhalf::SVector{4, Float64}
    Kconhalf::SVector{4, Float64}
    dX_dθo::SVector{4, Float64}  # Derivada de la posición respecto a la inclinación del observador
    dK_dθo::SVector{4, Float64}  # Derivada del momento respecto a la inclinación del observador
    dX_da::SVector{4, Float64}   # Derivada de la posición respecto al spin del agujero negro
    dK_da::SVector{4, Float64}   # Derivada del momento respecto al spin del agujero negro
end

# ==============================================================================
# INCLUSIÓN DE ARCHIVOS EXTERNOS
# "include" en Julia es equivalente a pegar el contenido del archivo
# directamente aquí. Es la forma de modularizar el código en Julia.
# ==============================================================================

include("constants.jl")      # Constantes físicas (c, G, etc.) y parámetros globales
include("set_globals.jl")    # Inicialización de variables globales del modelo
include("camera.jl")         # Geometría de la cámara virtual del observador
include("debug_functions.jl")# Funciones auxiliares para depuración
include("metrics.jl")        # Métricas del espaciotiempo (Kerr, Minkowski, etc.)
include("coords.jl")         # Transformaciones de coordenadas
include("tetrads.jl")        # Base de tetrads (marco local del observador en RG)
include("utils.jl")          # Funciones utilitarias generales
include("radiation.jl")      # Transferencia radiativa (emisividad, absorción, etc.)

# MODELOS FÍSICOS DE LA FUENTE
include("maxwell_juettner.jl")        # Distribución de Maxwell-Jüttner para electrones relativistas
include("grid.jl")                    # Manejo de la grilla de datos GRMHD
include("./models/$(MODEL).jl")       # Modelo de la fuente seleccionado dinámicamente
                                      # MODEL es una variable de entorno o global definida en set_globals.jl

# Inclusión condicional de slow light:
# Si SLOW_LIGHT es true, se carga el módulo que implementa la prescripción
# de slow light (los fotones viajan con tiempo de viaje real, no instantáneo)
if(SLOW_LIGHT)
    println("Adding slowlight.jl file...")
    include("./slowlight_test.jl")
end

include("geodesics.jl")       # Integración de geodésicas (trayectorias de los fotones)
include("autodiff.jl")        # Diferenciación automática para calcular gradientes de la imagen
include("gradientdescent.jl") # Optimización por gradiente conjugado para ajuste de parámetros


# ==============================================================================
# FUNCIÓN: IpoleGeoIntensityIntegration
# Integra la intensidad específica a lo largo de cada geodésica previamente
# calculada, produciendo la imagen final pixel por pixel.
# ==============================================================================
function IpoleGeoIntensityIntegration(traj, freq_cgs::Float64, nx::Int, ny::Int, scalefactor::Float64, bhspin::Float64, data = nothing)
    """
    Once the trajectories are calculated, integrate the intensity for each pixel.
    Parameters:
    @traj: Matrix of geodesic trajectories for each pixel.
    @freq_cgs: Frequency in cgs units.
    @res: Resolution of the image (number of pixels).
    @scalefactor: Scale factor for the image intensity.

    Returns:
    A matrix representing the integrated intensity for each pixel in the image.
    """
    # Inicializa la imagen como una matriz de ceros de nx x ny píxeles
    Image = zeros(Float64, nx, ny)

    # Threads.@threads paraleliza el loop externo entre todos los hilos disponibles.
    # Cada hilo procesa un subconjunto de píxeles en la dirección x de forma independiente.
    Threads.@threads for i in 0:(nx - 1)
        for j in 0:(ny - 1)
            # integrate_emission! integra la ecuación de transferencia radiativa
            # a lo largo de la trayectoria del fotón para el píxel (i+1, j+1).
            # El "!" al final del nombre es convención en Julia para funciones
            # que modifican sus argumentos (aquí modifica Image).
            integrate_emission!(traj[i+1, j+1], length(traj[i+1, j+1]), Image, i + 1, j + 1, freq_cgs, bhspin, data)
        end
    end

    # Multiplica por freq_cgs^3: corrección relativista por el corrimiento al rojo
    # (transformación de intensidad específica Iν → Iν/ν³ es invariante de Lorentz,
    # por lo tanto Iν = ν³ × [cantidad invariante])
    return (Image * freq_cgs^3)
end


# ==============================================================================
# FUNCIÓN: OutputStokesParameters
# Calcula y muestra estadísticas de la imagen: flujo total, intensidad promedio
# y máxima, y luminosidad νLν.
# ==============================================================================
function OutputStokesParameters(Image, freq_cgs, scale_factor, res, Dsource)
    println("Image processing complete. Calculating total flux and averages...")

    # Inicialización de variables de salida
    Ftot::Float64 = 0.0   # Flujo total integrado en Jansky
    Iavg::Float64 = 0.0   # Intensidad promedio
    Imax::Float64 = 0.0   # Intensidad máxima encontrada
    imax::Int = 0          # Índice i del píxel más brillante
    jmax::Int = 0          # Índice j del píxel más brillante

    # Recorre todos los píxeles para acumular estadísticas
    for i in 1:res
        for j in 1:res
            Ftot += Image[i, j] * scale_factor  # Acumula flujo escalado
            Iavg += Image[i, j]                 # Acumula intensidad para el promedio
            if (Image[i,j]) > Imax
                imax = i
                jmax = j
                Imax = Image[i, j]              # Actualiza el máximo encontrado
            end
        end
    end

    Iavg *= 1.0/ (res * res)   # Normaliza el promedio dividiendo por el número de píxeles

    # @printf permite imprimir con formato de precisión específica (como en C)
    @printf("Scale = %.15e\n", scale_factor)
    println("imax = $imax, jmax = $jmax, Imax = $Imax, Iavg = $Iavg")
    @printf("Total Flux Fnu = %.15e Jy\n", Ftot)

    # νLν = luminosidad monocromática, una forma estándar de reportar luminosidad en astrofísica
    # Se calcula como: Fν × 4π × D² × ν
    println("nuLnu = $(Ftot * Dsource * Dsource * JY * freq_cgs * 4.0 * π)")
end


# ==============================================================================
# FUNCIÓN: CalculateScaleFactor
# Calcula el factor de escala que convierte la intensidad integrada en unidades
# físicas de flujo (Jansky por píxel).
# El factor de escala depende del tamaño angular del píxel y la distancia a la fuente.
# ==============================================================================
function CalculateScaleFactor(sizex, sizey, pixelsx, pixelsy, SourceD, LengthUnit)
    """
    Calculate the scale factor for the image based on the camera parameters and the source distance.

    Parameters:
    @sizex: Size of the image in the x direction in Rg.
    @sizey: Size of the image in the y direction in Rg.
    @pixelsx: Number of pixels in the x direction.
    @pixelsy: Number of pixels in the y direction.
    @SourceD: Distance to the source in cm.
    @LengthUnit: Length unit in cm (e.g., Rg).
    
    Returns:
    A scalar scale factor for the image intensity.
    """
    # Fórmula: (tamaño físico del píxel en x) × (tamaño físico del píxel en y) / D² / JY
    # Convierte de unidades de Rg² a ángulo sólido por píxel, luego a Jansky
    return (sizex * LengthUnit / pixelsx) * (sizey * LengthUnit / pixelsy) / (SourceD * SourceD) / JY
end


# ==============================================================================
# FUNCIÓN PRINCIPAL: main()
# Orquesta todo el pipeline de generación de imagen:
# 1. Verifica parámetros
# 2. Configura la cámara
# 3. Calcula geodésicas
# 4. Integra la transferencia radiativa
# @time mide el tiempo total de ejecución del bloque
# ==============================================================================
function main()
    @time begin
        check_parameters()  # Verifica que los parámetros globales sean consistentes

        # Imprime información del modelo actual
        println("Generating an image with size $(nx) x $(ny) pixels")
        println("Model Parameters: A = $A, α = $α_analytic, height = $height, l0 = $l0, a = $a")
        println("MBH = $MBH, L_unit = $L_unit")
        println("Dsource = $Dsource")

        # Calcula la posición 4D de la cámara en coordenadas de Boyer-Lindquist
        # a partir de las coordenadas esféricas (r, θ, φ) del observador
        Xcam = camera_position(rcam, thcam, phicam, Rout)

        # Campo de visión (field of view) en radianes, normalizado por la distancia rcam
        fovx = DX/(rcam)
        fovy = DY/(rcam)

        # Calcula el factor de escala de la imagen
        scale_factor = CalculateScaleFactor(DXsize, DYsize, nx, ny, SourceD, L_unit)
        println("Running on ", Threads.nthreads(), " threads")

        # PASO 1: Traza las geodésicas hacia atrás desde la cámara hasta la región de emisión
        # usando el integrador RK2 (Runge-Kutta de segundo orden)
        trajectory = CalculateGeodesics(Xcam, fovx, fovy, freqcgs, maxnstep, nx, ny)

        # PASO 2: Integra la ecuación de transferencia radiativa a lo largo de cada geodésica
        # para obtener la intensidad específica en cada píxel
        Image = zeros(Float64, nx, ny)
        Image = IpoleGeoIntensityIntegration(trajectory, freqcgs, nx, ny, scale_factor)
    end
end
