#################################################################################
# CELDA 1 — Configuración global del modelo
# SLOW_LIGHT = true activa slowlight.jl dentro de main.jl.
# El include hereda estas constantes porque no crea un scope nuevo en Julia.


const MODEL = "iharm"
const MBH = 6.5e9        # Masa del agujero negro en masas solares (M87*)
const SLOW_LIGHT = true
include("Jipole/src/main.jl");

#################################################################################
# CELDA 2 — Ruta a los dumps y configuración de la ventana temporal

# La plantilla %05d se reemplaza por el índice del dump con 5 dígitos.
# Ejemplo: índice 21000 → "torus.out0.21000.h5"
#
# dump_start: índice del primer dump a usar
# dump_max:   índice del último dump a usar
# ImageCadence: cada cuántas unidades M se genera una imagen
#
# Datos de tu simulación:
#   ΔT entre dumps = 0.5 M
#   Rango 21000–21100 = 100 dumps × 0.5 M = 50 M
#   Imágenes posibles con ImageCadence=10 → 5 imágenes
#
# Recomendación: empieza con un rango pequeño (21000–21100) para probar
# que todo funciona antes de correr los 875 dumps completos.

const all_dumps_path = "/work/hdd/bekt/bprather/InterpStudy/LowRes/torus.out0.%05d.h5"

const dump_start = 21000   # índice del primer dump
const dump_max   = 21100   # índice del último dump (100 dumps = 50 M)
const ImageCadence = 10    # una imagen cada 10 M → ~5 imágenes en este rango

# Inicializa el struct de slow light con el contador en dump_start
# Los campos 0.0 y "" se llenan en celdas posteriores
params_slowlight = OfSlowLight(dump_max, dump_start, ImageCadence, 0.0, 0.0, 0.0, "")

# Construye la ruta del primer dump y avanza el contador a dump_start + 1
params_slowlight.current_dumps_path = update_dump_path()

#################################################################################
# CELDA 3 — Parámetros del modelo de temperatura (prescripción R-β)
# Mościbrodzka et al. (2016), A&A, 586, A38

# trat_large = 20: Ti/Te en el disco (baja magnetización, beta > beta_crit)
# trat_small = 1:  Ti/Te en el jet  (alta magnetización, beta < beta_crit)
# beta_crit  = 1:  valor de plasma-β que separa disco de jet
# sigma_cut  = 1:  celdas con σ > 1 se excluyen de la emisión (jet)

trat_large       = 20.
const trat_small = 1.
const beta_crit  = 1.0
const th_beg     = 1.74e-2
const sigma_cut  = 1.0
const sigma_cut_high = -1.0;

#################################################################################
# CELDA 4 — Lectura del header del primer dump
# Lee los metadatos: spin del agujero negro (params.a), tamaño del grid,
# coordenadas, radio exterior del dominio (params.Rout), etc.

const params = read_header(params_slowlight.current_dumps_path);

#################################################################################
# CELDA 5 — Carga inicial de los tres primeros dumps (ventana deslizante)
# Slow light mantiene solo 3 dumps en RAM simultáneamente.
# Cada llamada a load_data avanza el contador interno automáticamente:
#   simulation_data[1] → dump dump_start     (tA)
#   simulation_data[2] → dump dump_start + 1 (tB)
#   simulation_data[3] → dump dump_start + 2 (reserva)
#
# tA y tB definen el intervalo de interpolación inicial.
# tf es el tiempo del dump_max (límite temporal superior).

const simulation_data = Vector{IharmData}(undef, 3)

# Cada llamada avanza automáticamente al siguiente dump
simulation_data[1] = load_data(params_slowlight.current_dumps_path, trat_large)
simulation_data[2] = load_data(params_slowlight.current_dumps_path, trat_large)
simulation_data[3] = load_data(params_slowlight.current_dumps_path, trat_large)

params_slowlight.tA = simulation_data[1].t  # tiempo del dump más antiguo en memoria
params_slowlight.tB = simulation_data[2].t  # tiempo del dump siguiente

params_slowlight.tf = get_specific_dump_time(params_slowlight.dump_max)  # límite superior

println("tA = $(params_slowlight.tA) M")
println("tB = $(params_slowlight.tB) M")
println("tf = $(params_slowlight.tf) M")
println("ΔT entre dumps = $(params_slowlight.tB - params_slowlight.tA) M")
println("Rango temporal total = $(params_slowlight.tf - params_slowlight.tA) M")
println("Imágenes esperadas ≈ $(floor(Int, (params_slowlight.tf - params_slowlight.tA) / ImageCadence))")

#################################################################################
# CELDA 6 — Parámetros del observador y la cámara
# ro   = 1000 Rg: distancia del observador (infinito asintótico)
# th   = 163°: equivale a ver el sistema con 17° de inclinación (casi de frente
#              al jet), similar a la geometría de M87*
# freq = 230 GHz: frecuencia del Event Horizon Telescope
# res  = 128: resolución de prueba (128×128 = 16384 geodésicas)
#             Aumentar a 256 o 512 para resultados de mayor calidad
#
# DXsize/DYsize: tamaño del plano imagen en Rg, escalado desde la distancia
#                física a la fuente usando el ángulo de 160 μas de campo de vista
const ro  = 1000.0
const th  = 163.0
const phi = 0.0

const res      = 128
const pixels_x = 128
const pixels_y = 128

const SourceD = 16.9e6 * PC   # distancia a M87* en cm

const Rh = 1 + sqrt(1. - params.a * params.a)  # radio del horizonte de eventos

const freq = 230e9   # Hz

const DXsize = SourceD / L_unit / MUAS_PER_RAD * 160
const DYsize = SourceD / L_unit / MUAS_PER_RAD * 160

const fovx = DXsize / ro
const fovy = DYsize / ro

const xoff = 0.0
const yoff = 0.0

#################################################################################
# CELDA 7 — Ray tracing con almacenamiento permanente de geodésicas
# DIFERENCIA CLAVE respecto a fast-light:
#   - all_geodesics guarda TODAS las geodésicas permanentemente (una por píxel)
#   - En fast-light los buffers se sobreescribían con cada píxel
#   - Aquí se conservan porque process_slowlight_images! las reutiliza
#     para cada fotograma del movie sin recalcularlas
#
# Se calculan además tres tiempos diagnóstico (siguiendo convención de ipole):
#   t0    → tiempo más antiguo absoluto de todos los fotones
#   tgeof → tiempo más antiguo en la zona de emisión activa (r < 100 Rg)
#   tgeoi → tiempo más reciente en la zona de emisión activa
#
# Estos valores determinan qué dumps necesita cargar slow light.

using ProgressMeter

Xcamera = MVec4(camera_position(ro, th, phi, params.a, params.Rout))

const freq_unitless = freq * HPL / (ME * CL * CL)

midplane_crossings = zeros(Int, pixels_x, pixels_y)
nsteps             = zeros(Int, pixels_x, pixels_y)

const nthreads = Threads.nthreads() + 1
println("Corriendo en $nthreads hilos")

const maxnstep = 15000

dummy_svec = @SVector zeros(4)
dummy_traj = OfTrajS(0.0, dummy_svec, dummy_svec, dummy_svec, dummy_svec)

# Buffer temporal por hilo para evitar condiciones de carrera
thread_trajs = [Vector{OfTrajS}(undef, maxnstep) for _ in 1:nthreads]
for t in 1:nthreads
    for k in 1:maxnstep
        thread_trajs[t][k] = dummy_traj
    end
end

# Almacenamiento permanente: una geodésica completa por píxel
const all_geodesics = Matrix{Vector{OfTrajS}}(undef, pixels_x, pixels_y)

# Arrays de diagnóstico temporal por hilo
thread_t0    = zeros(Float64, nthreads)
thread_tgeoi = fill(-1e100, nthreads)   # busca el máximo → inicializa muy negativo
thread_tgeof = zeros(Float64, nthreads) # busca el mínimo → inicializa en cero

p = Progress(pixels_x * pixels_y; desc="Raytracing...", showspeed=true, barlen=30)
ProgressMeter.ijulia_behavior(:clear)

println("Trazando geodésicas...")
Threads.@threads for i in 0:(pixels_x - 1)
    tid = Threads.threadid()

    for j in 0:(pixels_y - 1)
        # Traza la geodésica del píxel (i,j) hacia atrás desde la cámara
        nstep, midplane_crossings[i+1, j+1] = get_pixel(
            thread_trajs[tid], i, j, Xcamera,
            fovx, fovy, freq_unitless,
            pixels_x, pixels_y, params.a,
            Rh, params.Rout, params.rmax_geo, xoff, yoff
        )
        nsteps[i+1, j+1] = nstep

        # Guarda la geodésica permanentemente (solo los pasos reales, no los 15000)
        all_geodesics[i+1, j+1] = thread_trajs[tid][1:nstep]

        # Tiempo del último paso (el más antiguo = más negativo)
        final_step_time = thread_trajs[tid][nstep].X[1]
        if final_step_time < thread_t0[tid]
            thread_t0[tid] = final_step_time
        end

        # Calcula tgeoi y tgeof recorriendo cada paso de la geodésica
        pixel_tgeoi = 1.0
        pixel_tgeof = 1.0
        for k in 1:nstep
            X    = thread_trajs[tid][k].X
            K    = thread_trajs[tid][k].Kcon
            log_r   = X[2]
            t_coord = X[1]
            k_r     = K[2]

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

        ProgressMeter.next!(p; showvalues=[
            (:thread_id, tid),
            (:pixel, "($i, $j)"),
            (:total_done, "$(i*pixels_y + j)/$(pixels_x * pixels_y)")
        ])
    end
end

finish!(p)

# Reducción final: combina extremos de todos los hilos
t0    = minimum(thread_t0)
tgeof = minimum(thread_tgeof)
tgeoi = maximum(thread_tgeoi)

println("t0    (tiempo más antiguo absoluto):         $t0")
println("tgeof (dump más antiguo en zona de emisión): $tgeof")
println("tgeoi (dump más reciente en zona de emisión): $tgeoi")

# Libera RAM: los buffers temporales ya no se necesitan
# all_geodesics se conserva porque process_slowlight_images! la reutiliza
thread_t0    = nothing
thread_tgeoi = nothing
thread_tgeof = nothing
thread_trajs = nothing
GC.gc()
println("Buffers temporales liberados. Procediendo a producir imágenes...")

#################################################################################
# CELDA 8 — Guardado opcional de all_geodesics
# all_geodesics contiene la geometría completa de todas las geodésicas.
# Como es estática (no depende del plasma), guardarla permite reutilizarla
# en corridas futuras sin repetir el ray tracing, que es el paso más costoso.
#
# Se guarda en el directorio actual con el nombre "all_geodesics.jld2".
# Para cargarlo en una sesión futura:
#   using JLD2
#   @load "all_geodesics.jld2" all_geodesics nsteps
#
# NOTA: puede ocupar varios GB dependiendo de la resolución y maxnstep.
# Para res=128 con maxnstep=15000 ocupa aproximadamente 1-3 GB.
# Comenta estas líneas si no necesitas reutilizar las geodésicas.

using JLD2
println("Guardando all_geodesics en disco...")
@save "all_geodesics.jld2" all_geodesics nsteps
println("all_geodesics guardado exitosamente.")

#################################################################################
# CELDA 9 — Producción del movie en slow-light
# Las imágenes se guardan en el DIRECTORIO ACTUAL desde donde se corre Julia,
# con el formato: Image.XXXXX.txt
# donde XXXXX es el tiempo de coordenada del fotograma (ej: Image.19500.txt)
#
# Son matrices de intensidad en texto plano, legibles con:
#   Julia:  readdlm("Image.19500.txt")
#   Python: numpy.loadtxt("Image.19500.txt")
#
# Con dump_start=21000, dump_max=21100 e ImageCadence=10:
#   ΔT = 0.5 M por dump → rango total = 50 M → ~5 imágenes producidas
#
# process_slowlight_images! orquesta el pipeline completo:
#   1. Para cada fotograma, reutiliza all_geodesics (geometría estática)
#   2. Interpola el plasma entre dumps según el tiempo de emisión de cada fotón
#   3. Integra la transferencia radiativa → Image[i,j]
#   4. Escala por freq³ (corrección por invariante de Lorentz de Iν)
#   5. Escribe el fotograma a disco como Image.XXXXX.txt
#   6. Avanza la ventana deslizante con update_data! cuando es necesario
#
# El espaciotiempo es estático → all_geodesics es la misma para todos los frames
# El plasma evoluciona       → simulation_data se actualiza con cada nuevo dump

#################################################################################
# process_slowlight_images! orquesta el pipeline completo:
#   1. Para cada fotograma, reutiliza all_geodesics (geometría estática)
#   2. Interpola el plasma entre dumps según el tiempo de emisión de cada fotón
#   3. Integra la transferencia radiativa → Image[i,j]
#   4. Escala por freq³ (corrección por invariante de Lorentz de Iν)
#   5. Escribe el fotograma a disco
#   6. Avanza la ventana deslizante con update_data! cuando es necesario
#
# El espaciotiempo es estático → all_geodesics es la misma para todos los frames
# El plasma evoluciona       → simulation_data se actualiza con cada nuevo dump

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
