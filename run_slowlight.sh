#!/bin/bash
#SBATCH --job-name=SlowLightJipole
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=03:00:00
#SBATCH --partition=cpu-preempt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8            # 8 hilos para Threads.@threads
#SBATCH --mem=120G                    # misma memoria que tu job de referencia
#SBATCH --account=bekt-delta-cpu
#SBATCH --constraint="scratch"

# Información del job
echo "El trabajo empieza en $(hostname)"
echo "Fecha: $(date)"
echo "Memoria total: $(grep MemTotal /proc/meminfo)"
echo "Hilos disponibles: $SLURM_CPUS_PER_TASK"

# Definir número de hilos para Julia
# SLURM_CPUS_PER_TASK toma automáticamente el valor de --cpus-per-task
export JULIA_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Correr el script de slow light
# --project="." le dice a Julia que use el Project.toml de JipoleBrisk/
julia --project="." SlowLight_torus.jl

echo "El trabajo terminó en $(date)"
