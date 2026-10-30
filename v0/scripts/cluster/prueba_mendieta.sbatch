#!/bin/bash
# Prueba del entorno y del backbone en un nodo con GPU de Mendieta (partición short, ~15 min).
#   sbatch scripts/cluster/prueba_mendieta.sbatch && tail -f logs/prueba-*.out
# Mide: copia del entorno al /scratch, import de torch, GPU, matmul, y el modelo del paper (8,5 M parámetros,
# 2560 tokens) con distintos lotes en fp32 y bf16: pasos/s y memoria pico. Sirve para estimar el costo de la fase 4.
#SBATCH --job-name=job2vec_prueba
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-0:30
#SBATCH --output=logs/prueba-%j.out

. /etc/profile                          # ANTES de set -u (skill ccad-unc, lecciones §3)
set -euo pipefail
source scripts/cluster/entorno_scratch.sh
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

rc=0
python3 -u -X faulthandler -W ignore::DeprecationWarning scripts/prueba_backbone.py \
    --dispositivo cuda --lotes 4 8 16 32 --pasos 20 --precision fp32 bf16 || rc=$?
echo "código de salida: $rc"
exit $rc
