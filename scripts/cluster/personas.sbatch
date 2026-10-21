#!/bin/bash
# Fase 7: vectores de persona, gradientes (TCAV) y saliency de un modelo ajustado, en una A30 de Mendieta. Uso:
#
#   sbatch --mail-type=BEGIN,END,FAIL --mail-user=CORREO -J f7_NOMBRE scripts/cluster/personas.sbatch NOMBRE CONFIG MODELO [args de extraer]
#
# Ejemplos (los resultados quedan en ~/labor2vec/runs/fase7/NOMBRE/{personas,saliency}.npz):
#   ... personas.sbatch t1  configs/ajuste_t1.yaml runs/t1_l2v_lr0001/modelo_final.pt
#   ... personas.sbatch t3  configs/ajuste_t3.yaml runs/t3_l2v_s0/modelo_final.pt
#   ... personas.sbatch pre configs/ajuste_t1.yaml runs/pre_150ep/modelo_final.pt --sin-cabeza
#SBATCH --job-name=f7_personas
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --output=logs/personas-%x-%j.out

. /etc/profile                          # ANTES de set -u (skill ccad-unc, lecciones §3)
set -euo pipefail
NOMBRE=${1:?falta NOMBRE (carpeta de resultados en runs/fase7)}
CONFIG=${2:?falta CONFIG (YAML)}
MODELO=${3:?falta MODELO (modelo_final.pt)}
shift 3
cd "$HOME/labor2vec"
source scripts/cluster/entorno_scratch.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, resultados en runs/fase7/$NOMBRE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Comprobación de importaciones (si falla, no se gasta el resto)"
python3 -u -X faulthandler -c "
import torch, lightning, performer_pytorch, pyarrow, yaml
import labor2vec.analisis.extraer
print('importaciones OK; CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))
"

echo "== Extracción"
python3 -u -m labor2vec.analisis.extraer --config "$CONFIG" --modelo "$MODELO" --salida "runs/fase7/$NOMBRE" "$@"
echo "fin"
