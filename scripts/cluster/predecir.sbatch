#!/bin/bash
# Fase 8: predicciones de modelos ya ajustados con un control temporal sobre la entrada (sin entrenar), en una A30 de
# Mendieta. Uso:
#
#   sbatch --mail-type=BEGIN,END,FAIL --mail-user=CORREO -J f8_ctrl_TAREA scripts/cluster/predecir.sbatch CONFIG [args de predecir]
#
# Ejemplo (deja runs/ctrl_{ninguno,tiempo,orden}/t3_l2v_s{0,1,2}/predicciones_{val,test}.parquet):
#   ... predecir.sbatch configs/ajuste_t3.yaml --corridas runs/t3_l2v_s0 runs/t3_l2v_s1 runs/t3_l2v_s2
#SBATCH --job-name=f8_predecir
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --output=logs/predecir-%x-%j.out

. /etc/profile                          # ANTES de set -u (skill ccad-unc, lecciones §3)
set -euo pipefail
CONFIG=${1:?falta CONFIG (YAML de la tarea)}
shift 1
cd "$HOME/labor2vec"
source scripts/cluster/entorno_scratch.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Comprobación de importaciones (si falla, no se gasta el resto)"
python3 -u -X faulthandler -c "
import torch, lightning, performer_pytorch, pyarrow, yaml
import labor2vec.tareas.predecir
print('importaciones OK; CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))
"

echo "== Predicciones"
python3 -u -m labor2vec.tareas.predecir --config "$CONFIG" "$@"
echo "fin"
