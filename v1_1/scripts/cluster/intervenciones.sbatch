#!/bin/bash
# Fase 9 (B3): vectores y logits del test de un modelo con cada intervención sobre la historia (una pasada por intervención),
# en una A30 de Mendieta. Uso:
#
#   BASE=runs/fase9 sbatch --mail-type=BEGIN,END,FAIL --mail-user=CORREO -J f9_int_NOMBRE scripts/cluster/intervenciones.sbatch NOMBRE CONFIG MODELO [args de extraer]
#
# Los resultados quedan en ~/labor2vec/v1_1/$BASE/NOMBRE_<intervencion>/personas.npz (sólo test). El modelo sin intervención es el que ya
# dejó la fase 7 en runs/fase7/NOMBRE. Ejemplos:
#   ... intervenciones.sbatch t1  configs/ajuste_t1.yaml runs/t1_l2v_lr0001/modelo_final.pt
#   ... intervenciones.sbatch t3  configs/ajuste_t3.yaml runs/t3_l2v_s0/modelo_final.pt
#   ... intervenciones.sbatch pre configs/ajuste_t1.yaml runs/pre_150ep/modelo_final.pt --sin-cabeza
#SBATCH --job-name=f9_intervenciones
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --output=logs/intervenciones-%x-%j.out

. /etc/profile                          # ANTES de set -u (skill ccad-unc, lecciones §3)
set -euo pipefail
NOMBRE=${1:?falta NOMBRE}
CONFIG=${2:?falta CONFIG (YAML)}
MODELO=${3:?falta MODELO (modelo_final.pt)}
shift 3
BASE=${BASE:-runs/fase9}
cd "$HOME/labor2vec/v1_1"
source scripts/cluster/entorno_scratch.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, resultados en $BASE/${NOMBRE}_<intervencion>"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Comprobación de importaciones (si falla, no se gasta el resto)"
python3 -u -X faulthandler -c "
import torch, lightning, performer_pytorch, pyarrow, yaml
from labor2vec.tokens.documento import INTERVENCIONES
import labor2vec.analisis.extraer
print('importaciones OK; CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0), INTERVENCIONES)
"

# una pasada por intervención; si una falla se sigue con las demás y el job termina con error al final
fallo=0
for INT in sin_ultimo_anio sin_cambios_recientes salario_arriba salario_abajo atrasar_2_anios; do
    echo "== $INT"
    python3 -u -m labor2vec.analisis.extraer --config "$CONFIG" --modelo "$MODELO" --salida "$BASE/${NOMBRE}_$INT" \
        --sin-grad --solo-test --intervencion "$INT" "$@" || { echo "FALLÓ $INT"; fallo=1; }
done
echo "fin (fallo=$fallo)"
exit $fallo
