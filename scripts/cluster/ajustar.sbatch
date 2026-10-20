#!/bin/bash
# Ajuste fino de T1 (fase 6) en una A30 de Mendieta. Uso:
#
#   sbatch --mail-type=BEGIN,END,FAIL --mail-user=CORREO -J t1_NOMBRE scripts/cluster/ajustar.sbatch NOMBRE CONFIG [--set clave=valor ...]
#
# Ejemplos (NOMBRE es la carpeta de resultados ~/labor2vec/runs/NOMBRE; CONFIG, configs/ajuste_t1.yaml o gru_t1.yaml):
#   ... ajustar.sbatch t1_l2v_s0 configs/ajuste_t1.yaml --set semilla=0
#   ... ajustar.sbatch t1_scratch_s0 configs/ajuste_t1.yaml --set semilla=0 --set ajuste.desde_cero=true
#   ... ajustar.sbatch t1_gru_s0 configs/gru_t1.yaml --set semilla=0
#
# Igual que preentrenar.sbatch: se reanuda solo (--reanudar) y para a tiempo (job menos un margen) con código 0, así que
# para seguir una corrida cortada se vuelve a encolar el MISMO comando con --dependency=afterany:JOBID. Al terminar deja
# modelo_final.pt y predicciones_{val,test}.parquet en la carpeta de resultados.
#SBATCH --job-name=t1_ajuste
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --output=logs/ajustar-%x-%j.out

. /etc/profile                          # ANTES de set -u (skill ccad-unc, lecciones §3)
set -euo pipefail
NOMBRE=${1:?falta NOMBRE (carpeta de resultados)}
CONFIG=${2:?falta CONFIG (YAML)}
shift 2
cd "$HOME/labor2vec"
source scripts/cluster/entorno_scratch.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, resultados en runs/$NOMBRE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Comprobación de importaciones (si falla, no se gasta el resto)"
python3 -u -X faulthandler -c "
import torch, lightning, performer_pytorch, pyarrow, yaml
import labor2vec.ajustar
print('importaciones OK; CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))
"

# minutos que quedan antes de parar (límite del job menos el margen)
minutos_utiles() {
    local fin limite
    fin=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
    limite=$(squeue -h -j "$SLURM_JOB_ID" -o %l)
    python3 - "$fin" "$limite" <<'PY'
import datetime as d, sys
fin, limite = d.datetime.fromisoformat(sys.argv[1]), sys.argv[2]
dias, _, resto = limite.rpartition("-")
h = [int(x) for x in resto.split(":")]
h = [0] * (3 - len(h)) + h
total_min = int(dias or 0) * 1440 + h[0] * 60 + h[1] + h[2] / 60
margen = 5 if total_min <= 60 else 15
print(max(1.0, (fin - d.datetime.now()).total_seconds() / 60 - margen))
PY
}

MAXMIN=$(minutos_utiles)
echo "== Ajuste (para solo a los $MAXMIN min)"
python3 -u -m labor2vec.ajustar --config "$CONFIG" --salida "runs/$NOMBRE" --reanudar --max-minutos "$MAXMIN" "$@"
echo "fin"
