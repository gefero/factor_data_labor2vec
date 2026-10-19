#!/bin/bash
# Pre-entrenamiento (fase 4) en una A30 de Mendieta. Uso:
#
#   sbatch --mail-type=BEGIN,END,FAIL --mail-user=CORREO -J l2v_NOMBRE [-p multi --time=1-12:00] \
#          scripts/cluster/preentrenar.sbatch NOMBRE [--set clave=valor ...]
#
# NOMBRE es la carpeta de resultados (~/labor2vec/runs/NOMBRE: checkpoints, config.json, logs/ con el CSV de
# métricas y, al terminar, modelo_final.pt). Los checkpoints van al /home y no al /scratch, que se borra al terminar.
# El job se reanuda solo (--reanudar): si ya existe last.ckpt retoma de ahí, así que para seguir un entrenamiento
# cortado por el tiempo se vuelve a encolar el MISMO comando, encadenado con afterany y con nombres de job distintos:
#
#   sbatch --dependency=afterany:JOBID ... (mismos argumentos)
#
# Para el límite de tiempo se calcula cuánto le queda al job (squeue %e) menos un margen (5 min si el job dura
# <= 1 h, 15 min si dura más) y se lo pasa a --max-minutos: el entrenamiento guarda y sale con código 0
# (el job termina COMPLETED, no TIMEOUT). Variable opcional: CORTE_PRUEBA=N corta en el paso N y reanuda dentro
# del mismo job (prueba de reanudación en la GPU). Pasarla por el entorno del envío: `CORTE_PRUEBA=1500 sbatch ...`;
# NO con `sbatch --export=ALL,VAR=...`: el job 1180677 quedó retenido con «user env retrieval failed requeued held».
#SBATCH --job-name=l2v_pre
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --output=logs/preentrenar-%x-%j.out

. /etc/profile                          # ANTES de set -u (skill ccad-unc, lecciones §3)
set -euo pipefail
NOMBRE=${1:?falta NOMBRE (carpeta de resultados)}
shift
cd "$HOME/labor2vec"
source scripts/cluster/entorno_scratch.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES, resultados en runs/$NOMBRE"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

echo "== Comprobación de importaciones (si falla, no se gasta el resto)"
python3 -u -X faulthandler -c "
import torch, lightning, performer_pytorch, pyarrow, yaml
import labor2vec.entrenar
print('importaciones OK; CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))
"

# minutos que quedan antes de parar (límite del job menos el margen)
minutos_utiles() {
    local fin limite
    fin=$(squeue -h -j "$SLURM_JOB_ID" -o %e)
    limite=$(squeue -h -j "$SLURM_JOB_ID" -o %l)    # límite del job, p. ej. 1:00:00 o 1-12:00:00
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

CMD=(python3 -u -m labor2vec.entrenar --config configs/preentrenamiento.yaml --salida "runs/$NOMBRE" --reanudar "$@")

if [ -n "${CORTE_PRUEBA:-}" ]; then
    echo "== Corte de prueba en el paso $CORTE_PRUEBA (después se reanuda)"
    "${CMD[@]}" --max-pasos "$CORTE_PRUEBA" --max-minutos "$(minutos_utiles)"
    echo "== Reanudación"
fi
MAXMIN=$(minutos_utiles)
echo "== Entrenamiento (para solo a los $MAXMIN min)"
"${CMD[@]}" --max-minutos "$MAXMIN"
echo "fin"
