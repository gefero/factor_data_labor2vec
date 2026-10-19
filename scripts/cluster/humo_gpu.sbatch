#!/bin/bash
# Prueba corta de pre-entrenamiento con datos reales en una A30 de Mendieta (fase 3, plan 3.5; partición short, <= 1 h).
#   sbatch scripts/cluster/humo_gpu.sbatch && tail -f logs/humo_gpu-*.out
# Mide: (1) cuánto entrega el DataLoader con 10 cores; (2) bf16 contra fp32 con la misma semilla (1.000 pasos, lote 16,
# el mayor que entra en fp32 con 2560 tokens, fase 0); (3) bf16 con lote 32: pasos/s y memoria pico.
# Cada paso se registra aunque falle el anterior, pero la comprobación inicial (0) aborta todo.
# Los resultados (CSV y config, sin checkpoints) se copian a ~/labor2vec/runs/humo_gpu_<jobid>/ al terminar.
#SBATCH --job-name=l2v_humo_gpu
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --output=logs/humo_gpu-%j.out

. /etc/profile                          # ANTES de set -u (skill ccad-unc, lecciones §3)
set -uo pipefail
cd "$HOME/labor2vec"
source scripts/cluster/entorno_scratch.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK PYTHONWARNINGS=ignore
echo "nodo $(hostname), CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

TRABAJO=/scratch/$USER/$SLURM_JOB_ID/runs
DESTINO=$HOME/labor2vec/runs/humo_gpu_$SLURM_JOB_ID
copiar() {   # también si el job falla: sin checkpoints (pesan ~100 MB cada uno)
    mkdir -p "$DESTINO" && rsync -a --exclude '*.ckpt' "$TRABAJO/" "$DESTINO/" ; echo "resultados en $DESTINO"
}
trap copiar EXIT

COMUN=(--config configs/preentrenamiento.yaml
       --set datos.num_workers=9 --set entrenamiento.epocas=1 --set entrenamiento.pasos_por_epoca=1000
       --set entrenamiento.val_cada_pasos=250 --set entrenamiento.val_lotes=50 --set entrenamiento.log_cada_pasos=10
       --set entrenamiento.guardar_cada_pasos=500)

echo "== 0. Comprobación de importaciones (si falla, no se gasta el resto)"
python3 -u -X faulthandler -c "
import torch, lightning, performer_pytorch, pyarrow, yaml
import labor2vec.entrenar, labor2vec.tokens.datos
from labor2vec.tokens.vocabulario import Vocabulario
v = Vocabulario.cargar('data/processed/corpus/vocabulario.parquet', 'data/processed/corpus/ramas.parquet')
print('importaciones OK; vocabulario', len(v), 'tokens; CUDA', torch.cuda.is_available(), torch.cuda.get_device_name(0))
" || { echo "FALLÓ la comprobación inicial"; exit 1; }

echo "== 1. DataLoader solo"
python3 -u scripts/medir_carga.py --lote 32 --workers 9 --lotes 150
python3 -u scripts/medir_carga.py --lote 32 --workers 5 --lotes 150

echo "== 2. fp32, lote 16 (<= 15 min)"
python3 -u -m labor2vec.entrenar "${COMUN[@]}" --set datos.tamano_lote=16 --set entrenamiento.precision=32-true \
    --salida "$TRABAJO/fp32_l16" --max-minutos 15
echo "== 3. bf16, lote 16 (<= 12 min)"
python3 -u -m labor2vec.entrenar "${COMUN[@]}" --set datos.tamano_lote=16 --set entrenamiento.precision=bf16-mixed \
    --salida "$TRABAJO/bf16_l16" --max-minutos 12
echo "== 4. bf16, lote 32, 300 pasos (<= 8 min)"
python3 -u -m labor2vec.entrenar "${COMUN[@]}" --set datos.tamano_lote=32 --set entrenamiento.precision=bf16-mixed \
    --set entrenamiento.pasos_por_epoca=300 --set entrenamiento.val_cada_pasos=150 \
    --salida "$TRABAJO/bf16_l32" --max-minutos 8
echo "fin"
