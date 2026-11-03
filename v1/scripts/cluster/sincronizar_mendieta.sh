#!/usr/bin/env bash
# Copia el proyecto a Mendieta (CCAD-UNC) o trae resultados. Se corre en la máquina local, desde la raíz del repo.
#
#   bash scripts/cluster/sincronizar_mendieta.sh subir [USUARIO]     # código, configs y scripts (sin datos)
#   bash scripts/cluster/sincronizar_mendieta.sh datos [USUARIO]     # enlace data/processed -> ../../v0/data/processed (no copia datos)
#   bash scripts/cluster/sincronizar_mendieta.sh bajar [USUARIO]     # logs e informes (sin checkpoints)
#
# Se copia el árbol de trabajo, no un git clone (puede haber cambios sin commitear). Adaptado de land2vec.
set -euo pipefail
cd "$(dirname "$0")/../.."

ACCION=${1:?subir | datos | bajar}
USUARIO=${2:-grosati}
HOST=${HOST:-mendieta.ccad.unc.edu.ar}
DEST=${DEST:-labor2vec/v1}          # relativo al home del cluster
R="$USUARIO@$HOST:$DEST"

case "$ACCION" in
subir)
    # logs/ tiene que existir antes del sbatch: Slurm no crea el directorio de --output y el job falla sin aviso
    ssh "$USUARIO@$HOST" "mkdir -p $DEST/logs $DEST/data"
    rsync -avz --exclude __pycache__ --exclude '*.egg-info' \
        pyproject.toml README.md src scripts configs tests "$R/"
    ;;
datos)
    # v1 usa los datos de v0 (mismo panel y mismo corpus): un enlace, no una copia; lo propio de v1 va a data/.
    # Es el mismo enlace relativo que en local; ~/labor2vec/v0/data/processed tiene que existir en el cluster.
    ssh "$USUARIO@$HOST" "mkdir -p $DEST/data && ln -sfn ../../v0/data/processed $DEST/data/processed && test -d $DEST/data/processed/corpus"
    ;;
bajar)
    mkdir -p logs/mendieta informes
    rsync -avz "$R/logs/" logs/mendieta/ 2>/dev/null || true
    rsync -avz "$R/informes/" informes/ 2>/dev/null || true
    ;;
*)
    echo "acción desconocida: $ACCION (subir | datos | bajar)" >&2; exit 1 ;;
esac
