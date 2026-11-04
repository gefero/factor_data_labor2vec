# Se usa con `source` dentro de un sbatch de Mendieta (desde ~/labor2vec/v1_1): descomprime el entorno empaquetado en el
# /scratch local del nodo y lo pone primero en el PATH. Slurm borra el /scratch al terminar el job. Copiado de
# land2vec/scripts/cluster/entorno_scratch.sh.
# v1 **no tiene entorno propio**: usa el tar de v0 (~/labor2vec/v0/entorno, armado con v0/scripts/cluster/entorno_mendieta.sh),
# y como su .pth editable apunta a ~/labor2vec/v0/src, acá se antepone el src de esta carpeta con PYTHONPATH.
ENTORNO_TAR=${ENTORNO_TAR:-$HOME/labor2vec/v0/entorno/py313_labor2vec.tar}
ENTORNO_DIR=/scratch/$USER/$SLURM_JOB_ID/py
[ -f "$ENTORNO_TAR" ] || { echo "falta $ENTORNO_TAR: armar el entorno de v0 (v0/scripts/cluster/entorno_mendieta.sh en la cabecera)" >&2; exit 1; }
mkdir -p "$ENTORNO_DIR"
_t0=$(date +%s)
tar -xf "$ENTORNO_TAR" -C "$ENTORNO_DIR"
export PATH="$ENTORNO_DIR/bin:$PATH" PYTHONNOUSERSITE=1 PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
hash -r
echo "entorno en $ENTORNO_DIR ($(du -sh "$ENTORNO_DIR" | cut -f1)), copiado en $(( $(date +%s) - _t0 )) s; python: $(command -v python3)"
