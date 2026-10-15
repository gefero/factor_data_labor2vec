#!/usr/bin/env bash
# Empaqueta el entorno en un solo .tar para correrlo desde el /scratch local de cada nodo. Se corre en la cabecera
# de Mendieta, desde ~/labor2vec (lo llama entorno_mendieta.sh). Rehacerlo cada vez que cambie .venv.
#
# Por qué: en un nodo, `import torch` desde el /home (NFS) tardó 710 s; desde el /scratch, 4 s (skill ccad-unc,
# lecciones §1). El paquete es UN árbol de Python: el CPython de uv con el site-packages de .venv adentro, así los
# .pth (incluido el del paquete editable labor2vec, que apunta a ~/labor2vec/src) se procesan como siempre.
# Adaptado de land2vec/scripts/cluster/empaquetar_entorno.sh.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY_DIR=$(ls -d "$HOME"/.local/share/uv/python/cpython-3.13.*-linux-x86_64-gnu | sort -V | tail -1)
SITE=.venv/lib/python3.13/site-packages
SALIDA=entorno/py313_labor2vec.tar
[ -x "$PY_DIR/bin/python3.13" ] || { echo "falta el Python de uv en $PY_DIR (correr entorno_mendieta.sh)" >&2; exit 1; }
[ -d "$SITE" ] || { echo "falta $SITE (correr entorno_mendieta.sh)" >&2; exit 1; }

mkdir -p entorno
echo "Python: $PY_DIR"
echo "bibliotecas: $SITE ($(du -sh "$SITE" | cut -f1))"
tar -cf "$SALIDA.tmp" --exclude __pycache__ \
    -C "$PY_DIR" . \
    -C "$PWD/.venv/lib/python3.13" --transform 's,^site-packages,./lib/python3.13/site-packages,' site-packages
mv "$SALIDA.tmp" "$SALIDA"
ls -lh "$SALIDA"
