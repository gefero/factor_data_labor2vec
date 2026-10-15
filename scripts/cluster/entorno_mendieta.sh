#!/usr/bin/env bash
# Crea el entorno de Python del proyecto en Mendieta. Se corre en la CABECERA (los nodos no tienen internet),
# desde ~/labor2vec, después de sincronizar_mendieta.sh subir:
#
#   bash scripts/cluster/entorno_mendieta.sh
#   sbatch scripts/cluster/prueba_mendieta.sbatch     # después: verificar en un nodo con GPU
#
# Adaptado de land2vec/scripts/cluster/entorno_mendieta.sh. Notas (ver skill ccad-unc, lecciones §1):
# - Python 3.13 de uv, variante x86-64 base: los Xeon Ivy Bridge de los nodos no tienen AVX2.
# - only-managed: el Python de uv (relocalizable), no el de Miniconda que esté primero en el PATH.
# - torch cu126 (A30 = sm_80) a través del extra `cu126` de pyproject.toml; uv.lock fija las versiones.
set -euo pipefail
cd "$(dirname "$0")/../.."

PY=${PY:-cpython-3.13-linux-x86_64-gnu}
export UV_PYTHON_PREFERENCE=only-managed

uv python install "$PY"
uv sync --frozen --python "$PY" --extra cu126 --link-mode=copy

.venv/bin/python -c "import torch, lightning, performer_pytorch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
# los jobs no corren .venv desde el /home (lento por NFS): lo empaquetan para el /scratch del nodo
bash scripts/cluster/empaquetar_entorno.sh
echo "Entorno listo. Falta probarlo en un nodo con GPU: sbatch scripts/cluster/prueba_mendieta.sbatch"
