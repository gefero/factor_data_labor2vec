"""Mide cuántas secuencias/s entrega el DataLoader solo (sin modelo), con los workers dados.

    python -u scripts/medir_carga.py --corpus data/processed/corpus --lote 32 --workers 10 --lotes 150
"""

import argparse
import time

from labor2vec.tokens.datos import DataModuleMLER
from labor2vec.tokens.documento import Opciones

ap = argparse.ArgumentParser()
ap.add_argument("--corpus", default="data/processed/corpus")
ap.add_argument("--lote", type=int, default=32)
ap.add_argument("--workers", type=int, default=10)
ap.add_argument("--lotes", type=int, default=150)
ap.add_argument("--fin-ventana", type=int, default=239)
a = ap.parse_args()

t0 = time.time()
dm = DataModuleMLER(a.corpus, op=Opciones(fin_ventana=a.fin_ventana), tamano_lote=a.lote, num_workers=a.workers)
dm.setup()
print(f"{time.time() - t0:.1f} s en cargar el corpus ({len(dm.ds['train']):,} trabajadores de train)", flush=True)
it = iter(dm.train_dataloader())
next(it)  # arranque de los workers
t, n, tok = time.time(), 0, 0
for _, b in zip(range(a.lotes), it):
    n += b["input_ids"].shape[0]
    tok += b["input_ids"].shape[0] * b["input_ids"].shape[2]
dt = time.time() - t
print(f"workers={a.workers} lote={a.lote}: {n / dt:.0f} secuencias/s, {tok / dt:,.0f} tokens/s (con relleno)", flush=True)
