"""Pérdida del MLM de un modelo pre-entrenado, por categoría de token objetivo (validación).

    PYTHONPATH=src ../v0/.venv/bin/python scripts/mlm_por_categoria.py runs/humo_cpu_grilla [--lotes 50]

Lee `config.json` y `modelo_final.pt` de la carpeta de la corrida, arma el DataModule con la misma configuración y recorre
la validación (orden y enmascarado deterministas). Sirve para ver cuánto de la pérdida del MLM viene de los tokens SINREG
(plans/PLAN_v1.md, fase 2) y para comparar el resto de las categorías entre una corrida con grilla y otra sin ella."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from labor2vec import entrenar


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corrida", type=Path)
    ap.add_argument("--lotes", type=int, default=None, help="máximo de lotes de validación (por defecto, todos)")
    ap.add_argument("--salida", type=Path, default=None)
    a = ap.parse_args()

    cfg = json.loads((a.corrida / "config.json").read_text())
    cfg.pop("hparams_modelo")
    dm, modelo = entrenar.preparar(cfg)
    ck = torch.load(a.corrida / "modelo_final.pt", map_location="cpu", weights_only=False)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()

    nombres = np.array(dm.vocab.tokens)
    cats = np.array(dm.vocab.categorias)
    acum = defaultdict(lambda: np.zeros(3))  # n, nll, aciertos
    with torch.no_grad():
        for i, batch in enumerate(dm.val_dataloader()):
            if a.lotes is not None and i >= a.lotes:
                break
            mlm, _ = modelo(batch)
            t = batch["target_tokens"].long()
            ok = t != 0
            logits, t = mlm[ok].float(), t[ok]
            nll = F.cross_entropy(logits, t, reduction="none").numpy()
            acierto = (logits.argmax(-1) == t).numpy()
            for tok, x, y in zip(t.numpy(), nll, acierto):
                clave = nombres[tok] if cats[tok] == "sinreg" else cats[tok]
                acum[clave] += (1, x, y)

    total_n = sum(v[0] for v in acum.values())
    total_nll = sum(v[1] for v in acum.values())
    filas = {k: {"objetivos": int(v[0]), "frac_objetivos": float(v[0] / total_n), "nll_medio": float(v[1] / v[0]),
                 "acierto": float(v[2] / v[0]), "frac_de_la_perdida": float(v[1] / total_nll)}
             for k, v in sorted(acum.items(), key=lambda kv: -kv[1][0])}
    sin = [v for k, v in acum.items() if not k.startswith("SINREG")]
    resto = {"objetivos": int(sum(v[0] for v in sin)), "nll_medio": float(sum(v[1] for v in sin) / sum(v[0] for v in sin))}
    out = {"corrida": str(a.corrida), "por_categoria": filas, "sin_sinreg": resto,
           "nll_medio_total": float(total_nll / total_n)}
    (a.salida or a.corrida / "mlm_por_categoria.json").write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"{'categoría':14s} {'objetivos':>9s} {'% obj.':>7s} {'NLL':>7s} {'acierto':>8s} {'% pérdida':>9s}")
    for k, v in filas.items():
        print(f"{k:14s} {v['objetivos']:9d} {100*v['frac_objetivos']:6.1f}% {v['nll_medio']:7.3f} {v['acierto']:8.3f} "
              f"{100*v['frac_de_la_perdida']:8.1f}%")
    print(f"NLL medio total {out['nll_medio_total']:.3f}; sin SINREG {resto['nll_medio']:.3f} ({resto['objetivos']} objetivos)")


if __name__ == "__main__":
    main()
