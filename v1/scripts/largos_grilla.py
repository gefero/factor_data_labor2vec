"""Fase 1 de v1: largos de los documentos con y sin grilla y proporción de meses SINREG (train, ventana a 2015-12).

    PYTHONPATH=src ../v0/.venv/bin/python scripts/largos_grilla.py [--verificar 5000] [--salida informes/largos_grilla.json]

Carga el corpus de train en memoria (~1 GB) y no escribe nada salvo el JSON de salida. Antes de informar, compara el largo
que da `DatasetPretrain.largos()` con el del documento armado de verdad en una muestra de trabajadores."""

import argparse
import json
from pathlib import Path

import numpy as np

from labor2vec.tareas.datos import leer_etiquetas
from labor2vec.tokens.datos import DatasetPretrain
from labor2vec.tokens.documento import N_FONDO, Opciones
from labor2vec.tokens.vocabulario import Vocabulario
from labor2vec.utiles.rutas import CORPUS, T1

FIN = 239  # 2015-12
SIN_TOPE = 10**9


def _resumen(largos: np.ndarray) -> dict:
    p = np.percentile(largos, [50, 90, 99])
    return {"mediana": float(p[0]), "p90": float(p[1]), "p99": float(p[2]), "media": float(largos.mean()),
            "max": int(largos.max()), "sobre_2560": float((largos > 2560).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verificar", type=int, default=5000)
    ap.add_argument("--salida", type=Path, default=Path("informes/largos_grilla.json"))
    a = ap.parse_args()

    vocab = Vocabulario.cargar(CORPUS / "vocabulario.parquet", CORPUS / "ramas.parquet")
    base = Opciones(max_length=SIN_TOPE, fin_ventana=FIN)
    v0 = DatasetPretrain(CORPUS / "eventos_train", vocab, base, augmentar=False)
    v1 = DatasetPretrain(CORPUS / "eventos_train", vocab, Opciones(max_length=SIN_TOPE, fin_ventana=FIN, grilla=True),
                         augmentar=False)
    l0, l1 = v0.largos() - N_FONDO, v1.largos() - N_FONDO  # tokens de eventos (sin background)

    # exactitud de largos() frente al documento armado
    rng = np.random.default_rng(0)
    muestra = rng.choice(len(v1), size=min(a.verificar, len(v1)), replace=False)
    reales = np.array([len(v1.documento(int(k), v1._generador(int(k)), False)) for k in muestra])
    assert (reales == v1.largos()[muestra]).all(), "largos() no coincide con el documento"

    # meses sin registro por tipo: salida = entre el último registro y el fin de la ventana
    c = v1.corpus
    mes, off = c.col["mes_idx"].astype(np.int64), c.offsets
    idx = v1.indices
    primero, ultimo = mes[off[:-1]][idx], mes[v1.fin - 1][idx]
    huecos = v1._meses_sin_registro()[idx]
    salida = FIN - ultimo
    inter = huecos - salida
    meses_grilla = FIN - primero + 1
    out = {
        "trabajadores": int(len(idx)),
        "tokens_eventos_v0": _resumen(l0), "tokens_eventos_v1": _resumen(l1),
        "razon_mediana_v1_v0": float(np.median(l1) / np.median(l0)),
        "meses_grilla_mediana": float(np.median(meses_grilla)),
        "meses_sinreg_mediana": float(np.median(huecos)), "meses_sinreg_media": float(huecos.mean()),
        "proporcion_meses_sinreg": float(huecos.sum() / meses_grilla.sum()),
        "proporcion_meses_int": float(inter.sum() / meses_grilla.sum()),
        "proporcion_meses_sal": float(salida.sum() / meses_grilla.sum()),
        "trabajadores_con_algun_int": float((inter > 0).mean()), "trabajadores_con_sal_al_final": float((salida > 0).mean()),
        "trabajadores_que_empiezan_en_1996_01": float((primero == 0).mean()),
        "tokens_sinreg_sobre_total_v1": float(2 * huecos.sum() / l1.sum()),
        "verificados": int(len(muestra)),
    }

    # cohorte de T1: los 12 meses de 2015 registrados, de modo que no hay hueco final
    if T1.exists():
        ids = c.id_trabajador[idx]
        cohorte = np.isin(ids, np.array(list(leer_etiquetas(T1, "y_2019")), dtype=ids.dtype))
        out["cohorte_t1"] = {
            "trabajadores": int(cohorte.sum()), "tokens_eventos_v0": _resumen(l0[cohorte]),
            "tokens_eventos_v1": _resumen(l1[cohorte]),
            "proporcion_meses_sinreg": float(huecos[cohorte].sum() / meses_grilla[cohorte].sum()),
            "con_sal_al_final": float((salida[cohorte] > 0).mean()),
            "con_algun_int": float((inter[cohorte] > 0).mean())}

    a.salida.parent.mkdir(parents=True, exist_ok=True)
    a.salida.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
