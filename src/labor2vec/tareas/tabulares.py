"""Líneas de base tabulares de T1: tabla de vida (edad y sexo), regresión logística y FFNN sobre conteos de tokens.

Como en el life2vec original, la entrada es la proporción de cada token (sin especiales ni background) en el documento
que ve el transformer (misma ventana, mismo recorte a 2560), más sexo y edad (mín-máx entre 25 y 55). Todos se entrenan
con las clases balanceadas, así que sus probabilidades son comparables con las del transformer.

    python -m labor2vec.tareas.tabulares [--salida runs/t1_tab] [--semillas 3]

Escribe `runs/t1_tab/<modelo>/predicciones_{val,test}.parquet` con el mismo formato que `ajustar`.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from labor2vec.modelo.clasificador import perdida_balanceada
from labor2vec.modelo.transformer_utils import ScaleNorm, swish
from labor2vec.tareas import metricas
from labor2vec.tareas.datos import DatasetT1
from labor2vec.tokens.documento import N_FONDO, Opciones
from labor2vec.tokens.vocabulario import N_ESPECIALES, Vocabulario
from labor2vec.utiles.rutas import CORPUS, RAIZ, T1

EDAD_MIN, EDAD_MAX = 25, 55


def atributos(ds: DatasetT1, n_vocab: int) -> np.ndarray:
    """(n, n_vocab - N_ESPECIALES + 2): proporciones de tokens, sexo (1 = mujer) y edad normalizada."""
    x = np.zeros((len(ds), n_vocab - N_ESPECIALES + 2), dtype=np.float32)
    for k in range(len(ds)):
        t = ds[k]["input_ids"][0, N_FONDO:]
        t = t[t >= N_ESPECIALES] - N_ESPECIALES
        c = np.bincount(t, minlength=n_vocab - N_ESPECIALES).astype(np.float32)
        x[k, :-2] = c / max(c.sum(), 1.0)
    x[:, -2] = ds.sexo == 1
    x[:, -1] = (ds.edad - EDAD_MIN) / (EDAD_MAX - EDAD_MIN)
    return x


def cargar(corpus=CORPUS, ruta_t1=T1, etiqueta: str = "y_2019", fin_ventana: int = 239, max_length: int = 2560) -> dict:
    vocab = Vocabulario.cargar(Path(corpus) / "vocabulario.parquet", Path(corpus) / "ramas.parquet")
    op = Opciones(max_length=max_length, fin_ventana=fin_ventana)
    datos = {}
    for split in ("train", "val", "test"):
        ds = DatasetT1(Path(corpus) / f"eventos_{split}", vocab, op, False, ruta_t1, etiqueta)
        datos[split] = dict(x=atributos(ds, len(vocab)), y=ds.y, id=ds.id)
    return datos


def escribir(salida: Path, datos: dict, p_val, p_test, ruta_t1=T1) -> None:
    t = pq.read_table(ruta_t1, columns=["id_trabajador", "y_2019", "y_2021"]).to_pydict()
    fila = {i: k for k, i in enumerate(t["id_trabajador"])}
    salida.mkdir(parents=True, exist_ok=True)
    for split, p in (("val", p_val), ("test", p_test)):
        ids = datos[split]["id"]
        k = [fila[int(i)] for i in ids]
        pq.write_table(pa.table({"id_trabajador": ids, "y_2019": [t["y_2019"][j] for j in k],
                                 "y_2021": [t["y_2021"][j] for j in k], "p": np.asarray(p, dtype=np.float64)}),
                       salida / f"predicciones_{split}.parquet")


def logistica(datos: dict, columnas=None, C: float = 1.0):
    """Regresión logística L2 con clases balanceadas. `columnas` elige las entradas (tabla de vida: sexo y edad)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    cols = slice(None) if columnas is None else columnas
    esc = StandardScaler().fit(datos["train"]["x"][:, cols])
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(
        esc.transform(datos["train"]["x"][:, cols]), datos["train"]["y"])
    return [m.predict_proba(esc.transform(datos[s]["x"][:, cols]))[:, 1] for s in ("val", "test")]


class FFNN(nn.Module):
    """3 bloques Linear + Swish + ScaleNorm + AlphaDropout y una capa lineal a 2 clases (`src/models/ffn.py`)."""

    def __init__(self, entrada: int, ancho: int = 646, capas: int = 3, dropout: float = 0.35):
        super().__init__()
        bloques, n = [], entrada
        for _ in range(capas):
            bloques += [nn.Linear(n, ancho), _Swish(), ScaleNorm(ancho), nn.AlphaDropout(dropout)]
            n = ancho
        self.red = nn.Sequential(*bloques, nn.Linear(ancho, 2))

    def forward(self, x):
        return self.red(x)


class _Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def ffnn(datos: dict, semilla: int = 0, epocas: int = 35, paciencia: int = 5, lote: int = 64, lr: float = 1.7e-4):
    """FFNN con muestreo balanceado, RAdam y parada temprana por AUL de validación. Devuelve (p_val, p_test)."""
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    xt, yt = torch.from_numpy(datos["train"]["x"]), torch.from_numpy(datos["train"]["y"])
    red = FFNN(xt.shape[1])
    opt = torch.optim.RAdam(red.parameters(), lr=lr, weight_decay=1e-4, eps=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, 0.9)
    w = (1.0 / np.maximum(np.bincount(datos["train"]["y"], minlength=2), 1))[datos["train"]["y"]]
    w /= w.sum()

    def predecir(split):
        red.eval()
        with torch.no_grad():
            return torch.softmax(red(torch.from_numpy(datos[split]["x"])), -1)[:, 1].numpy()

    mejor, mejor_estado, sin_mejora = -1.0, None, 0
    for _ in range(epocas):
        red.train()
        orden = rng.choice(len(yt), size=9600, p=w)  # una «época» de 9.600 secuencias, como en el original
        for a in range(0, len(orden), lote):
            idx = torch.from_numpy(orden[a: a + lote])
            opt.zero_grad()
            perdida_balanceada(red(xt[idx]), yt[idx]).backward()
            torch.nn.utils.clip_grad_value_(red.parameters(), 1.0)
            opt.step()
        sched.step()
        a = metricas.aul(datos["val"]["y"], predecir("val"))
        if a > mejor + 0.001:
            mejor, sin_mejora = a, 0
            mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(mejor_estado)
    return predecir("val"), predecir("test")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--salida", default="runs/t1_tab")
    ap.add_argument("--semillas", type=int, default=3)
    ap.add_argument("--etiqueta", default="y_2019")
    args = ap.parse_args(argv)
    salida = Path(args.salida) if Path(args.salida).is_absolute() else RAIZ / args.salida
    datos = cargar(etiqueta=args.etiqueta)
    print({s: (len(d["y"]), float(d["y"].mean())) for s, d in datos.items()})
    escribir(salida / "tabla_vida", datos, *logistica(datos, columnas=[-2, -1]))
    escribir(salida / "logistica", datos, *logistica(datos))
    for s in range(args.semillas):
        p = ffnn(datos, s)
        escribir(salida / f"ffnn_s{s}", datos, *p)
        print(f"ffnn semilla {s}: AUL val {metricas.aul(datos['val']['y'], p[0]):.4f}")
    for nombre in ("tabla_vida", "logistica"):
        t = pq.read_table(salida / nombre / "predicciones_val.parquet").to_pydict()
        print(nombre, f"AUL val {metricas.aul(np.array(t['y_2019']), np.array(t['p'])):.4f}")


if __name__ == "__main__":
    main()
