"""Líneas de base tabulares de T1-T4: tabla de vida (edad y sexo), regresión logística y FFNN sobre conteos de tokens, y
(en T2-T4) regresión logística con el estado actual del trabajador.

Como en el life2vec original, la entrada es la proporción de cada token (sin especiales ni background) en el documento
que ve el transformer (misma ventana, mismo recorte a 2560), más sexo y edad (mín-máx entre 25 y 55). Todos se entrenan
con las clases balanceadas, así que sus probabilidades son comparables con las del transformer. La «logística con estado
actual» usa sólo edad, sexo y el estado de 2015 de la tarea (quintil salarial, letra o tramo de tamaño), una línea de base
barata que ya incorpora la persistencia.

    python -m labor2vec.tareas.tabulares [--tarea t1|t2|t3|t4] [--salida runs/t1_tab] [--semillas 3]
    python -m labor2vec.tareas.tabulares --tarea t4 --tiempo completo|sin_ultimo_valor   (fase 8b, `runs/t4_tabt/`)

Con `--tiempo` se agregan las variables de `tareas.tiempo` (antigüedad, tiempo desde el último cambio, cambios
recientes, huecos y tendencia salarial) y sólo se entrenan los modelos que las usan: `logistica_t`, `ffnn_t_s*` y, en
T2-T4, `logistica_estado_t` (edad, sexo, estado de 2015 y esas variables). Escribe `runs/<tarea>_tab/<modelo>/predicciones_{val,test}.parquet` con el mismo formato que `ajustar`.
"""

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn as nn

from labor2vec.ajustar import columnas_prediccion
from labor2vec.modelo.clasificador import perdida_balanceada
from labor2vec.modelo.transformer_utils import ScaleNorm, swish
from labor2vec.tareas import metricas
from labor2vec.tareas import tiempo as vars_tiempo
from labor2vec.tareas.datos import DatasetT1
from labor2vec.tokens.documento import N_FONDO, Opciones
from labor2vec.tokens.vocabulario import N_ESPECIALES, Vocabulario
from labor2vec.utiles.rutas import CORPUS, RAIZ, T1, TAREAS

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


# columna con el estado de 2015 en la tabla de cada tarea (línea de base de «estado actual»)
ESTADO = {"t2": "q2015", "t3": "letra2015", "t4": "tramo2015", "t2c": "q2015", "t3d": "letra2015", "t4c": "tramo2015"}


def variables_de_tiempo(split: str, ids: np.ndarray, corpus=CORPUS, ruta_t1=T1, fin_ventana: int = 239) -> np.ndarray:
    """Variables de `tareas.tiempo` (sin transformar) de `ids`, en ese orden. Se calculan una vez para toda la cohorte de
    T1 del split y se guardan en `data/processed/tareas/tiempo_<split>.npz`."""
    cache = TAREAS / f"tiempo_{split}_v{fin_ventana}.npz"
    if cache.exists():
        z = np.load(cache)
        todos, x = z["id"], z["x"]
    else:
        cohorte = pq.read_table(ruta_t1, columns=["id_trabajador"])["id_trabajador"].to_numpy()
        todos, x = vars_tiempo.calcular(Path(corpus) / f"eventos_{split}", cohorte, fin_ventana)
        np.savez(cache, id=todos, x=x)
    fila = {int(i): k for k, i in enumerate(todos)}
    return x[[fila[int(i)] for i in ids]]


def cargar(corpus=CORPUS, ruta_t1=T1, etiqueta: str = "y_2019", fin_ventana: int = 239, max_length: int = 2560,
           estado: str | None = None, atributos_evento: str = "todos", tiempo: str | None = None) -> dict:
    """Atributos, etiquetas e ids de train/val/test. Con `estado` agrega su codificación one-hot (categorías de train);
    con `tiempo` (`completo` o `sin_ultimo_valor`), las variables de tiempo, transformadas con la media y el desvío de
    train, en `datos[split]["tiempo"]`."""
    vocab = Vocabulario.cargar(Path(corpus) / "vocabulario.parquet", Path(corpus) / "ramas.parquet")
    op = Opciones(max_length=max_length, fin_ventana=fin_ventana, atributos=atributos_evento)
    datos = {}
    for split in ("train", "val", "test"):
        ds = DatasetT1(Path(corpus) / f"eventos_{split}", vocab, op, False, ruta_t1, etiqueta)
        datos[split] = dict(x=atributos(ds, len(vocab)), y=ds.y, id=ds.id)
    datos["n_clases"] = int(datos["train"]["y"].max()) + 1
    if tiempo:
        col = vars_tiempo.indices(tiempo)
        crudo = {s: variables_de_tiempo(s, datos[s]["id"], corpus, ruta_t1, fin_ventana)[:, col] for s in datos
                 if s != "n_clases"}
        # la transformación mira todas las columnas de `NOMBRES`: se rellena y se descarta lo que no se usa
        def plano(x):
            z = np.zeros((len(x), len(vars_tiempo.NOMBRES)))
            z[:, col] = x
            return z
        _, media, desvio = vars_tiempo.transformar(plano(crudo["train"]))
        for s, x in crudo.items():
            datos[s]["tiempo"] = vars_tiempo.transformar(plano(x), media, desvio)[0][:, col]
    if estado:
        t = pq.read_table(ruta_t1, columns=["id_trabajador", estado]).to_pydict()
        valor = dict(zip(t["id_trabajador"], t[estado]))
        categorias = sorted({valor[int(i)] for i in datos["train"]["id"]})
        for split in ("train", "val", "test"):
            e = np.array([valor[int(i)] for i in datos[split]["id"]])
            datos[split]["estado"] = (e[:, None] == np.array(categorias)[None, :]).astype(np.float32)
    return datos


def escribir(salida: Path, datos: dict, p_val, p_test, ruta_t1=T1) -> None:
    tabla = pq.read_table(ruta_t1)
    cols = [c for c in tabla.column_names if c == "y" or c.startswith("y_")]
    t = tabla.select(["id_trabajador", *cols]).to_pydict()
    fila = {i: k for k, i in enumerate(t["id_trabajador"])}
    salida.mkdir(parents=True, exist_ok=True)
    for split, p in (("val", p_val), ("test", p_test)):
        ids = datos[split]["id"]
        k = [fila[int(i)] for i in ids]
        pq.write_table(pa.table({"id_trabajador": ids, **{c: [t[c][j] for j in k] for c in cols},
                                 **columnas_prediccion(p, t)}), salida / f"predicciones_{split}.parquet")


def _entrada(datos: dict, split: str, modo: str) -> np.ndarray:
    x = datos[split]["x"]
    if modo == "conteos":
        return x
    if modo == "conteos_tiempo":
        return np.concatenate([x, datos[split]["tiempo"]], 1)
    base = x[:, [-2, -1]]  # sexo y edad
    if modo == "edad_sexo":
        return base
    if modo == "estado_tiempo":
        return np.concatenate([base, datos[split]["estado"], datos[split]["tiempo"]], 1)
    return np.concatenate([base, datos[split]["estado"]], 1)


def logistica(datos: dict, modo: str = "conteos", C: float = 1.0):
    """Regresión logística L2 con clases balanceadas (multinomial con más de 2 clases). `modo` elige las entradas:
    `conteos` (proporciones de tokens, sexo y edad), `edad_sexo` (tabla de vida), `estado` (edad, sexo y estado de 2015) y,
    con variables de tiempo, `conteos_tiempo` y `estado_tiempo`."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    esc = StandardScaler().fit(_entrada(datos, "train", modo))
    m = LogisticRegression(C=C, class_weight="balanced", max_iter=1000).fit(
        esc.transform(_entrada(datos, "train", modo)), datos["train"]["y"])
    p = [m.predict_proba(esc.transform(_entrada(datos, s, modo))) for s in ("val", "test")]
    return [q[:, 1] if datos["n_clases"] == 2 else q for q in p]


class FFNN(nn.Module):
    """3 bloques Linear + Swish + ScaleNorm + AlphaDropout y una capa lineal a 2 clases (`src/models/ffn.py`)."""

    def __init__(self, entrada: int, ancho: int = 646, capas: int = 3, dropout: float = 0.35, n_clases: int = 2):
        super().__init__()
        bloques, n = [], entrada
        for _ in range(capas):
            bloques += [nn.Linear(n, ancho), _Swish(), ScaleNorm(ancho), nn.AlphaDropout(dropout)]
            n = ancho
        self.red = nn.Sequential(*bloques, nn.Linear(ancho, n_clases))

    def forward(self, x):
        return self.red(x)


class _Swish(nn.Module):
    def forward(self, x):
        return swish(x)


def ffnn(datos: dict, semilla: int = 0, epocas: int = 35, paciencia: int = 5, lote: int = 64, lr: float = 1.7e-4,
         monitor: str = "aul", modo: str = "conteos"):
    """FFNN con muestreo balanceado, RAdam y parada temprana por la métrica `monitor` de validación (`aul`, o `qwk` o
    `auc_macro` en multiclase). `modo` elige las entradas como en `logistica`. Devuelve (p_val, p_test)."""
    k = datos["n_clases"]
    torch.manual_seed(semilla)
    rng = np.random.default_rng(semilla)
    xs = {sp: np.ascontiguousarray(_entrada(datos, sp, modo), dtype=np.float32) for sp in ("train", "val", "test")}
    xt, yt = torch.from_numpy(xs["train"]), torch.from_numpy(datos["train"]["y"])
    red = FFNN(xt.shape[1], n_clases=k)
    opt = torch.optim.RAdam(red.parameters(), lr=lr, weight_decay=1e-4, eps=1e-6)
    sched = torch.optim.lr_scheduler.StepLR(opt, 1, 0.9)
    w = (1.0 / np.maximum(np.bincount(datos["train"]["y"], minlength=2), 1))[datos["train"]["y"]]
    w /= w.sum()

    def predecir(split):
        red.eval()
        with torch.no_grad():
            p = torch.softmax(red(torch.from_numpy(xs[split])), -1).numpy()
            return p[:, 1] if k == 2 else p

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
        pv = predecir("val")
        a = metricas.aul(datos["val"]["y"], pv) if k == 2 else metricas.multiclase(datos["val"]["y"], pv)[monitor]
        if a > mejor + 0.001:
            mejor, sin_mejora = a, 0
            mejor_estado = {k: v.clone() for k, v in red.state_dict().items()}
        else:
            sin_mejora += 1
            if sin_mejora >= paciencia:
                break
    red.load_state_dict(mejor_estado)
    return predecir("val"), predecir("test")


MONITOR = {"t1": "aul", "t2": "qwk", "t3": "aul", "t4": "auc_macro", "t2c": "auc_macro", "t3d": "auc_macro", "t4c": "aul"}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tarea", default="t1", choices=list(MONITOR))
    ap.add_argument("--atributos", default="todos", choices=["todos", "sin_ultimo_valor"],
                    help="sin_ultimo_valor: el documento sin rama, provincia ni antigüedad (fase 8)")
    ap.add_argument("--tiempo", choices=["completo", "sin_ultimo_valor"],
                    help="agrega las variables de tiempo (fase 8b); con sin_ultimo_valor también quita esos atributos")
    ap.add_argument("--salida")
    ap.add_argument("--semillas", type=int, default=3)
    args = ap.parse_args(argv)
    t = args.tarea
    atributos = "sin_ultimo_valor" if args.tiempo == "sin_ultimo_valor" else args.atributos
    nombre = f"{t}_{'uv_' if atributos != 'todos' else ''}tab{'t' if args.tiempo else ''}"
    salida = Path(args.salida or f"runs/{nombre}")
    salida = salida if salida.is_absolute() else RAIZ / salida
    ruta, etiqueta = (T1, "y_2019") if t == "t1" else (TAREAS / f"{t}.parquet", "y")
    datos = cargar(ruta_t1=ruta, etiqueta=etiqueta, estado=ESTADO.get(t), atributos_evento=atributos,
                   tiempo=args.tiempo)
    print({s: (len(d["y"]), np.bincount(d["y"]).tolist()) for s, d in datos.items() if s != "n_clases"})
    if args.tiempo:
        escribir(salida / "logistica_t", datos, *logistica(datos, "conteos_tiempo"), ruta)
        if t in ESTADO:
            escribir(salida / "logistica_estado_t", datos, *logistica(datos, "estado_tiempo"), ruta)
        for s in range(args.semillas):
            escribir(salida / f"ffnn_t_s{s}", datos, *ffnn(datos, s, monitor=MONITOR[t], modo="conteos_tiempo"), ruta)
            print(f"ffnn con tiempo, semilla {s} listo")
        print("listo")
        return
    escribir(salida / "tabla_vida", datos, *logistica(datos, "edad_sexo"), ruta)
    escribir(salida / "logistica", datos, *logistica(datos), ruta)
    if t in ESTADO:
        escribir(salida / "logistica_estado", datos, *logistica(datos, "estado"), ruta)
    for s in range(args.semillas):
        p = ffnn(datos, s, monitor=MONITOR[t])
        escribir(salida / f"ffnn_s{s}", datos, *p, ruta)
        print(f"ffnn semilla {s} listo")
    print("listo")


if __name__ == "__main__":
    main()
