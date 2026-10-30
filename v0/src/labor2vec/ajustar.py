"""Ajuste fino de T1-T4 (transformer pre-entrenado, transformer desde cero o GRU) y predicciones de val y test.

    python -m labor2vec.ajustar --config configs/ajuste_t1.yaml [--set a.b=valor ...]
                                [--reanudar] [--max-minutos N] [--max-pasos N] [--salida DIR]

Reutiliza de `entrenar` la lectura de configuración y los callbacks (corte por tiempo, guardia de NaN, rendimiento). La
tarea sale del YAML: `datos.t1` (tabla de etiquetas), `datos.etiqueta`, `ajuste.n_clases`, `ajuste.perdida` y
`entrenamiento.monitor` (`aul` en las binarias, `qwk` o `auc_macro` en las multiclase).

Al terminar escribe en la carpeta de salida `modelo_final.pt` (el mejor por la métrica de monitoreo de validación) y
`predicciones_{val,test}.parquet` con `id_trabajador`, las etiquetas (`y_2019` e `y_2021` en T1, `y` en T2-T4) y `p`
(probabilidad de la clase 1) o `p_0..p_{K-1}`. Con `--reanudar` sigue desde `last.ckpt`.
"""

import argparse
import json
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger

from labor2vec.entrenar import GuardiaNaN, ParadaSegura, Rendimiento, _ruta, cargar_config
from labor2vec.modelo.clasificador import ClasificadorT1
from labor2vec.tareas.datos import DataModuleT1
from labor2vec.tokens.documento import opciones_de


def hp_encoder(cfg: dict) -> dict:
    ruta = Path(cfg["preentrenado"]).expanduser()
    return torch.load(ruta, map_location="cpu", weights_only=False)["hparams"]


def preparar(cfg: dict, epoca_inicial: int = 0, pre_cargado: bool = True):
    d, ft = cfg["datos"], cfg.get("ajuste", {})
    op = opciones_de(d)
    dm = DataModuleT1(_ruta(d["corpus"]), None, op, _ruta(d["t1"]), d["etiqueta"], d["tamano_lote"],
                      d["num_workers"], cfg["semilla"], epoca_inicial)
    dm.setup()
    modelo = ClasificadorT1(hp_encoder(cfg), ft)
    if pre_cargado and ft.get("tipo", "transformer") == "transformer" and not ft.get("desde_cero", False):
        modelo.cargar_preentrenado(Path(cfg["preentrenado"]).expanduser())
    return dm, modelo


def columnas_prediccion(p, columnas_y: dict) -> dict:
    """Columnas de predicción: `p` con 2 clases, `p_0..p_{K-1}` con más (p es la matriz (n, K))."""
    p = np.asarray(p, dtype=np.float64)
    return {"p": p} if p.ndim == 1 else {f"p_{k}": p[:, k] for k in range(p.shape[1])}


def escribir_predicciones(modelo, dm, salida: Path, ruta_t1) -> None:
    """Predicciones de val y test con las etiquetas de la tabla de la tarea (`y`, o `y_2019` e `y_2021` en T1)."""
    tabla = pq.read_table(ruta_t1)
    cols = [c for c in tabla.column_names if c == "y" or c.startswith("y_")]
    t = tabla.select(["id_trabajador", *cols]).to_pydict()
    fila = {i: k for k, i in enumerate(t["id_trabajador"])}
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo.to(dispositivo)
    for split in ("val", "test"):
        r = modelo.predecir(getattr(dm, f"{split}_dataloader")(), dispositivo)
        k = [fila[int(i)] for i in r["id"]]
        pq.write_table(pa.table({"id_trabajador": r["id"], **{c: [t[c][j] for j in k] for c in cols},
                                 **columnas_prediccion(r["p"], t)}), salida / f"predicciones_{split}.parquet")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", default="configs/ajuste_t1.yaml")
    ap.add_argument("--set", action="append", default=[], dest="cambios", metavar="CLAVE=VALOR")
    ap.add_argument("--reanudar", action="store_true")
    ap.add_argument("--max-minutos", type=float)
    ap.add_argument("--max-pasos", type=int)
    ap.add_argument("--salida")
    args = ap.parse_args(argv)

    cfg = cargar_config(_ruta(args.config), args.cambios)
    e = cfg["entrenamiento"]
    salida = _ruta(args.salida or e["salida"])
    salida.mkdir(parents=True, exist_ok=True)
    last = salida / "last.ckpt"
    retomar = args.reanudar and last.exists()
    epoca_inicial = torch.load(last, map_location="cpu", weights_only=False)["epoch"] if retomar else 0
    pl.seed_everything(cfg["semilla"], workers=True)
    dm, modelo = preparar(cfg, epoca_inicial)
    (salida / "config.json").write_text(json.dumps(cfg, indent=2, default=str))

    acumular = e.get("acumular", 1)
    monitor = f"val/{e.get('monitor', 'aul')}"  # aul (binarias), qwk o auc_macro (multiclase)
    mejor = ModelCheckpoint(salida, filename="mejor", monitor=monitor, mode="max", save_top_k=1)
    parada = ParadaSegura(last, args.max_minutos * 60 if args.max_minutos else None, args.max_pasos)
    trainer = pl.Trainer(
        min_epochs=e["epocas_min"], max_epochs=e["epocas_max"], precision=e["precision"], accelerator="auto", devices=1,
        accumulate_grad_batches=acumular, limit_train_batches=e["pasos_por_epoca"] * acumular,
        gradient_clip_val=e.get("clip_gradiente"), gradient_clip_algorithm="value",
        log_every_n_steps=e["log_cada_pasos"], num_sanity_val_steps=0, enable_progress_bar=False,
        callbacks=[ModelCheckpoint(salida, save_last=True, every_n_train_steps=e["guardar_cada_pasos"], save_top_k=0),
                   mejor, EarlyStopping(monitor, mode="max", patience=e["paciencia"], min_delta=0.001),
                   parada, GuardiaNaN(), Rendimiento(e["log_cada_pasos"]), LearningRateMonitor("epoch")],
        logger=CSVLogger(salida, name="logs"), default_root_dir=salida)
    t0 = time.time()
    trainer.fit(modelo, datamodule=dm, ckpt_path=str(last) if retomar else None)
    if not parada.paro:
        trainer.save_checkpoint(last)
        if mejor.best_model_path:
            modelo = ClasificadorT1.load_from_checkpoint(mejor.best_model_path)
        torch.save({"state_dict": modelo.state_dict(), "hparams": dict(modelo.hparams)}, salida / "modelo_final.pt")
        escribir_predicciones(modelo, dm, salida, _ruta(cfg["datos"]["t1"]))
    print(f"listo en {time.time() - t0:.0f} s; época {trainer.current_epoch}, paso {trainer.global_step}; {salida}")


if __name__ == "__main__":
    main()
