"""Predicciones de modelos ya ajustados con un control temporal sobre la entrada (fase 8; plans/PLAN_fase_8.md). No entrena.

    python -m labor2vec.tareas.predecir --config configs/ajuste_t3.yaml --corridas runs/t3_l2v_s0 runs/t3_l2v_s1 \\
        --controles ninguno tiempo orden

Para cada corrida (carpeta con `modelo_final.pt`) y cada control, escribe `predicciones_{val,test}.parquet` en
`<runs>/ctrl_<control>/<nombre de la corrida>/`, con el mismo formato que `ajustar`. Controles:
- `ninguno`: la entrada de siempre (sirve para comprobar que se reproducen las predicciones originales);
- `tiempo`: se permutan fechas y edades entre los eventos de cada persona (`Opciones.control_temporal`);
- `orden`: se permuta el orden de los eventos en la secuencia, cada uno con su fecha.

La configuración (`--config`) aporta los datos de la tarea (tabla de etiquetas, ventana, largo) y se puede pisar con `--set`.
Sin polars ni scipy: tiene que correr en los nodos de Mendieta.
"""

import argparse
import time

import torch

from labor2vec.ajustar import escribir_predicciones
from labor2vec.analisis.extraer import cargar_modelo
from labor2vec.entrenar import _ruta, cargar_config
from labor2vec.tareas.datos import DataModuleT1
from labor2vec.tokens.documento import opciones_de

CONTROLES = ("ninguno", "tiempo", "orden")


def datamodule(cfg: dict, control: str) -> DataModuleT1:
    d = {**cfg["datos"], "control_temporal": None if control == "ninguno" else control}
    dm = DataModuleT1(_ruta(d["corpus"]), None, opciones_de(d), _ruta(d["t1"]), d["etiqueta"], d["tamano_lote"],
                      d["num_workers"], cfg["semilla"])
    dm.setup()
    return dm


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--corridas", nargs="+", required=True, help="carpetas con modelo_final.pt")
    ap.add_argument("--controles", nargs="+", default=list(CONTROLES), choices=CONTROLES)
    ap.add_argument("--runs", default="runs", help="carpeta que contiene `ctrl_<control>/`")
    ap.add_argument("--set", action="append", default=[], dest="cambios", metavar="CLAVE=VALOR")
    args = ap.parse_args(argv)

    cfg = cargar_config(_ruta(args.config), args.cambios)
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    for control in args.controles:
        dm = datamodule(cfg, control)
        for corrida in args.corridas:
            origen = _ruta(corrida)
            modelo = cargar_modelo(origen / "modelo_final.pt", cfg, sin_cabeza=False).to(dispositivo)
            salida = _ruta(args.runs) / f"ctrl_{control}" / origen.name
            salida.mkdir(parents=True, exist_ok=True)
            escribir_predicciones(modelo, dm, salida, _ruta(cfg["datos"]["t1"]))
            print(f"{control:8s} {origen.name}: {time.time() - t0:.0f} s", flush=True)
    print("listo")


if __name__ == "__main__":
    main()
