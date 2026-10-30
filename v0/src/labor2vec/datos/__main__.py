"""Corre toda la fase 1 en orden: ingesta, descriptores, población y splits, salario relativo,
validación contra los cuadros oficiales y reporte de calidad.

    python -m labor2vec.datos [--forzar-ingesta]
"""

import argparse
import time

from labor2vec.datos import descriptores, ingesta, poblacion, qa, salario, validacion
from labor2vec.utiles.rutas import INFORMES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--forzar-ingesta", action="store_true", help="rehacer panel_base desde el CSV")
    args = ap.parse_args()
    t0 = time.time()

    def etapa(nombre):
        print(f"\n== {nombre} ({time.time() - t0:.0f} s)", flush=True)

    etapa("ingesta")
    ingesta.ingerir(args.forzar_ingesta)
    etapa("descriptores")
    descriptores.verificar(descriptores.construir())
    etapa("población y splits")
    poblacion.construir()
    etapa("salario relativo")
    salario.construir()
    etapa("validación")
    INFORMES.mkdir(exist_ok=True)
    (INFORMES / "validacion_mler.md").write_text(validacion.informe(validacion.validar()))
    etapa("reporte de calidad")
    (INFORMES / "qa_mler.md").write_text(qa.generar())
    print(f"\nfase 1 completa en {time.time() - t0:.0f} s; informes en {INFORMES}")


if __name__ == "__main__":
    main()
