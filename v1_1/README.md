# labor2vec v1.1

v1 con el pre-entrenamiento corregido. Los documentos son los de v1 (grilla mensual con los meses sin empleo registrado como `SINREG_INT` y `SINREG_SAL`); cambia cómo se entrena sobre ellos. Plan en `../plans/PLAN_v1_1.md`; motivo y evidencia en `../v1/informes/fase3_evaluacion.md`; informes de v1.1 en `informes/`. `v1/` queda congelada.

## Qué cambia respecto de v1

Tres opciones de `datos` (valor por defecto = v1; `configs/preentrenamiento.yaml` y `humo_cpu.yaml` las activan):

- `sop_sinreg_fijo: true`: el SOP permuta sólo los meses con empleo; los `SINREG` quedan en su lugar (si no, una salida antes de un empleo delataba la alteración) y una alteración que deja el documento igual no cuenta como alterado.
- `mlm_sinreg: entrada`: los `SINREG` se ven pero no se tapan ni se usan de reemplazo al azar; la línea de base unigrama no les da masa.
- `mask_base: sin_sinreg`: se tapa el 30 % del largo sin los bloques `SINREG`, que es la cantidad del documento de v0 de la misma persona. Con la opción anterior, v0 y v1.1 tapan los mismos casos.

Además, la validación registra la pérdida, la perplejidad y el acierto del MLM por grupo (`val/mlm_loss_empleo`, `_fondo` y `_sinreg` si hay objetivos). Las configuraciones `ajuste_*` y `gru_*` leen el modelo de `~/labor2vec/v1_1/runs/pre_150ep/`.

## Lo que viene de v1

## Qué comparte con v0 (para no duplicar disco)

- **Datos:** `data/processed` es un enlace simbólico a `../../v0/data/processed` (mismo panel, mismo corpus de eventos, mismas cohortes). Los huecos se derivan al armar el documento, así que el corpus no se regenera. Lo que genera v1 va a `data/` y `runs/`.
- **Entorno:** no tiene `.venv` propio. Se usa el de v0 con el código de esta carpeta delante:

  ```bash
  cd v1_1
  PYTHONPATH=src ../v0/.venv/bin/python -m pytest -q          # pruebas
  PYTHONPATH=src ../v0/.venv/bin/python -m labor2vec.entrenar --config configs/humo_cpu.yaml
  ```

- **Mendieta:** `scripts/cluster/sincronizar_mendieta.sh subir` copia el código a `~/labor2vec/v1_1`, y `... datos` crea allí el enlace a `~/labor2vec/v0/data/processed`. Los jobs usan el tar de entorno de v0 (`entorno_scratch.sh` antepone `~/labor2vec/v1_1/src` con `PYTHONPATH`).

## Qué cambia respecto de v0

- `Opciones.grilla` (`datos.grilla` en los YAML; las configuraciones de v1 traen `true`). Con `false` se obtiene exactamente la codificación de v0, que sirve de control.
- `Opciones.sinreg`: `"dos"` (por defecto) o `"uno"` (un solo token; control de la fase 5 del plan).
- Vocabulario de 521 tokens: los 519 de v0 y, detrás, `SINREG_INT` y `SINREG_SAL`. No se guardan en `vocabulario.parquet`.
- `ANT_k` de v0 pasa a `EMPINI_k` (categoría `inicio_empresa`): es el inicio de actividades de la empresa (`tr_anio_ini`), no la antigüedad del trabajador. El archivo compartido conserva los nombres de v0; `Vocabulario.cargar` los traduce y `guardar` los deshace.
- El SOP permuta sólo entre bloques del mismo largo (con la grilla los bloques miden 6 o 2 tokens).
- Las configuraciones `ajuste_*` y `gru_*` leen el modelo pre-entrenado de `~/labor2vec/v1_1/runs/pre_150ep/` (el de v1.1).
- Los scripts de entorno (`entorno_mendieta.sh`, `empaquetar_entorno.sh`) quedaron en `v0/scripts/cluster/`.

## Pendiente de v0 que no se adaptó todavía

`analisis/espacio.py`, `analisis/personas.py` y `scripts/cluster/fase9_encolar.sh` siguen pensados para la codificación de v0 y para corridas que sólo existen en v0 (fase 6 del plan de v1).
