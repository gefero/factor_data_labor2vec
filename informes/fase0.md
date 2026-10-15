# Fase 0: entorno, estructura y prueba en GPU

Fecha: 2026-10-15.

## Versiones

| | Local (devcontainer, CPU) | Mendieta (A30) |
|---|---|---|
| Python | 3.13.16 (uv, only-managed) | 3.13 (uv, x86-64 base) |
| torch | 2.11.0+cpu | 2.11.0 cu126 |
| lightning | 2.6.6 | ídem (uv.lock) |
| performer-pytorch | 1.1.4 | ídem |
| torchmetrics | 1.9.0 | ídem |

El Python 3.14 del sistema no se usa porque le falta el módulo `decimal`.

## Backbone

- Copiado de `carlomarxdk/life2vec-light` a `src/labor2vec/modelo/` (licencia MIT, con encabezado de atribución en cada archivo).
- Cambios: rutas de import, `pytorch_lightning` → `lightning.pytorch`, y calendario OneCycle parametrizable (`epochs`, `steps_per_epoch`; los valores por defecto son los originales).
- Con la configuración del paper (`HPARAMS_PAPER` en `modelo/config.py`, tomada de `conf/experiment/pretrain.yaml` del repo completo) el modelo tiene **8,50 M de parámetros**, contra 8,4 M en la tabla A2 del paper. La diferencia viene del vocabulario.

Pruebas (`pytest`, CPU), 4 en verde:
- formas de salida de MLM y SOP;
- sobreajuste de un lote fijo (la pérdida baja más del 20 % en 80 pasos);
- el relleno no altera las posiciones válidas;
- cantidad de parámetros con la configuración del paper.

## Valores fijos del código original que dependen del vocabulario (pendientes para la fase 2)

1. `embeddings.py` y `transformer.py` (decoder MLM): `ignore_index = [0, 4, 5, 6, 7, 8]`. Son los índices de los tokens especiales y de reserva que se excluyen del centrado de embeddings; dependen de cómo se arme nuestro vocabulario.
2. `embeddings.py`: `pos[:, :5] *= 0` anula edad y posición absoluta en los 5 primeros tokens. En el original son `[CLS]` + 3 tokens de background + `[SEP]`. En la MLER el background tiene 2 tokens (sexo, año de nacimiento), así que serían 4.
3. `pretrain.py`: SOP de 3 clases con pesos fijos `[1/0.8, 1/0.1, 1/0.1]`, que deben coincidir con las probabilidades que use la tarea.

## Medición en GPU

_Pendiente: job de prueba en Mendieta (`scripts/cluster/prueba_mendieta.sbatch`)._
