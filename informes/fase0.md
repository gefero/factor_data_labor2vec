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

Job `job2vec_prueba` (1180374): partición `short`, nodo `ivb14`, NVIDIA A30 (24 GB, driver 580.178.04, capacidad 8.0). Terminó `COMPLETED` en 1 min 30 s.

| Etapa | Tiempo |
|---|---|
| Copia del entorno (6,4 GB) al `/scratch` | 34 s |
| `import torch` | 4 s |
| Matmul fp32 4096² | 6,6 TFLOP/s |

Configuración del paper (8,50 M parámetros, secuencias de 2560 tokens sin relleno, datos sintéticos), 20 pasos por medición:

| Lote | fp32: pasos/s | fp32: secuencias/s | fp32: memoria pico | bf16: pasos/s | bf16: secuencias/s | bf16: memoria pico |
|---|---|---|---|---|---|---|
| 4 | 6,70 | 26,8 | 3,8 GiB | 11,62 | 46,5 | 2,4 GiB |
| 8 | 3,54 | 28,3 | 7,4 GiB | 7,05 | 56,4 | 4,7 GiB |
| 16 | 1,83 | 29,3 | 14,7 GiB | 3,83 | 61,3 | 9,3 GiB |
| 32 | sin memoria | — | — | 1,97 | 63,2 | 18,4 GiB |

`seff`: 7 % de eficiencia de CPU (esperable en una prueba que no carga datos) y 14,2 GB de memoria del nodo (casi todo es el caché de archivos del `.tar`, ver lecciones §10).

### Lectura

- **bf16 duplica el rendimiento** (~60 contra ~29 secuencias/s) y usa ~35 % menos memoria. Desde lote 8 el rendimiento casi no crece: la GPU ya está saturada.
- **Estimación del pre-entrenamiento con el esquema del paper** (30 épocas × 30.000 secuencias = 900 k secuencias de 2560 tokens): ~4 h en bf16 o ~9 h en fp32, en una A30. Es una cota superior: casi todas las secuencias de la MLER son mucho más cortas (mediana ≈ 280 tokens). Con relleno dinámico o `max_length` más chico el costo bajaría varias veces.
- Una pasada completa sobre el train de la MLER (~400 k trabajadores) con largo 2560 llevaría ~2 h en bf16.
- **Pendiente**:
  - validar que bf16 no degrade la convergencia del Performer con datos reales, comparando contra fp32 en una corrida corta de la fase 3;
  - medir si la carga de datos (10 cores) alcanza para alimentar ~60 secuencias/s.

## Criterio de aceptación

1. `uv sync` + `pytest` en local: ✔ (4 pruebas).
2. Job de prueba `COMPLETED` en Mendieta con las mediciones: ✔.
3. Esta nota: ✔.

**Fase 0 cerrada.**
