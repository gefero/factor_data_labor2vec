# Fase 3: script de entrenamiento y prueba de humo en CPU

Generado el 2026-10-16. Código: `src/labor2vec/entrenar.py`, `src/labor2vec/tokens/linea_base.py`, métricas en `modelo/pretrain.py`; configuración en `configs/preentrenamiento.yaml` (real) y `configs/humo_cpu.yaml` (prueba). Pruebas: `tests/test_entrenar.py` (6 nuevas, 37 en total).

## Qué se agregó

- **Métricas** (train cada 10 pasos, validación por época): pérdidas total, MLM y SOP; exactitud y perplejidad del MLM sobre las posiciones enmascaradas (el relleno no cuenta); exactitud de SOP y recall por clase.
- **Línea de base unigrama**: log-probabilidades de los tokens no especiales según 20.000 documentos de train; se registra `ppl_unigrama` sobre los mismos objetivos que el MLM.
- **Script** `python -m labor2vec.entrenar`: configuración YAML con `--set clave=valor`, checkpoints (`last.ckpt` cada N pasos y `mejor.ckpt` por pérdida MLM de validación), `--reanudar`, `--max-minutos` y `--max-pasos`. Registro en CSV (una carpeta `version_N` por ejecución, así reanudar no pisa el historial).
- El corte por tiempo o por paso guarda el checkpoint **desde dentro del bucle**: guardar después de `fit` registraba la época ya incrementada, y al reanudar Lightning habría dado por terminada una época a medias.
- DataModule: validación y test con orden mezclado pero fijo (un subconjunto de lotes es siempre el mismo y representativo), reanudación con el orden de lotes que correspondía y `mask_ratio` configurable en el `collate`.

## Prueba de humo (modelo chico, muestra del 1 %)

Modelo de 35,7 mil parámetros (`hidden_size=32`, 2 capas), vocabulario real (519 tokens), lote de 32, 3.480 trabajadores de train con ventana a 2015-12 (108 pasos por época), `fp32`.

| Corrida | Pasos | Tiempo | Pérdida (primeros 50 → últimos 50) | Perplejidad MLM val | Unigrama | Exactitud MLM val |
|---|---|---|---|---|---|---|
| 5 épocas | 540 | 2 min | 4,10 → 3,59 (88 %) | 55,4 | 78,7 | 18,6 % |
| 15 épocas | 1.620 | 6 min | 4,24 → 3,19 (**75 %**) | **32,3** | 78,7 | 22,6 % |

Perplejidad MLM de validación de la corrida de 15 épocas: 84,7 (época 0), 79,1, 69,1, 52,9, 44,8 (época 4), …, 33,0 (época 10), 32,3 (época 12), 32,2, 32,3 (época 14). Pasa por debajo de la unigrama en la época 3 y se estabiliza hacia la época 12.

### Criterios de aceptación

| # | Criterio | Resultado |
|---|---|---|
| 1 | Pérdida de los últimos 50 pasos < 80 % de la de los primeros 50 | **Cumple en 1.620 pasos (75 %).** En 540 pasos da 88 %: el calentamiento de OneCycle (5 %) y 5 épocas sobre 3.480 trabajadores son pocas. |
| 2 | Perplejidad MLM de validación por debajo de la unigrama | **Cumple**: 32,3 contra 78,7 (la corrida de 540 pasos también: 55,4). |
| 3 | Sin NaN ni explosión | **Cumple.** La pérdida máxima fue 4,4 en todas las corridas. |
| 4 | Reanudar continúa la misma curva | **Cumple.** Corte en el paso 270 y reanudación: la pérdida pasa de 3,80 (paso 269) a 3,80 (paso 279), y la perplejidad final es 55,6 contra 55,4 sin interrupción (la pequeña diferencia viene del orden de lotes y del muestreo aleatorio tras reanudar). Además, una prueba automática comprueba que cortar a mitad de época y reanudar llega exactamente al paso final. |
| 5 | Lotes idénticos con `num_workers > 0` y la misma semilla | **Cumple** (prueba automática con 0 y 2 workers). |

### Modelo del paper en CPU (sólo estructura)

Configuración del paper (8,5 M de parámetros), lote de 8, `max_length=512`, 40 pasos: 77 s en total y 2,6 GB de RAM pico. Corre sin errores y la pérdida no explota (≈4,4); no se espera aprendizaje en 40 pasos. Se acotó el lote y el largo a propósito: con lote 16 y 2.560 tokens la memoria sería de ~15 GB, como en la GPU (fase 0).

## Hallazgos para la fase 4

1. **SOP no aprende en estas pruebas.** La pérdida de SOP se queda en ≈1,58 y el recall de la clase «orden normal» es 0 en todas las épocas: con pesos de clase 1/p (1,1 / 20 / 20) el modelo predice las clases minoritarias y la exactitud de SOP (≈4-5 %) no sirve como indicador. Puede deberse al modelo y al entrenamiento minúsculos, pero conviene vigilarlo en la corrida real. Si en la fase 4 sigue plano, las opciones son un peso menor para las clases raras o más peso/tiempo al SOP; no se cambió nada todavía.
2. **Calendario OneCycle**: con menos de ~40 pasos totales la fase de calentamiento (5 %) queda de largo cero y `OneCycleLR` falla por división por cero. No afecta corridas reales.
3. La época se definió como una pasada completa por train (`steps_per_epoca = len(sampler)`); con la ventana a 2015-12 y lote de 64 son ~5.400 pasos por época. El número de épocas y el lote se fijan en la fase 4 con los tiempos de la A30.

## Pendiente de la fase 3

La prueba corta en Mendieta (plan 3.5: bf16 contra fp32, pasos/s con largo variable, memoria con lote 64) requiere un OK aparte antes de encolar.
