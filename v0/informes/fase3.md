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

## Prueba en la A30 de Mendieta (plan 3.5)

Job 1180666, partición `short`, nodo ivb04, NVIDIA A30 (24 GB), 10 cores; terminó `COMPLETED` en 9 min 55 s. Datos reales: 344.025 trabajadores de train (ventana a 2015-12), modelo con la configuración del paper y vocabulario de 519 tokens (8,1 M de parámetros), `max_length=2560`, variante A, 9 workers de carga.

**Un primer intento (job 1180509) no corrió nada**: todos los pasos murieron con `Illegal instruction` aunque el job figuró `COMPLETED` (el script no cortaba al primer error). La causa era `import polars`: su binario necesita AVX2 y los nodos y la cabecera de Mendieta (Ivy Bridge) no la tienen. `vocabulario.py` ahora usa pyarrow, una prueba impide que el camino de entrenamiento importe polars, y el sbatch empieza con una comprobación de importaciones que aborta si falla.

### Carga de datos

| Workers | Secuencias/s | Tokens/s (con relleno) |
|---|---|---|
| 9 (lote 32) | ~10.000 | 4,6 M |

La carga entrega 40 veces más de lo que consume la GPU (240 secuencias/s como máximo, ver abajo): 10 cores alcanzan de sobra.

### bf16 contra fp32 (misma semilla, 1.000 pasos, lote 16)

| | Pérdida de train (últimos 10 pasos) | Perplejidad MLM val (paso 999) | Exactitud MLM val | Unigrama | Secuencias/s | Memoria pico |
|---|---|---|---|---|---|---|
| fp32 | 2,75 | 19,4 | 24,9 % | 84,1 | 122 | 14,6 GiB |
| bf16 | 2,69 | **18,3** | 25,4 % | 84,1 | **163** | **9,1 GiB** |

- **bf16 converge igual que fp32** (incluso un poco mejor en esta corrida, dentro del ruido): pérdida de train por tramos de 200 pasos 3,98 / 3,33 / 2,99 / 2,86 / 2,70 contra 3,85 / 3,20 / 3,00 / 2,91 / 2,77, y sin NaN. El pendiente de la fase 0 queda cerrado: **se usa bf16**.
- bf16 es 1,3 veces más rápido que fp32 con lote 16 (no 2 veces como con 2.560 tokens fijos: con largos variables la GPU está menos saturada) y usa un 38 % menos de memoria.
- Con sólo 1.000 pasos (16.000 secuencias, 5 % de una pasada por train) la perplejidad MLM ya es **4,6 veces menor que la unigrama** (18,3 contra 84,1).

### Lote y memoria (bf16)

| Lote | Pasos medidos | Secuencias/s | Tokens/s | Pasos/s | Memoria pico |
|---|---|---|---|---|---|
| 16 | 1.000 | 163 | 71.000 | 10,2 | 9,1 GiB |
| 32 | 300 | **242** | 115.000 | 7,6 | **18,0 GiB** |

Con lote 32 el pico llega a 18 GiB de 24; un lote con las 32 secuencias más largas (2.560 tokens) usa ~18,4 GiB (fase 0), así que **el lote 32 es el máximo seguro con `max_length=2560`**. Un lote 64 no entra en el peor caso.

### Costo estimado de la fase 4

Con bf16 y lote 32 (242 secuencias/s):

| Calendario | Secuencias | Tiempo |
|---|---|---|
| Una pasada por train (344.025) | 344 k | ~24 min |
| Equivalente al paper (30 épocas × 30.000) | 900 k | ~1 h |
| 10 pasadas | 3,4 M | ~4 h |
| 30 pasadas | 10,3 M | ~12 h |

Es mucho menos que la estimación de la fase 0 (~4 h para 900 k secuencias de 2.560 tokens fijos), porque casi todas las secuencias son cortas. Cabe de sobra en `multi` (2 días) con jobs encadenados.

### SOP sigue sin aprender

En las tres corridas la pérdida de SOP de validación queda en ≈1,62 (1,643 → 1,615 en bf16 y en fp32 a lo largo de los 1.000 pasos). **Ese valor es el de un predictor que ignora la entrada**: con los pesos de clase 1/p y `label_smoothing=0.1`, un modelo que da probabilidades uniformes (1/3, 1/3, 1/3) obtiene 1,636 (calculado con la misma función de pérdida; la mejor constante posible da lo mismo). Es decir que la cabeza de SOP no aprendió nada útil todavía, y el «recall de la clase normal = 0» es sólo el `argmax` de probabilidades casi uniformes, que salta de una clase a otra entre validaciones. La métrica que sirve es la pérdida de SOP contra la referencia 1,636, no el recall ni la exactitud.

Por qué es difícil (medido en la muestra del 1 %): invertir o mezclar eventos cambia en promedio sólo el 25-28 % de los tokens del documento, y en el 41-44 % de los documentos cambia menos del 20 %, porque rama, provincia, tamaño y antigüedad suelen repetirse en eventos consecutivos. El modelo no tiene posición por índice ni posición relativa en la atención local (`performer.py:33`, `rel_pos = None`), así que sólo puede detectar la perturbación comparando el contenido de un evento con su fecha y edad (que quedan en su lugar), por ejemplo tendencias de salario o de antigüedad.

Los 1.000 pasos son el 5 % de una pasada por train, así que no hay evidencia de que no pueda aprenderlo con más entrenamiento. La fase 4 tiene que decidirlo con datos antes de gastar horas: medir la pérdida de SOP contra 1,636 en la primera pasada y, si sigue ahí, cambiar el diseño (proporciones balanceadas de clases, perturbaciones sólo en documentos con muchos eventos) o dejar SOP con peso casi nulo y reportarlo como desvío del paper.

## Comparación con el repo completo de life2vec (revisado el 2026-10-19)

Se leyeron `SocialComplexityLab/life2vec` (`src/transformer/performer.py`, `embeddings.py`, `models.py`, `src/tasks/{base,mlm}.py`, `src/data_new/augment.py`, `conf/`). La atención y los embeddings coinciden con los de `life2vec-light`: sin posición por índice, `rel_pos = None` en la atención local y sin rotación en las cabezas globales (líneas comentadas bajo «OUR EDITS TO THE PACKAGE»). En el SOP, `cls_task` sólo reordena `document.sentences`: la posición absoluta, la edad y el segmento no se mueven, igual que en esta implementación.

| | life2vec (repo completo) | labor2vec |
|---|---|---|
| Ciclo de segmentos por evento | `[2, 3, 1]` (background en 1) | era `(2, 3)`; **corregido a `(2, 3, 1)`** |
| Pesos de clase de SOP | `[1/0,9, 1/0,1, 1/0,1]` = 1,1 / 10 / 10 (con probabilidades reales 0,9 / 0,05 / 0,05) | 1,1 / 20 / 20 (1/p exacto): **el doble** |
| Suavizado de etiquetas, pesos MLM/SOP | 0,1; 0,8 / 0,2 | igual |
| Recorte de gradiente | 1,0 | era `null`; **ahora 1,0** en la configuración |
| Lote efectivo | lote 8 × `accumulate_grad_batches` 8 = **64** | lote 16-32 sin acumulación |
| Época | 3.000 lotes de 8 = **24.000 secuencias**, 375 pasos del optimizador; 30 épocas = 720 k secuencias | una pasada por train (344 k) |
| Precisión | 32 | bf16 (validado en esta fase) |
| Validación | cada 3 épocas; recarga de dataloaders cada 5 | cada época |
| Orden de tokens dentro de cada evento | **se mezcla al azar en cada acceso** (`shuffle_within_sentences: true`, incluso en validación) | orden fijo |
| Ruido temporal | ±5 días | ±15 días (granularidad mensual) |
| `drop_tokens` (p = 0,01) | por evento, quita un token | por documento, quita el 10 % de los tokens |

Notas:
- El orden aleatorio de tokens dentro del evento hace que el original sea invariante a ese orden; acá se eligió un orden fijo (subplan de la fase 2, punto 3). Es una decisión de diseño a revisar en la ablación de la fase 8.
- El subplan y `PLAN.md` hablaban de «30 épocas de 30.000 secuencias»: según `conf/trainer/pretrain.yaml` son 3.000 lotes de 8 = 24.000.
- Con lote efectivo 64, cada paso del optimizador ve ~3 ejemplos de cada clase minoritaria de SOP; con lote 16 y pesos 20 el gradiente de SOP era mucho más ruidoso. Conviene fijar el lote efectivo en la fase 4 (acumulando gradientes si hace falta).

## Criterios de aceptación de la fase 3

1. Script de entrenamiento, configuración y métricas con pruebas en verde: ✔ (38 pruebas).
2. Informe con la prueba de humo en CPU: ✔; con la comparación bf16 contra fp32 y los tiempos en la A30: ✔ (arriba).
3. Código sincronizado a Mendieta: ✔.
4. Los cinco criterios de la prueba de humo: ✔.

**Fase 3 cerrada.**
