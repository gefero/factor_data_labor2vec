# Fase 4: pre-entrenamiento en Mendieta

Generado el 2026-10-19. Plan: [plans/PLAN_fase_4.md](../plans/PLAN_fase_4.md). Código: `src/labor2vec/entrenar.py`, `scripts/cluster/preentrenar.sbatch`, `configs/preentrenamiento.yaml`.

## Configuración

Backbone del paper (8,1 M de parámetros con el vocabulario de 519 tokens), variante A, ventana a 2015-12 (344.025 trabajadores de train), bf16, lote 32 × 2 acumulaciones = **lote efectivo 64**, recorte de gradiente 1,0, `learning_rate=5e-3` con OneCycle, pesos de clase del SOP 1,1 / 10 / 10 (como el repo completo de life2vec), época = 375 pasos del optimizador (24.000 secuencias), semilla 0, validación sobre 200 lotes fijos (6.400 trabajadores de validación).

| Corrida | Épocas | Pasos | Secuencias vistas | Jobs |
|---|---|---|---|---|
| Diagnóstico (`diag_sop`) | 14 | 5.250 | 336.000 (1 pasada) | 1180689 (36 min, con corte y reanudación en el paso 1.499) |
| Pre-entrenamiento (`pre_30ep`) | 30 | 11.250 | 720.000 (~2,1 pasadas) | 1180699 (56 min) + 1180700 (14 min), encadenados |

Son entrenamientos independientes (cada uno con su propio calendario OneCycle): el segundo no continúa al primero. Rendimiento: 214-215 secuencias/s, 18 GiB de memoria pico de la GPU (A30, 24 GiB). Memoria del nodo 17 GB de 30; los 10 cores de carga no limitan.

## Resultados de validación

| | Diagnóstico (14 épocas) | **Pre-entrenamiento (30 épocas)** |
|---|---|---|
| Perplejidad MLM | 7,06 | **6,36** |
| Perplejidad de la unigrama | 81,4 | 81,4 |
| Exactitud top-1 MLM | 41,9 % | **43,2 %** |
| AUC de SOP | 0,741 | **0,795** |
| Pérdida de SOP | 1,332 | **1,261** |

Criterio del plan (AUC de SOP ≥ 0,70 con pérdida < 1,375): **cumplido** por las dos corridas. Con el pre-entrenamiento de 30 épocas, el AUC supera 0,70 desde la época 11 y llega a 0,78-0,80 desde la época 17.

Curva del pre-entrenamiento (cada 3 épocas):

| Época | Paso | Perplejidad MLM | Exactitud MLM | AUC SOP | Pérdida SOP |
|---|---|---|---|---|---|
| 2 | 1.124 | 14,19 | 27,0 % | 0,563 | 1,396 |
| 5 | 2.249 | 9,51 | 32,1 % | 0,637 | 1,384 |
| 8 | 3.374 | 8,21 | 37,6 % | 0,664 | 1,365 |
| 11 | 4.499 | 7,54 | 39,7 % | 0,710 | 1,345 |
| 14 | 5.624 | 6,92 | 41,9 % | 0,757 | 1,308 |
| 17 | 6.749 | 6,75 | 42,2 % | 0,782 | 1,291 |
| 20 | 7.874 | 6,60 | 42,7 % | 0,777 | 1,277 |
| 23 | 8.999 | 6,47 | 43,0 % | 0,796 | 1,267 |
| 26 | 10.124 | 6,38 | 43,2 % | 0,795 | 1,262 |
| 29 | 11.249 | 6,36 | 43,2 % | 0,795 | 1,261 |

- **MLM**: la perplejidad es 12,8 veces menor que la de la unigrama. Mejora un 10 % sobre el diagnóstico al duplicar las secuencias vistas, y las últimas tres validaciones casi no cambian (la tasa de aprendizaje ya es mínima al final de OneCycle), así que no se puede distinguir un límite del modelo de un efecto del calendario con esta única corrida.
- **Sin sobreajuste visible**: la pérdida MLM de train (2,0-2,04, con dropout y augmentación) es mayor que la de validación (1,85) en todo el tramo final.
- **SOP**: en validación, la exactitud global es 86 % y el recall por clase es 91 % (orden normal), 32 % (invertido) y 45 % (mezclado). La clase «normal» es el 90 % de la validación, así que la exactitud global engaña; el AUC (0,795) es la medida a usar. El modelo distingue mejor los documentos mezclados que los invertidos.
- Sin NaN en ninguna de las corridas.

## Operación en el cluster

- La reanudación en la GPU (corte en el paso 1.499 del diagnóstico) funcionó, y el pre-entrenamiento en dos tramos encadenados con `afterany` también: el tramo B restauró el checkpoint del A y completó los 11.250 pasos. El tramo A paró solo con 5 minutos de margen y terminó `COMPLETED`.
- El tramo B esperó ~2,5 h en la cola de `short` por un nodo ocupado por otro usuario, aunque tenía la mayor prioridad de la partición.
- Un job enviado con `sbatch --export=ALL,VAR=valor` (1180677) quedó retenido con `user env retrieval failed requeued held` y hubo que cancelarlo; las variables se pasan por el entorno del envío (`CORTE_PRUEBA=1500 sbatch ...`).
- polars no se puede usar en el cluster (necesita AVX2; fase 3).

## Entregables

- `modelo_final.pt` (sólo pesos, 33 MB) y `config.json` de cada corrida, en `~/labor2vec/runs/{diag_sop,pre_30ep}/` en Mendieta y con copia local en `runs/*_vista/` (no versionado). Los `last.ckpt` y `mejor.ckpt` (97 y 33 MB) quedan sólo en el cluster.
- CSV de métricas por ejecución en `logs/version_N/metrics.csv`.

## Pendiente

1. **Corrida larga (4.4)**: ¿vale la pena? La perplejidad bajó un 10 % al duplicar las secuencias vistas, pero con una sola corrida no se puede separar el efecto de entrenar más del efecto del calendario. Costaría ~4 h (150 épocas, unas 10 pasadas).
2. **Semillas extra para el test de aleatorización** (fase 5): 2 corridas más de la configuración elegida, ~1 h cada una con 30 épocas.
