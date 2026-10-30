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

## Corrida larga (`pre_150ep`, 4.4)

150 épocas × 375 pasos = 56.250 pasos = 3,6 M de secuencias (unas 10 pasadas por train), misma configuración y semilla 0. Seis tramos de `short` encadenados con `afterok` (jobs 1180708 a 1180713), todos `COMPLETED`, sin NaN: del 19/10 a las 15:19 hasta las 20:27 hora de Argentina, 5 h 8 min en total, y entre tramos prácticamente no hubo espera en la cola. Rendimiento: 225 secuencias/s, 18 GiB de pico.

| | Diagnóstico (14 épocas) | Pre-entrenamiento (30 épocas) | **Corrida larga (150 épocas)** |
|---|---|---|---|
| Perplejidad MLM de validación | 7,06 | 6,36 | **4,73** |
| Exactitud top-1 MLM | 41,9 % | 43,2 % | **47,8 %** |
| AUC de SOP | 0,741 | 0,795 | **0,865** |
| Pérdida de SOP | 1,332 | 1,261 | **1,196** |
| Recall de SOP por clase (normal / invertido / mezclado) | — | 91 / 32 / 45 % | 91 / 39 / 63 % |
| Perplejidad de la unigrama | 81,4 | 81,4 | 81,4 |

- **Perplejidad MLM**: 17 veces menor que la unigrama y un 26 % menor que la de 30 épocas. La curva baja de forma sostenida (6,87 en la época 29, 6,04 en la 47, 5,14 en la 83, 4,81 en la 119) y se aplana al final (4,75 en la época 128, 4,73 en la 137, 4,73 en la 149), cuando la tasa de aprendizaje llega a su mínimo.
- **SOP**: el AUC sube a lo largo de toda la corrida y llega a 0,865. El modelo ahora distingue mejor los documentos mezclados (recall 63 %) que los invertidos (39 %).
- **Sin sobreajuste visible**: la pérdida MLM de train (1,67-1,72, con dropout y augmentación) sigue por encima de la de validación (1,55) al final de la corrida, aun después de unas 10 pasadas.
- **Respuesta a la pregunta del 4.4**: la corrida larga sí valió la pena. La mejora de duplicar las secuencias vistas (14 a 30 épocas) fue de ~10 % en la perplejidad; la de multiplicar por 5 (30 a 150 épocas) fue de ~26 %. No se puede afirmar si 300 épocas seguirían mejorando: la tasa de aprendizaje en el mínimo al final de OneCycle hace que la curva se aplane sin que eso muestre un límite del modelo.

El modelo recomendado para las fases siguientes es `pre_150ep/modelo_final.pt` (33 MB; en el cluster, y con copia local en `runs/pre_150ep_vista/`, no versionada).

## Tres semillas de la configuración de 150 épocas (para el test de aleatorización de la fase 5)

Se entrenaron dos modelos más con la misma configuración y las semillas 1 y 2 (`pre_150ep_s1` y `pre_150ep_s2`): 2 cadenas de 6 tramos de `short` con `afterok`, jobs 1180789 a 1180794 y 1180795 a 1180800, todos `COMPLETED`, corriendo en paralelo en nodos distintos entre el 19/10 a las 21:25 y el 20/10 a las 01:56 hora de Argentina (4,5 h de cómputo cada una; el sexto tramo de cada cadena sólo retomó un entrenamiento ya completo y salió en 1 minuto). Sin NaN; 56.250 pasos en las dos.

| Semilla | Perplejidad MLM | Exactitud MLM | AUC de SOP | Pérdida de SOP | Recall SOP (normal / invertido / mezclado) |
|---|---|---|---|---|---|
| 0 (`pre_150ep`) | 4,725 | 47,8 % | 0,865 | 1,196 | 91 / 39 / 63 % |
| 1 | 4,855 | 47,5 % | 0,878 | 1,154 | 89 / 47 / 70 % |
| 2 | 4,838 | 47,5 % | 0,866 | 1,210 | 91 / 43 / 64 % |
| **Media ± desvío** | **4.806 ± 0.071** | 47.6 ± 0.2 % | 0.870 ± 0.007 | 1.187 ± 0.029 | |

- Los tres modelos son consistentes: la dispersión entre semillas es de ~1,5 % en la perplejidad y de 0,01 en el AUC de SOP, mucho menor que la diferencia con la corrida de 30 épocas (6,36 / 0,795). El resultado de la corrida larga no depende de la semilla.
- Los modelos están en `~/labor2vec/runs/pre_150ep`, `pre_150ep_s1` y `pre_150ep_s2` (`modelo_final.pt` en cada carpeta), y como copia local en `runs/pre_150ep_vista/`, `runs/pre_150ep_s1_vista/` y `runs/pre_150ep_s2_vista/` (no versionado).

## Pendiente

1. Fase 5: evaluación del espacio de conceptos (vecindades por coseno, PaCMAP, test de aleatorización con los 3 modelos y hubness).
