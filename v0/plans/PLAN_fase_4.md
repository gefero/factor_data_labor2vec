# Subplan: fase 4 (pre-entrenamiento en Mendieta)

Estado: decisiones tomadas el 2026-10-19; falta la aprobación final del plan. Complementa [PLAN.md](PLAN.md), sección 6.

**Objetivo**: obtener un backbone pre-entrenado (MLM + SOP) sobre la MLER, con curvas de pérdida y perplejidad MLM de validación, y saber con evidencia si el SOP aprende. El entregable es un checkpoint con el que arrancan las fases 5 (espacio de conceptos) y 6 (fine-tuning).

## Punto de partida (medido en la fase 3)

- bf16 converge igual que fp32. Con lote 32: 242 secuencias/s y 18 GiB de pico en la A30 (24 GiB). El lote 64 no entra.
- El DataLoader entrega ~10.000 secuencias/s con 9 workers: sobra.
- Con 1.000 pasos de lote 16 el MLM llega a perplejidad 18,3 contra 84,1 de la unigrama.
- **El SOP no aprendió nada** (pérdida ≈ la de un modelo que ignora la entrada).
- El repo completo de life2vec usa: pesos de clase de SOP `[1/0,9, 1/0,1, 1/0,1]` = 1,1 / 10 / 10, recorte de gradiente 1,0, lote efectivo 64 (lote 8 × 8 acumulaciones) y épocas de 375 pasos del optimizador (24.000 secuencias), 30 épocas. Ver [informes/fase3.md](../informes/fase3.md).

## 4.1 Cambios de código (local, con pruebas)

1. **Acumulación de gradientes**: `entrenamiento.acumular` (por defecto 2). Lote efectivo = `tamano_lote` × `acumular` = 32 × 2 = 64. El calendario OneCycle se calcula en pasos del optimizador (`steps_per_epoch` = pasos de la época / `acumular`).
2. **Pesos de clase del SOP explícitos**: hiperparámetro `sop_pesos` (por defecto `[1/0,9, 10, 10]`, como el original). `sop_proporciones` queda sólo para las probabilidades de la perturbación (0,9 / 0,05 / 0,05).
3. **Época como en el original**: `pasos_por_epoca = 375` pasos del optimizador (24.000 secuencias de lote efectivo 64). Esto **cambia** lo aprobado en la fase 3 («época = una pasada por train»): una pasada completa son ~5.375 pasos, o sea ~14 épocas de las nuevas. Se usa la definición del original para que el calendario de OneCycle y las comparaciones con el paper sean directas.
4. **Métrica de SOP libre de umbral**: AUC de «perturbado contra normal» usando 1 − P(normal) (`val/sop_auc`). El recall por clase y la exactitud engañan cuando las probabilidades son casi uniformes (el `argmax` salta de una clase a otra); el AUC vale 0,5 si el modelo no distingue nada.
5. **Job de pre-entrenamiento** `scripts/cluster/preentrenar.sbatch`:
   - carga el entorno en el `/scratch` del nodo (como `humo_gpu.sbatch`) y comprueba las importaciones antes de empezar;
   - escribe los checkpoints en el `/home` (`~/labor2vec/runs/<nombre>/`), no en el `/scratch`: el `/scratch` se borra al terminar el job;
   - calcula el tiempo restante con `squeue -h -j $SLURM_JOB_ID -o %e` menos un margen (5 min en `short`, 15 min en `multi`) y se lo pasa a `--max-minutos`; el job termina `COMPLETED`, no `TIMEOUT`;
   - se reanuda con el mismo comando (`--reanudar`); para encadenar jobs se usa `afterany` con nombres distintos (lecciones de Mendieta, §4 y §9).
6. **Protección contra pérdidas NaN**: si la pérdida no es finita, se guarda y se corta el job con error.
7. Pruebas: acumulación (el calendario cuenta pasos del optimizador), pesos de SOP, AUC sobre un caso construido a mano, y que reanudar desde un checkpoint con acumulación llega al mismo paso final.

## 4.2 Diagnóstico del SOP (job en `short`, ≤ 1 h)

Una pasada por train con la configuración nueva: lote efectivo 64, pesos 1,1 / 10 / 10, bf16, recorte 1,0, **14 épocas de 375 pasos** (~24 min de cómputo más las validaciones).

- Incluye una **prueba de corte y reanudación en la GPU** (corta en el paso 1.500 con `--max-pasos` y reanuda en el mismo job): la lección de Mendieta §5 es que reanudar falló en GPU y no en CPU.
- **Referencia del SOP** con estos pesos y `label_smoothing=0,1` (calculada con la misma función de pérdida): un modelo que ignora la entrada obtiene **1,39-1,40** (la mejor constante posible da 1,388).
- **Criterio** (definido con vos el 2026-10-19): el SOP «aprende» si, al final de la pasada, `val/sop_auc ≥ 0,70`. La pérdida de SOP de validación es un control de coherencia, no un umbral independiente: debe quedar por debajo de **1,375** y seguir bajando.
  - Relación entre las dos medidas (simulada con la misma función de pérdida y el mejor modelo posible para cada AUC): AUC 0,50 → 1,388; 0,60 → 1,385; 0,65 → 1,374; **0,70 → 1,366**; 0,75 → 1,348; 0,80 → 1,332; 0,90 → 1,284. La pérdida casi no se mueve con AUC bajos, por eso el AUC es la medida principal. Un modelo real, peor calibrado que el ideal, dará una pérdida algo mayor para el mismo AUC.
- Se corre una sola semilla. Si el resultado es dudoso (AUC 0,55-0,60) se decide con vos.

**Qué se hace con el resultado**:
- *Aprende*: se sigue con 4.3 sin cambios.
- *No aprende*: se detiene el avance antes de gastar más cómputo y se decide con vos entre (a) dejar el SOP con peso casi nulo (0,02-0,05) y reportarlo como desvío del paper, (b) cambiar la perturbación (sólo documentos con muchos eventos, proporciones más balanceadas) o (c) activar la posición relativa en las cabezas locales (desvío de arquitectura). El MLM aprende por sí solo, así que ninguna de las opciones bloquea las fases siguientes.

## 4.3 Pre-entrenamiento fiel al paper

30 épocas × 375 pasos × 64 secuencias = **720.000 secuencias** (≈ 2,1 pasadas por train). Con 242 secuencias/s son ~50 minutos más las validaciones.

- Mismos hiperparámetros del paper (`HPARAMS_PAPER`), `learning_rate=5e-3`, OneCycle sobre 30 × 375 pasos, bf16, recorte 1,0, ventana a 2015-12, variante A, semilla 0.
- Validación cada 3 épocas (1.125 pasos) sobre 200 lotes fijos de validación.
- Checkpoint `last.ckpt` cada 500 pasos, `mejor.ckpt` por pérdida MLM de validación y `modelo_final.pt` (sólo pesos, ~35 MB) para traer al repo local.
- Se encola como un job de `short` en dos tramos o como un job de `multi` (la estimación de arranque decide; se mide con `sbatch --test-only`).

## 4.4 Pre-entrenamiento largo (opcional, se decide con las curvas de 4.3)

Si la perplejidad MLM de validación sigue bajando al final de 4.3, repetir con más pasos (por ejemplo 150 épocas × 375 pasos = 3,6 M de secuencias, unas 10 pasadas, **~4 h** en `multi`). La población es ~10 veces menor que la del paper, así que más pasadas pueden compensar parte de la diferencia. Se decide mirando la curva de validación (si se aplana, no hace falta).

## 4.5 Las tres semillas para el test de aleatorización (fase 5)

El análisis del espacio de conceptos del paper usa 3 modelos entrenados con semillas distintas. La configuración final de 4.3 o 4.4 se repetirá con 2 semillas más antes de la fase 5. Con 4.3 son ~1 h cada una; con 4.4, ~4 h. No se encolan sin tu OK.

## Pruebas y entregables

1. Cambios de 4.1 con pruebas en verde y código sincronizado a Mendieta.
2. `informes/fase4.md`:
   - curvas de pérdida (total, MLM, SOP) y perplejidad MLM de validación contra la unigrama;
   - exactitud top-1 del MLM;
   - AUC y pérdida de SOP contra la referencia;
   - tiempos y memoria por paso en la GPU, y el resultado de la prueba de corte y reanudación;
   - la decisión sobre el SOP y su justificación.
3. `modelo_final.pt` y `config.json` de cada pre-entrenamiento en `modelos/` (no versionado, ya está en `.gitignore`).

## Criterio de aceptación

1. 4.2 ejecutado, con el SOP evaluado contra la referencia y el AUC, y una decisión documentada.
2. Un pre-entrenamiento (4.3 como mínimo) completado, con perplejidad MLM de validación por debajo de la unigrama y sin NaN.
3. La reanudación en GPU probada.
4. `modelo_final.pt` disponible y el informe escrito.

## Qué se encola y cuándo

Cada job (4.2, 4.3, 4.4, 4.5) se encola **sólo con tu OK explícito** y con aviso por mail en BEGIN, END y FAIL. Los resultados se revisan después de cada uno.

## Decisiones (tomadas el 2026-10-19)

1. **Cambio de la definición de época** (punto 4.1.3): aceptado, 375 pasos del optimizador (24.000 secuencias) como el original.
2. **Criterio del SOP** (4.2): resuelto, AUC ≥ 0,70 con la pérdida como control (< 1,375).
3. **Calendario base** (4.3): resuelto, 30 épocas (720 k secuencias, ~1 h) primero y 4.4 según las curvas.
4. **Si el SOP no aprende**: resuelto, me detengo y te consulto.
