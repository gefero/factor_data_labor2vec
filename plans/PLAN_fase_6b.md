# Subplan: fase 6b (T2, T3 y T4)

Estado: aprobado (2026-10-20). Complementa [PLAN.md](PLAN.md) (sección 4) y [PLAN_fase_6.md](PLAN_fase_6.md).

## Contexto

La fase 6 cerró T1 (salida del empleo registrado): el transformer pre-entrenado (AUL 0,619) empata con la regresión logística (0,617) y el GRU (0,616), y le gana por ~0,06 al transformer sin pre-entrenar. La pregunta que queda es si en otras tareas del mismo registro el pre-entrenamiento muestra más ventaja. PLAN.md definía tres tareas más, que acá se concretan con las decisiones que tomaste hoy:

| # | Tarea | Qué predice | Clases | Cohorte |
|---|---|---|---|---|
| T2 | Posición salarial futura | Quintil del percentil salarial medio de 2019 (ordinal) | 5 | quienes siguen registrados en 2019 |
| T3 | Cambio de sector | Si la rama (`letra`) de la relación principal en 2019 es distinta de la de 2015 (binaria) | 2 | ídem |
| T4 | Transición de tamaño de empresa | Tamaño de la empresa principal en 2019 respecto de 2015: más chica / igual / más grande (ordinal) | 3 | ídem |

## Datos medidos (lectura sobre el panel, sin escribir nada)

- Cohorte de T1: 122.492. De ellos **102.885 siguen registrados en 2019** (84 %): es la población de T2-T4, con los mismos splits de la fase 1 (~72 k train, ~15 k validación, ~15 k test).
- **T2:** el percentil salarial medio de 2015 y el de 2019 tienen correlación 0,84. «Repetir el quintil de 2015» acierta el 64 % (diagonal de la matriz de transición). 381 sobrevivientes no tienen ningún mes con remuneración positiva en 2019 y se excluyen de T2.
- **T3:** cambia la letra el **11,3 %**. Los destinos más frecuentes son las letras 11 y 7 (~2.000 cada una).
- **T4:** cambia el tramo el 19,0 %: 9,3 % pasa a una empresa más grande y 9,8 % a una más chica (tramo 1 = más de 200 empleados).
- Estas tasas se vuelven a calcular al construir las tablas; si difieren mucho se revisa antes de seguir.

## Definiciones precisas (a fijar en `tareas/derivadas.py`)

- **Atributos de un trabajador en un año** (2015: meses 228-239; 2019: 276-287): se toma la **relación principal**, la de más meses registrados en el año (desempate por mayor remuneración total). De ahí salen `letra` y `tramo_empleo`.
- **Nivel salarial del año:** promedio de `pct_rem_mes` sobre los meses con remuneración positiva. Cortes de quintiles ajustados **sólo con train** sobre la distribución de 2019 y aplicados a 2015 y 2019.
- **Exclusiones:** sin letra o con `tramo_empleo` 0 («sin definir») en alguno de los dos años (T3 y T4); sin percentil en 2019 (T2).
- **Salida:** `data/processed/tareas/t2.parquet`, `t3.parquet`, `t4.parquet` con `id_trabajador, split, sexo, fnacim, edad_corte, y` y el **estado de 2015** (`q2015`, `letra2015`, `tramo2015`), que sirve de línea de base de persistencia.
- La entrada de los modelos es la misma de T1: historia hasta 2015-12. No se filtra información posterior.

## Lo que hace el repo original (lo que se replica)

Verificado en `hexaco_model.py`, `cls_model.py`, `transformer_utils.py`, `conf/experiment/psy*.yaml` y los cuadernos de métricas (la única tarea ordinal del original es la de personalidad, 5 clases):

- **Cabeza:** [CLS] y un decodificador plano (swish, dropout 0,3, 280→2210→280, salida `n_clases`). Nosotros reutilizamos la cabeza de atención de T1 (`AttentionDecoder`) con `n_clases` salidas, para que T1-T4 sean comparables.
- **Optimizador:** AdamW lr 5e-4, decaimiento por capa 0,95, `ExponentialLR(0,8)`, dropout 0,4 en el encoder, lote efectivo 64, época de 500 lotes, 25 a 50 épocas, parada temprana por QWK (paciencia 10).
- **Pérdida ordinal:** combina tres términos:
  - CDW-CE, que penaliza `-Σ log(1 − p_i)·w_i` con `w_i` creciente con la distancia a la clase verdadera (en el código `log(1+|i−y|)`; en el paper `|i−y|^1,5`);
  - *focal* (γ = 2 en el código, 5 en el paper);
  - suavizado de etiquetas.
  - **El paper y el código no coinciden** (exponente, γ, pesos fijos 0,3 / 1 / 0,1 contra CoV-weighting adaptativo). No se puede saber cuál produjo los números publicados.
- **Métricas:** kappa cuadrático ponderado (QWK) y MAE, F1 y MCC; en el paper el QWK de personalidad quedó entre 0,11 y 0,23 contra ~0,0 del azar.
- **Baselines:** GRU con la misma pérdida.

## Cambios de código (generalizar lo de T1, sin duplicar)

1. **`tareas/derivadas.py`** (nuevo, `python -m labor2vec.tareas.derivadas`): arma las tres tablas con DuckDB. Reutiliza `tareas/t1.py` (cohorte, `ANIO_CORTE`, `_ventana`). Imprime el informe de control (tamaños, tasas por split, sexo y edad).
2. **`tareas/datos.py`:** `DatasetT1` pasa a leer la columna que indique `etiqueta` (ya es un parámetro) y el sampler balanceado acepta K clases (hoy usa `bincount(minlength=2)`). `DataModuleT1` recibe `ruta_t1` y `etiqueta`, que ya existen.
3. **`modelo/clasificador.py`:** hiperparámetro `n_clases` en `ClasificadorT1` (salida de `AttentionDecoder` y `predecir`), y `perdida` = `balanceada` (T1 y T3) u `ordinal` (T2 y T4). Se agrega la pérdida ordinal (`perdida_ordinal`) y el monitor de validación configurable (`val/aul`, `val/qwk` o `val/macro_auc`).
4. **`tareas/metricas.py`:** QWK, MAE, F1 macro, exactitud balanceada, AUC macro uno-contra-resto y su bootstrap estratificado por clase, con la misma convención (sólo numpy, sin polars).
5. **`ajustar.py`:** la tarea sale del YAML (`tarea: {ruta, etiqueta, n_clases, perdida, monitor}`); las predicciones se escriben como `p` (binario) o `p_0..p_{K-1}` (multiclase) con la etiqueta `y`. Hay que conservar el formato de T1.
6. **`tareas/tabulares.py`:** salida multinomial en la regresión logística y en la FFNN, y las líneas de base de persistencia y «estado actual» (ver abajo).
7. **`tareas/evaluar.py`:** un registro de tareas (patrones de carpetas, métricas y subgrupos de cada una) y un `--tarea t2|t3|t4`; escribe `informes/fase6b/<tarea>/`.
8. **Configs:** `configs/ajuste_t2.yaml`, `ajuste_t3.yaml`, `ajuste_t4.yaml` y sus GRU; mismo transformer con lr 0,001 (el elegido por validación en T1, sin reajustarlo).
9. **`scripts/cluster/ajustar.sbatch`:** se reutiliza tal cual.

## Modelos y líneas de base (por tarea)

| Modelo | Comentario |
|---|---|
| Transformer pre-entrenado (L2V) | 3 semillas, lr 0,001 |
| Transformer sin pre-entrenar | 1 semilla |
| GRU | 3 semillas |
| FFNN | 3 semillas (conteos de tokens, sexo, edad) |
| Regresión logística (multinomial) | conteos de tokens, sexo y edad |
| **Logística con estado actual** | edad, sexo y el estado de 2015 (`q2015`, `letra2015` o `tramo2015`). Es la línea de base fuerte y barata que hay que superar |
| **Persistencia** | predice el estado de 2015 (T2: `q2015`; T4: «igual»; T3 binaria: sin información, AUL 0,5) |
| Tabla de vida (edad y sexo), mayoría y azar | como en T1 |

El entrenamiento usa muestreo balanceado por clase (como T1) y los resultados se reportan con probabilidades corregidas por prevalencia donde haga falta.

## Métricas (IC por bootstrap estratificado, 1.000 réplicas)

- **T2:** QWK (principal), MAE, F1 macro, exactitud balanceada y exactitud de la clase; QWK entre quienes **cambian de quintil** (para medir movilidad, no persistencia).
- **T3:** las de T1: AUL, AUC, PR-AUC, C-MCC (α = 0, β = 1).
- **T4:** AUC macro uno-contra-resto (monitor), QWK, exactitud balanceada y F1 macro; AUC de «más grande» y de «más chica» por separado.
- Subgrupos por sexo y edad, y la comparación con la persistencia, como en T1.

## Cómputo

- **Jobs en Mendieta:** L2V 3 + GRU 3 + sin pre-entrenar 1 = 7 por tarea, **21 en total**, de ~25 a 35 min cada uno en la partición `short` (los GRU pueden pasar de 1 h y se reanudan con el mismo comando). Con 3 o 4 corriendo en paralelo, unas 4 a 6 horas de pared.
- **Ablación de la pérdida:** 1 corrida de T2 y 1 de T4 (semilla 0) con entropía cruzada balanceada común, para ver si la pérdida ordinal aporta algo.
- **Local:** tablas de etiquetas (segundos) y baselines tabulares (minutos). Pido OK antes de cada corrida pesada y antes de encolar cada tanda, con aviso por mail (BEGIN, END, FAIL).
- Se sincroniza `data/processed/tareas/` al cluster; el modelo `pre_150ep` ya está allí.

## Pruebas

- Tablas de etiquetas con un panel sintético de casos de borde: relación principal con empate, trabajador sin letra o con tramo 0, quintil en el límite, sobreviviente con remuneración cero, y que los cortes se ajusten sólo con train.
- Pérdida ordinal: es 0 con la clase correcta segura y crece con la distancia a la clase verdadera; reduce a la CE balanceada con `n_clases = 2` si se pide.
- Métricas a mano: QWK perfecto, al azar y contra `sklearn.cohen_kappa_score`; AUC macro contra `roc_auc_score(multi_class="ovr")`.
- Sampler balanceado con 3 o 5 clases: lotes con frecuencias parecidas.
- Una pasada `fit` corta en CPU por tarea con el modelo chico, y recarga del checkpoint. Las pruebas actuales de T1 (83) siguen pasando, y el camino de entrenamiento sigue sin importar polars.

## Fuera de esta fase

- T3 de destino (14 letras), la variante «cambio de quintil» de T2 y la binaria de T4: quedan como extensión.
- Búsqueda de hiperparámetros: se usa lr 0,001 de T1 sin reajustar; si una tarea queda claramente mal entrenada se revisa y se documenta.
- Variantes de la ventana (1 año, 4 años) y el control sin atributos «último valor»: fase 8.

## Entregables y criterio de aceptación

1. Código con pruebas en verde y comandos reproducibles.
2. `data/processed/tareas/t{2,3,4}.parquet` con su informe de control.
3. `informes/fase6b.md` con, por tarea: la tabla de modelos con IC, la comparación con persistencia y con «estado actual», el aporte del pre-entrenamiento (con y sin), subgrupos, la ablación de la pérdida y una lectura honesta; más una síntesis que cruce T1-T4 y responda si el pre-entrenamiento aporta más en alguna tarea.

No hay umbral de aprobación: se acepta cuando todos los modelos están evaluados sobre el mismo test con su IC.

## Orden de trabajo

1. Guardar este plan como `plans/PLAN_fase_6b.md`.
2. `derivadas.py` y sus pruebas; correr las tablas en local (con OK) y revisar los números con vos.
3. Generalizar datos, clasificador, métricas, `ajustar`, tabulares y `evaluar`, con pruebas.
4. Baselines tabulares en local (con OK); una prueba de humo en CPU por tarea.
5. Encolar primero 1 job de L2V por tarea y revisar que entrenen bien; luego el resto (con OK).
6. Evaluar, escribir `informes/fase6b.md` y commitear cuando me lo pidas.

## Verificación

- `.venv/bin/python -m pytest tests -q` en verde.
- `python -m labor2vec.tareas.derivadas` reproduce los tamaños y tasas del informe de control (≈102,9 k sobrevivientes; 11,3 % de cambio de letra; 9,3 / 81 / 9,8 % en T4; persistencia ≈ 64 % en T2).
- Corrida de humo en CPU de `ajustar` con `configs/humo_t*.yaml` sobre `corpus_muestra`.
- En Mendieta, el `val/qwk` (T2 y T4) o `val/aul` (T3) del primer job tiene que superar a la persistencia o a «estado actual» antes de lanzar el resto, o se documenta por qué no.
