# Subplan: fase 6 (fine-tuning de T1 y baselines)

Estado: aprobado (2026-10-20). Complementa [PLAN.md](PLAN.md), secciones 3.3, 4 y 6.

## Contexto

Las fases 0-5 dejaron un modelo pre-entrenado (`pre_150ep`, más dos semillas) que aprendió la "gramática" de las trayectorias laborales. La fase 6 lo usa para una tarea concreta, **T1: predecir quién sale del empleo registrado**, y compara contra modelos más simples, como la tabla A2 del paper (mortalidad). El objetivo es medir si el transformer pre-entrenado le gana a esos modelos y por cuánto, y reportar el resultado aunque sea negativo.

Hoy el repo sólo pre-entrena: no hay código de etiquetas, ni dataset de clasificación, ni cabeza, ni métricas de tarea (`src/labor2vec/tareas/` está vacío).

## Decisiones ya tomadas (respuestas del 2026-10-20)

- **Cohorte**: trabajadores con **25 a 55 años** al 31-12-2015 (`2015 − fnacim`) y **registrados los 12 meses de 2015**. Según PLAN.md son unos 122.500 (a verificar). Se mantienen los splits de la fase 1, así que salen unos 86 k de train, 18 k de val y 18 k de test.
- **Etiqueta principal**: y = 1 si el trabajador **no tiene ningún mes registrado en 2019**. La tasa base esperada ronda el 16 %.
- **Prueba extra**: el mismo modelo, sin volver a entrenarlo, se evalúa contra "sin registro en 2021", para ver cuánto empeora con el shock del COVID.
- **Entrada**: la historia hasta 2015-12 (`fin_ventana=239`), igual que en el pre-entrenamiento. El modelo nunca vio datos posteriores, así que la etiqueta no se filtra.
- **Semillas**: 3 fine-tunings sobre `pre_150ep`, y lo mismo para el GRU. Se reportan la media ± el desvío y el IC por bootstrap.

## Lo que hace el repo original (lo que se replica)

Verificado en `cls_model.py`, `transformer_utils.py`, `metrics.py` y `conf/experiment/eos*.yaml` de `SocialComplexityLab/life2vec`:

- **Cabeza**: no usa `[CLS]` sino un *pooling* por atención sobre todos los tokens (`AttentionDecoder`). Primero aplica `Linear(280,280)`, Swish, ScaleNorm y AlphaDropout 0,1. Después pondera los tokens con `tanh(W·h)·contexto` y SigSoftmax, enmascarando el padding, y termina en `Linear(280,2)`.
- **Encoder**: no se congela. Se congelan los embeddings de tokens salvo los especiales.
- **Optimizador**: RAdam, `lr=0,01` con decaimiento por capa 0,95 y `ExponentialLR(0,8)` por época.
- **Lote y épocas**: lote efectivo 64; una época son 9.600 muestras, con 5 como mínimo y 35 como máximo. *Early stopping* sobre el AUL de validación (paciencia 5).
- **Recorte**: de gradiente por valor, en 1.
- **Pérdida**: la "asimétrica PU" del paper, en el código, es **entropía cruzada balanceada** (0,5 por clase): el término de penalización está comentado. Además, el *sampler* arma lotes 50/50.
  - Esto resuelve la duda del PLAN sobre el enfoque PU: replicar el código es usar entropía cruzada balanceada. Nuestra etiqueta es observada (no aparecer en 2019 está en los datos), así que no hace falta corregirla.
- **Bug del original**: el *weight decay* nunca se aplica al encoder. Lo replicamos como efecto (0 en el encoder, 0,01 en la cabeza) y lo documentamos.
- **Métricas**:
  - AUL (área bajo la curva de lift, que equivale a c/2 + (1 − c)·AUC);
  - C-MCC, F1 y exactitud corregidas, con el umbral 0,5 sobre p(positivo).
  - Con etiquetas observadas, el parámetro α del C-MCC vale 0 y la medida se reduce al MCC común. Se implementan las fórmulas generales y se reportan con α = 0.
- **Baselines**:
  - tabla de vida: regresión logística sobre edad y sexo, con 3 parámetros;
  - regresión logística sobre conteos de tokens normalizados más sexo y edad;
  - FFNN: 3 bloques de 646 con dropout 0,35, `lr=1,7e-4`;
  - GRU bidireccional: 3 capas, 256 en total, dropout 0,27, mismos embeddings y misma cabeza, sin pre-entrenar, `lr=1e-3`;
  - mayoría y azar.
- **Tabla A2 del paper** (mortalidad, C-MCC / AUL): L2V 0,413 / 0,845; GRU 0,369 / 0,834; FFNN 0,340 / 0,822; logística 0,149 / 0,735; tablas de vida 0,059 / 0,650. **Se compara el orden y las distancias relativas entre modelos, no los valores absolutos**, porque la tarea es otra.

## 6.1 Cohorte y etiquetas

- `src/labor2vec/tareas/t1.py` (`python -m labor2vec.tareas.t1`) calcula la cohorte y las etiquetas con DuckDB sobre `data/processed/panel.parquet`:
  - meses distintos con registro en 2015 (`mes_idx` 228-239);
  - edad al corte;
  - `y_2019` (ningún mes en 276-287) y `y_2021` (ningún mes en 300-311).
  - Escribe `data/processed/tareas/t1.parquet` con `id_trabajador, split, sexo, fnacim, edad_corte, y_2019, y_2021`.
- Informe de control: tamaño y tasa por split, por sexo y por tramo de edad. Si los números se alejan mucho de los de PLAN.md (122.492 y ~16 %), se revisa antes de seguir.

## 6.2 Datos para clasificación

- En `src/labor2vec/tareas/datos.py`:
  - `DatasetT1`: reutiliza `DatasetPretrain.documento(k, rng, augmentar, con_sop=False)` de `tokens/datos.py`, filtra a la cohorte y le pega la etiqueta.
  - `collate` sin MLM ni SOP: `input_ids`, `padding_mask` y `target`.
  - `SamplerBalanceado`: muestreo 50/50 con reposición y lotes agrupados por largo, como `SamplerPorLargo`.
  - Las etiquetas se leen con pyarrow, para no importar polars en el camino de entrenamiento: hay una prueba que lo exige.
- Augmentación sólo en train, la misma del pre-entrenamiento (`Opciones`). En val y test el documento es determinista.

## 6.3 Modelos

- `src/labor2vec/modelo/clasificador.py`, un `LightningModule`:
  - carga el encoder desde `modelo_final.pt` (reutiliza `analisis/embeddings.py:cargar_modelo`);
  - `AttentionDecoder` portado del original;
  - pérdida, optimizador y congelamiento como arriba;
  - registra `val/aul`, `val/auc` y `val/mcc`.
  - La opción `desde_cero=True` da el **mismo transformer sin pre-entrenar**: es la comparación que muestra cuánto aporta el pre-entrenamiento.
- `src/labor2vec/modelo/gru.py`: el GRU del original sobre el módulo `Embeddings` del backbone, con la misma cabeza.
- `src/labor2vec/ajustar.py`: CLI con YAML (`configs/ajuste_t1.yaml`, `configs/gru_t1.yaml`).
  - Reutiliza de `entrenar.py` `cargar_config`, `ParadaSegura`, `GuardiaNaN` y `Rendimiento`.
  - bf16, lote 32 × 2 acumulaciones, *early stopping* por `val/aul` y guardado del mejor checkpoint.
- `src/labor2vec/tareas/tabulares.py`:
  - genera los atributos de conteo (519 tokens normalizados, más sexo y edad min-max);
  - corre la tabla de vida y la logística con scikit-learn, y el FFNN con torch.
  - Para el FFNN se usa la configuración del paper, sin búsqueda de hiperparámetros.

## 6.4 Métricas y evaluación

- `src/labor2vec/tareas/metricas.py`:
  - AUL, AUC-ROC y PR-AUC;
  - C-MCC, F1 y exactitud corregidas con α y β generales;
  - MCC con umbral 0,5 y con el umbral óptimo en validación;
  - Brier y curva de calibración, con corrección de prevalencia porque el entrenamiento es balanceado;
  - IC al 95 % por **bootstrap estratificado** (1.000 réplicas).
- `src/labor2vec/tareas/evaluar.py`:
  - guarda las predicciones de test de cada modelo en un parquet;
  - arma la tabla tipo A2 para `y_2019` y la prueba extra con `y_2021`;
  - escribe `informes/fase6.md` y las figuras en `informes/fase6/`: curvas ROC y de lift, calibración y AUL por sexo y por edad.
- El test (~18 k) se evalúa completo, no submuestreado: el original enriquecía con positivos.

## 6.5 Cómputo

- **Mendieta** (partición `short`, ≤ 1 h por job):
  - 3 fine-tunings de L2V, 3 del GRU y 1 transformer desde cero, unos 7 jobs.
  - Estimación: 35 épocas × 9.600 secuencias a ~215 por segundo ≈ 26 min, más la validación. Probablemente quepan en un job cada uno; si no, se encadenan con `afterany` y `--reanudar`.
  - Hay que crear `scripts/cluster/ajustar.sbatch` (copia de `preentrenar.sbatch`), con `--mail-type=BEGIN,END,FAIL`. Hay que sincronizar `data/processed/tareas/` con `sincronizar_mendieta.sh`.
- **Local**: la cohorte y las etiquetas (DuckDB, segundos), los baselines tabulares (minutos) y la evaluación final. **Pido OK antes de cada corrida pesada y antes de encolar cada tanda de jobs.**
- No hay dependencias nuevas: scikit-learn ya está en el extra `analisis`, que es local.

## 6.6 Pruebas

- La cohorte y las etiquetas con el panel sintético de `tests/conftest.py`, ampliado con trabajadores que tienen meses posteriores a 2015: los casos de borde de edad, los 12 meses de 2015 y la presencia en 2019 y 2021.
- El documento de T1 nunca contiene eventos posteriores a 2015-12.
- El *sampler* balanceado da lotes de ~50/50.
- Métricas contra casos a mano:
  - el AUL de un clasificador perfecto y de uno al azar;
  - el AUL coincide con `c/2 + (1−c)·AUC`;
  - con α = 0, el C-MCC es igual al MCC de scikit-learn.
- El clasificador hace una pasada `fit` corta en CPU con el modelo chico (`HPARAMS_MINIMO`): la pérdida baja y el checkpoint se recarga.
- El camino de entrenamiento sigue sin importar polars.

## Fuera de esta fase

- Variantes de la etiqueta (horizonte de 1 año, "nunca vuelve") y el control sin atributos "último valor": quedan para las ablaciones de la fase 8.
- T2-T4: plan aparte cuando T1 esté cerrada.
- Búsqueda bayesiana de hiperparámetros de los baselines (el original la hizo con 25 evaluaciones): se agrega sólo si los baselines quedan muy lejos.

## Entregables y criterio de aceptación

1. Código con las pruebas en verde (las 54 actuales más las nuevas) y comandos reproducibles.
2. `data/processed/tareas/t1.parquet` con su informe de control.
3. `informes/fase6.md` con:
   - la tabla tipo A2: modelos × (C-MCC, AUL, AUC, PR-AUC, F1 y exactitud corregidas, con su IC);
   - la comparación con el paper;
   - el aporte del pre-entrenamiento (L2V contra el transformer desde cero);
   - la prueba de 2021;
   - una lectura honesta de los resultados.

No hay un umbral de aprobación: la fase se acepta cuando todos los modelos están evaluados sobre el mismo test con sus IC.

## Orden de trabajo

1. Guardar este plan como `plans/PLAN_fase_6.md`.
2. Escribir las secciones 6.1, 6.2, 6.4 y 6.3 con sus pruebas.
3. Correr la cohorte y las etiquetas en local (con OK) y revisar los números con vos.
4. Encolar los jobs de Mendieta (con OK) y correr los baselines tabulares en local (con OK).
5. Evaluar, escribir el informe y commitear cuando lo pidas.

## Verificación

- `.venv/bin/python -m pytest tests -q` en verde.
- `python -m labor2vec.tareas.t1` reproduce los tamaños y tasas del informe de control.
- Una corrida de humo en CPU de `python -m labor2vec.ajustar --config configs/humo_t1.yaml` sobre `corpus_muestra`.
- En Mendieta, el `val/aul` del primer job sube por encima del de la logística antes de lanzar el resto.
