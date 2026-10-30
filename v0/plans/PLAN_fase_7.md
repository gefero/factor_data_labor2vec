# Subplan: fase 7 (espacio de personas: proyección, TCAV y saliency)

Estado: aprobado (2026-10-21). Complementa [PLAN.md](PLAN.md) (sección 5, hoja de ruta fila 7). 

## Contexto

Las fases 6 y 6b mostraron *cuánto* predice el modelo. La fase 7 pregunta *qué* aprendió de las personas. Tiene tres piezas, como en el paper:

1. **Mapa de personas:** cada trabajador del test se resume en un vector y se proyecta en 2D (DensMAP y PaCMAP).
2. **TCAV:** mide si conceptos humanos (sexo, rama, tamaño de empresa, nivel salarial, antigüedad) son *direcciones* en ese espacio y si mover a alguien en esa dirección cambia la predicción.
3. **Saliency (SmoothGrad):** dice qué eventos de la historia de cada persona pesan más en su predicción (qué tipo de token y cuántos meses antes del corte).

Tarea principal: **T1** (lo que dice PLAN.md). Como el mismo código sirve para cualquier tarea binaria, se corre también **T3** (cambio de rama): es donde el pre-entrenamiento le gana claro a los baselines y donde interesa ver si el modelo mira el *orden y el momento* de los cambios de empresa. Hoy esa es una hipótesis sin probar (fase 6b). T2 y T4 quedan afuera porque son multiclase.

Modelo: el ajuste fino de semilla 0 de cada tarea (`t1_l2v_lr0001`, `t3_l2v_s0`). Sus `modelo_final.pt` están sólo en el cluster. Como control se usa el encoder pre-entrenado sin ajustar (`pre_150ep`), para ver qué parte de la organización ya estaba antes del fine-tuning.

## Definiciones

- **Vector de persona** (el "person summary" del paper): la salida del *pooling* por atención de la cabeza (`AttentionDecoder`), es decir, el vector de 280 dimensiones que entra a la capa lineal final. Para el control pre-entrenado, que no tiene cabeza, se usa el promedio enmascarado de la última capa del encoder.
- **Capa de TCAV:** la salida de la **última capa del encoder** resumida por persona como promedio enmascarado sobre tokens. El vector de persona no sirve como capa porque después de él sólo hay una capa lineal: la derivada sería la misma para todos y el TCAV saldría trivialmente 0 o 1. La sensibilidad de una persona a un concepto con vector `v` es la derivada del logit (clase 1 − clase 0) cuando se suma `ε·v` al estado de *todos* sus tokens. Esa derivada es `g·v`, con `g` = suma sobre tokens del gradiente del logit respecto de esa capa. Se guarda `g` por persona (un backward cada una) y después cualquier CAV se evalúa localmente en segundos. También se guarda `g` de la penúltima capa, para comparar.
- **Conceptos** (estado en 2015, del panel), para toda la cohorte de T1:
  - sexo;
  - `letra` de la relación principal (las 6 letras más frecuentes, cada una contra el resto);
  - `tramo_empleo` (empresa grande, más de 200, contra chica, hasta 25 o el corte que dé grupos razonables);
  - tercil del percentil salarial medio de 2015 (alto contra bajo);
  - antigüedad en el empleo registrado (años desde el primer mes en el panel; tercil alto contra bajo);
  - edad al corte (tercil).
- **Protocolo TCAV** (Kim et al. 2018, como el original):
  - Por concepto, 50 CAVs: regresión logística entre 500 personas del concepto y 500 al azar, tomadas del split **val**.
  - Nulo: 50 CAVs de "azar contra azar".
  - **Puntaje TCAV** = fracción de las personas del test con `y = 1` (y aparte, con `y = 0`) que tienen `g·v > 0`.
  - Significancia: t de Welch del concepto contra el nulo, con corrección de Bonferroni.
  - Se reporta también la exactitud de cada CAV en datos retenidos: indica si el concepto es linealmente legible en el espacio, aunque no influya en la predicción.
- **SmoothGrad:**
  - Muestra de 2.000 personas del test, estratificada por `y` y por quintil de `p`.
  - 25 copias por persona, con ruido gaussiano en los embeddings de token de la entrada, σ = 0,15 × el rango de esos embeddings.
  - Puntaje por token = |gradiente · embedding| promediado sobre las copias, normalizado para que sume 1 en cada persona.
  - Se agrega por categoría de token (salario, rama, tamaño, antigüedad, provincia, fondo…), por meses antes del corte y por cercanía a un cambio de empresa. Esto último sirve para la hipótesis de T3.
  - Más 4-6 historias de ejemplo dibujadas como línea de tiempo coloreada.
- **Mapa:**
  - DensMAP (`umap-learn`, `densmap=True`) y PaCMAP sobre los ~18 k vectores del test.
  - Paneles coloreados por `p`, por `y` y por cada concepto.
  - Medida numérica: pureza de los k vecinos para `y` y para cada concepto (se reusa `analisis/metricas.pureza`), en el modelo ajustado y en el control pre-entrenado.

## Implementación

1. **`src/labor2vec/analisis/personas.py`**, con tres subcomandos:
   - `conceptos` (local; duckdb sobre `data/processed/panel.parquet`): escribe `data/processed/tareas/conceptos_personas.parquet` para la cohorte de T1. Reusa el patrón de `tareas/derivadas.sql_estado`: se le agrega un parámetro para no filtrar a los que siguen en 2019 y para devolver sólo 2015, más el primer `mes_idx` del panel.
   - `extraer` (GPU, en Mendieta): carga el clasificador desde `modelo_final.pt` (`ClasificadorT1` + `load_state_dict`, el mismo formato que escribe `ajustar.py`). Recorre val y test con `DataModuleT1` (`ajustar.preparar`, `pre_cargado=False`). Guarda en un `.npz` por modelo: `id`, `y`, logit, vector de persona, promedio de la última y la penúltima capa, `g` de esas dos capas y, para la muestra, los puntajes SmoothGrad por token junto con el token, el mes y la posición. Las salidas intermedias se capturan con *hooks* (`register_forward_hook` en `encoder.encoders[-1]` y `[-2]`, y en `encoder.embedding.token` para inyectar el ruido de SmoothGrad), sin tocar el código copiado de life2vec-light. Se usa fp32 y `eval()` (sin dropout; las proyecciones del Performer quedan fijas).
   - `analizar` (local, CPU): proyecciones, TCAV, agregados de saliency, figuras y `informes/fase7/resultados.md`. Reusa `analisis/figuras.py` (`_guardar`, el estilo de los mapas) y `analisis/embeddings.cargar_conceptos` para la categoría y la descripción de cada token.
2. **`scripts/cluster/personas.sbatch`**: copia de `ajustar.sbatch` (comprobación de importaciones, `short`, 1 h, 1 GPU) que corre `python -m labor2vec.analisis.personas extraer --modelo runs/NOMBRE --tarea t1|t3`. Tres jobs: T1, T3 y el control pre-entrenado (`--sin-cabeza`, sin `g` ni saliency). Con `--mail-type=BEGIN,END,FAIL`. Antes de encolar se sube el código con `sincronizar_mendieta.sh` y se te pide el OK. Después se traen sólo los `.npz` (estimado: decenas de MB cada uno).
3. **`pyproject.toml`**: agregar `umap-learn` al extra `analisis` (sólo local) y `uv sync`.
4. **Tests** (`tests/test_personas.py`, con un modelo diminuto como `test_ajuste.py`):
   - `g·v` coincide con la derivada por diferencias finitas al sumar `ε·v` a todos los tokens;
   - SmoothGrad devuelve un puntaje por token real (cero en el relleno) que suma 1;
   - TCAV sobre datos sintéticos (un concepto que mueve el logit y otro que no) da puntaje alto y significativo en el primero y no en el segundo;
   - la tabla de conceptos tiene una fila por trabajador de la cohorte.
5. **`informes/fase7.md`**, con el mismo formato que `fase5.md` y `fase6b.md`: qué se hizo, mapas, tabla TCAV, perfiles de saliency, historias de ejemplo y conclusión en lenguaje simple. PLAN.md decía "notebook". Se reemplaza por script + informe para mantener la línea de las fases 5 y 6.

## Costos y avisos

- Local: la tabla de conceptos es un barrido del panel con duckdb, similar a `tareas.derivadas` (minutos). DensMAP y PaCMAP sobre 18 k × 280 tardan unos minutos de CPU. Se pide el OK antes de cada uno.
- Cluster: 3 jobs cortos. El más caro es SmoothGrad: 2.000 × 25 pasadas con backward, estimado bien por debajo de 1 h en la A30. Si no entra en el tiempo, se baja a 1.000 personas.
- Riesgo conocido: los atributos "último valor" (rama, provincia, antigüedad) pueden filtrar información posterior al corte. Si la saliency de T3 se concentra en esos tokens, se va a decir explícitamente; el control de verdad queda para la fase 8.

## Verificación

1. `uv run pytest` (los 100 tests actuales + los nuevos) y `ruff check` sobre los archivos nuevos.
2. Prueba de humo local de `extraer` con `configs/humo_cpu.yaml` y 64 personas: forma de los arreglos, sin NaN, `logit` igual al de `ClasificadorT1.predecir`.
3. En el cluster, chequeo de coherencia: la AUL recalculada con los logits extraídos debe coincidir con la de `runs/t1_l2v_lr0001/predicciones_test.parquet` (0,619 ± redondeo).
4. `analizar` produce `informes/fase7/` completo. Se revisan las figuras a mano antes de escribir el informe.
