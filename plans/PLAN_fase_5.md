# Subplan: fase 5 (evaluación del espacio de conceptos)

Estado: borrador para aprobación (2026-10-20). Complementa [PLAN.md](PLAN.md), secciones 5 y 6.

**Objetivo**: ver qué aprendió el modelo sobre los *conceptos* (los 519 tokens del vocabulario) mirando la matriz de embeddings de tokens, y medir si lo aprendido es estable entre semillas. Es análisis puro de una matriz de 519 × 280: **corre en el contenedor local, sin GPU ni cluster, en segundos**.

## Qué se analiza y con qué modelos

- **Modelo principal**: `pre_150ep` (semilla 0). Perplejidad MLM 4,73.
- **Robustez**: los tres modelos de 150 épocas (semillas 0, 1 y 2).
- **Efecto del entrenamiento**: `pre_30ep` (perplejidad 6,36) como comparación extra, igual que el repo original usa modelos adicionales en sus pruebas.
- Matriz de embeddings: `transformer.embedding.token.weight` de `modelo_final.pt` (la matriz de entrada, centrada por el propio modelo), y se le resta la media de los tokens no especiales. Con *weight tying* la matriz de salida es la misma. Los 10 tokens especiales no entran en ningún análisis.

## Lo que hace el repo original (lo que se replica)

Revisado en `analysis/embedding/emb_robust.ipynb`, `emb_hubs.ipynb` y `analysis/visualisation/concept_space.ipynb` de `SocialComplexityLab/life2vec`:

1. **Robustez entre modelos**: matrices de distancia coseno entre todos los pares de tokens para cada modelo; correlación de Spearman entre los triángulos superiores de dos modelos; comparación contra dos nulos (embeddings con las columnas permutadas de forma independiente, y embeddings gaussianos); **test de permutación Monte Carlo** (se permutan filas y columnas de una matriz, 5.000 iteraciones, una cola) y corrección de Benjamini-Hochberg entre las comparaciones.
2. **Hubness**: grafo de los 5 vecinos más cercanos por coseno; distribución del grado de entrada; mediana de grado de entrada y cantidad de nodos con grado 0, por categoría de tokens.
3. **Mapa de conceptos**: PaCMAP en 2D (`MN_ratio=1`, `FP_ratio=10`, distancia angular, `lr=0.5`, `random_state=0`), con recuadros de zoom por categoría; tablas de vecinos por distancia coseno.

## 5.1 Verificaciones propias de la MLER

El PLAN.md señala que la jerarquía de la MLER permite validaciones más fuertes que las del paper. Cada una se compara contra un nulo (etiquetas permutadas 1.000 veces) y se reporta con su valor, el nulo y el p-valor empírico:

| Categoría (tokens) | Pregunta | Medida |
|---|---|---|
| Salario `REM_000` a `REM_100` (101) | ¿los percentiles salen ordenados en una curva? | Spearman entre el percentil y la primera componente principal; proporción de los 4 vecinos más cercanos (entre tokens de salario) que están a ≤ 5 percentiles; distancia de `REM_000` (remuneración cero) al resto |
| Año de nacimiento (77) | ¿los años salen ordenados? | las mismas dos medidas |
| Tamaño de empresa (4) y antigüedad de la empresa (4) | ¿están ordenados? | Spearman entre la distancia coseno de cada par y la distancia ordinal |
| Rama `RAMA_xxx` (282) | ¿las ramas vecinas comparten sector? | pureza de los 5 vecinos más cercanos respecto de la `letra` y del `r32` |
| Provincia (25) | ¿las vecinas comparten región? | pureza de los 3 vecinos respecto de la `region` del descriptor provincial del MTEySS (GBA, Pampeana, NOA, NEA, Cuyo, Patagonia) |

- Las ramas `RAMA2_xx` (14) y las que no tienen vecinos de su clase se tratan aparte.
- La correspondencia `r34 → letra` se saca de la tabla de relaciones. Tamaño y antigüedad tienen sólo 4 niveles: ahí se reporta el valor sin test de significancia.
- Se reportan los resultados tal cual salgan, incluso los negativos (ver riesgos en PLAN.md sección 7).

## 5.2 Robustez y hubness

- **Robustez** (los tres modelos de 150 épocas): Spearman entre pares de modelos de las 509 × 508 / 2 distancias coseno, contra los dos nulos, con test de permutación (5.000 iteraciones) y FDR. Se repite dentro de cada categoría (por ejemplo sólo salarios, sólo ramas). Se agrega la comparación contra `pre_30ep` para ver cuánto cambia el espacio con más entrenamiento.
- **Hubness**: grafo de 5 vecinos, grado de entrada, asimetría (*skewness*) de su distribución, mediana por categoría y proporción de tokens con grado de entrada 0.

## 5.3 Figuras

- Mapa PaCMAP de los 509 tokens, coloreado por categoría, con recuadros: salarios con degradé por percentil, años de nacimiento por año, provincias por región, ramas por `letra`.
- Para varios tokens ejemplo, tablas de los 10 vecinos más cercanos con las descripciones de los descriptores (por ejemplo ramas).
- Gráficos de dispersión entre modelos de las distancias de 10.000 pares al azar, como en el original.
- Las figuras se guardan en `informes/fase5/` como PNG (unos 100-300 KB cada una).

## 5.4 Código

- `src/labor2vec/analisis/embeddings.py`: carga `modelo_final.pt` (reconstruye el modelo con los hiperparámetros guardados y extrae la matriz) y la tabla de tokens con categoría y descripción.
- `src/labor2vec/analisis/metricas.py`: orden, pureza, robustez, hubness y los nulos.
- `src/labor2vec/analisis/figuras.py` y `python -m labor2vec.analisis.conceptos`: corre todo y escribe `informes/fase5.md` y las figuras.
- **Dependencias nuevas** (extra opcional `analisis` en `pyproject.toml`): `scipy`, `scikit-learn`, `matplotlib`, `pacmap`. Hoy ninguna está instalada. `pacmap` depende de `numba`, y hay que comprobar que instale en Python 3.13; si no, se usa UMAP o t-SNE y se avisa.
- No se instala nada hasta que apruebes el plan.

## 5.5 Pruebas

- Embeddings sintéticos con estructura conocida: una línea ordenada da Spearman cercano a 1; grupos separados dan pureza alta; ruido da valores dentro del nulo.
- Los tres nulos (columnas permutadas, gaussiano, etiquetas permutadas) reproducen las propiedades esperadas.
- El cargador extrae la misma matriz que `model.transformer.embedding.token.weight` de un modelo chico guardado y recargado.
- Los tokens especiales nunca entran en las métricas.

## Entregables y criterio de aceptación

1. Código con pruebas en verde y un comando que reproduce todo.
2. `informes/fase5.md` con las tablas de 5.1 y 5.2, las figuras y una lectura honesta de qué se ve y qué no.
3. Las cifras y figuras de los tres modelos de 150 épocas, más la comparación con `pre_30ep`.

**No hay un umbral de aprobación**: la fase se acepta cuando las medidas están calculadas contra sus nulos y el informe dice qué se encontró. Las expectativas razonables (a verificar, no a asumir) son que los percentiles de salario y los años de nacimiento salgan ordenados y que las correlaciones entre semillas sean significativamente mayores que las de los nulos.

## Decisiones que necesito de vos

1. **Modelos**: principal `pre_150ep` (semilla 0), tres semillas para robustez y `pre_30ep` como comparación. **Recomiendo** eso.
2. **Dependencias nuevas**: instalar `scipy`, `scikit-learn`, `matplotlib` y `pacmap` como extra opcional, con UMAP o t-SNE como reemplazo si `pacmap` no instala en Python 3.13. **Recomiendo** que sí.
3. **Figuras en el repo**: versionar los PNG de `informes/fase5/` (unos pocos MB en total). **Recomiendo** que sí.
4. **Regiones de provincias**: usar la columna `region` del descriptor del MTEySS, que ya está en `data/processed/descriptores/provi.parquet` (GBA, Pampeana, NOA, NEA, Cuyo, Patagonia). **Recomiendo** esa fuente en vez de armar una propia.
