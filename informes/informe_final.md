# labor2vec: informe final (fases 0 a 8)

Escrito para quien no siguió el proyecto. Los detalles de cada fase están en `informes/fase{0..8}*.md`; el punteo de lo que se puede y no se puede decir en un paper, en [notas_paper.md](notas_paper.md).

## 1. Qué es y qué se hizo

**life2vec** (Savcisens et al., 2023) es un modelo tipo transformer que lee la historia de vida de una persona, escrita como una secuencia de eventos con fecha, y aprende un vector que la resume. Se entrena primero sin etiquetas (completar tokens ocultos y ordenar fragmentos de la historia) y después se ajusta a tareas concretas.

**labor2vec** adapta esa idea a la **MLER**, una muestra longitudinal de empleo asalariado registrado en Argentina (el 3 % de los trabajadores, 596.591 personas; quedan 575.642 tras excluir a quienes no tienen sexo o año de nacimiento). Cada mes de cada relación laboral es un evento descripto por remuneración (en percentiles), tamaño de empresa, rama de actividad, provincia y antigüedad. La pregunta del proyecto: ¿aprende el modelo estructura real del mercado laboral y le sirve eso para predecir, comparado con métodos más simples?

Todo está en castellano, con el cómputo pesado en el cluster Mendieta del CCAD (GPU A30). El código está en `src/labor2vec/` y los resultados en `informes/`.

## 2. Cómo se hizo

| Fase | Qué se hizo |
|---|---|
| 0-1 | Entorno; modelo de base copiado de life2vec-light; ingesta de la MLER, población, percentiles de salario, informes de calidad |
| 2-3 | Vocabulario de 519 tokens; documentos por trabajador (un bloque de 5 tokens por evento + `[SEP]`); tarea de enmascarado (MLM, 30 %) y de orden de fragmentos (SOP); entrenamiento y métricas; alineación con el repo completo de life2vec |
| 4 | Pre-entrenamiento: 150 épocas (≈ 5 h), 3 semillas |
| 5 | Análisis del espacio de **conceptos** (los 519 tokens) |
| 6, 6b | Ajuste fino a cuatro tareas (T1-T4) contra líneas de base |
| 7 | Espacio de **personas**: mapas, TCAV y saliency |
| 8 | Controles temporales, variantes de tareas, ablación sin atributos de último valor |

Las cuatro tareas, con la historia hasta 2015-12 y la etiqueta a 2019 (T1 también a 2021):

- **T1**: ¿sale del empleo registrado? (cohorte de 122.492 personas de 25 a 55 años; 16 % sale hasta 2019).
- **T2**: quintil de salario en 2019 (5 clases).
- **T3**: ¿cambia de rama de actividad? (11 %).
- **T4**: tamaño de la empresa: más chica, igual o más grande.

T2-T4 usan los 102.885 de la cohorte que siguen registrados en 2019.

**Qué se comparó:** el transformer pre-entrenado (3 semillas de ajuste fino), el mismo sin pre-entrenar, un GRU, una FFNN y una regresión logística sobre variables resumidas, la logística con el estado actual, una tabla de vida (edad y sexo), la persistencia, la mayoría y el azar. Intervalos de confianza al 95 % por bootstrap sobre las personas del test.

## 3. Resultados

### 3.1 El pre-entrenamiento y el espacio de conceptos

- Perplejidad de validación del enmascarado 4,73, contra 81,4 de predecir por frecuencias; AUC del SOP 0,865. Entre 3 semillas la perplejidad varía ~1,5 %.
- Los percentiles de salario forman una curva ordenada (vecinos cercanos en percentil: 1,00 contra 0,10 del azar). Los años de nacimiento están ordenados a nivel local. Ramas vecinas comparten sector (0,41 contra 0,22 del azar); provincias vecinas, región (0,49 contra 0,15). Las distancias son reproducibles entre semillas (Spearman 0,72), más en salarios y años que en ramas finas.

### 3.2 Predicción (test, media de 3 semillas)

| Tarea (métrica) | Transformer pre-entrenado | Mejores métodos simples | Sin pre-entrenar |
|---|---|---|---|
| T1 (AUL) | 0,619 | 0,617 (logística), 0,616 (GRU) | 0,563 |
| T2 (QWK) | 0,830 | 0,828 (GRU), 0,822 (persistencia) | 0,745 (*) |
| T3 (AUL) | **0,725** | 0,702 (GRU), 0,687 (FFNN) | 0,639 |
| T4 (AUC macro) | **0,772** | 0,746 (GRU), 0,691 (logística) | 0,686 |

(*) Esa corrida no convergió en 35 épocas; ver 3.4.

En lenguaje simple: **frente a métodos simples que no conocen el tiempo, el transformer pre-entrenado empata en T1 y T2 y los supera en T3 y T4** (ver 3.6 para lo que pasa cuando los métodos simples también reciben variables de tiempo). En T2 el estado actual (el quintil de 2015) ya predice casi todo: repetirlo da 0,822.

Variantes de la pregunta (fase 8): el transformer también gana al formular T2 como «baja / igual / sube de quintil» (AUC macro 0,741 contra 0,707 del GRU) y T4 como «cambia o no el tamaño» (AUL 0,702 contra 0,682), y empata al preguntar a qué rama va quien cambia (T3d). Patrón: gana en «se movió algo», empata en niveles.

### 3.3 ¿Por qué gana en T3 y T4? Dos controles

**Orden y momento.** Se destruyó el tiempo (fechas y edades permutadas entre eventos) y se reajustó. En T3 se pierde poco (0,725 a 0,710; sigue sobre el GRU). En T4 se pierde bastante (0,772 a 0,738; queda bajo el GRU, sobre la logística). **Para T4 hay respaldo de que importa el orden y el momento; para T3 no.**

**Fuga de información.** Rama, provincia y antigüedad se toman del último registro de cada relación y podrían traer datos posteriores al corte. Se pre-entrenó y reajustó todo sin esos atributos. Todos los modelos pierden algo (T3 −0,06 a −0,10; T4 −0,02 a −0,05; T1 y T2 poco) y el transformer **no es el que más cae**: su ventaja en T3 y T4 sobrevive (T3 0,661 contra 0,624; T4 0,749 contra 0,699). Esa caída es una cota superior de la fuga, porque la rama de 2015 es información legítima para predecir si se cambia de rama; no hay forma limpia de separarlas (el grupo que no puede tener fuga es el 3,8 % de la cohorte).

### 3.4 Aporte del pre-entrenamiento

Pre-entrenar suma +0,06 a +0,09 en T1, T3 y T4, también sin los atributos de último valor. **En T2 la cifra original estaba inflada:** la corrida sin pre-entrenar no había convergido (curva todavía subiendo a las 35 épocas); la que sí convergió da 0,818, igual al pre-entrenado (0,822). En T2c el modelo sin pre-entrenar colapsó a la clase mayoritaria. No hay evidencia de que pre-entrenar ayude en T2.

### 3.5 Espacio de personas (fase 7)

Cada persona resumida en un vector y proyectada a 2D. El mapa se organiza por **quiénes son**, no por lo que les pasa: salario, rama, tamaño, edad y antigüedad ordenan el mapa (por ejemplo T1 por salario: pureza de 5 vecinos 0,73 contra 0,33; T3 por rama: 0,62 contra 0,20); la salida del empleo casi no (0,742 contra 0,729). El ajuste fino agrega edad y antigüedad, que el modelo sin ajustar casi no tenía. El saliency indica que el modelo se apoya sobre todo en la historia reciente: los últimos 24 meses pesan ~2 veces lo que les toca por cantidad de tokens, lo de hace más de 5 años ~0,55. **TCAV quedó degenerado** (puntajes 0 o 1, gradientes casi idénticos entre personas) y no se puede usar como resultado.

### 3.6 Control final: métodos simples con variables de tiempo

Los métodos simples de arriba solo ven proporciones de tokens, sin saber *cuándo* pasó cada cosa. Se les dieron variables de tiempo resumidas (antigüedad, meses desde el último cambio de empresa, tamaño o rama, cambios en los últimos 1, 2 y 5 años, huecos de registro y tendencia salarial) y se volvió a comparar. Detalle en `informes/fase8.md`, sección 5.

| Tarea (métrica) | Transformer | Mejor simple con tiempo | Diferencia |
|---|---|---|---|
| T1 (AUL) | 0,619 | 0,634 (logística) | **−0,016**: gana el simple |
| T2 (QWK) | 0,830 | 0,821 (logística con estado) | +0,008 |
| T3 (AUL) | 0,725 | 0,731 (logística) | −0,006: empate |
| T4 (AUC macro) | 0,772 | 0,741 (logística con estado) | +0,031 |

Sin los atributos de último valor: T1 −0,013 (gana el simple), T3 −0,031 (gana el simple), T4 +0,011 (el transformer apenas arriba).

**Lectura honesta:** buena parte de lo que parecía ventaja del transformer en T3 se explica con magnitudes simples de tiempo que un modelo lineal aprovecha igual de bien, y en T1 esos métodos lo superan. **La única ventaja que sobrevive en todas las condiciones es la de T4** (y las variantes T2c y T4c), aunque más chica. Qué parte de esa ventaja es secuencia propiamente dicha no se midió. El pre-entrenamiento y la estructura del espacio de conceptos no se ven afectados por este control.

### 3.7 Comparación con el paper original

Contraste hecho leyendo el paper original (Savcisens et al., 2023, arXiv 2306.03009, copia en `data/docs/2306.03009v1.pdf`, con su material suplementario). El paper predice mortalidad a 4 años y rasgos de personalidad (y emigración en el SI) con datos de registros daneses; no hay tareas en común con las nuestras.

| # | Prueba | ¿Está en el paper? | Comparación |
|---|---|---|---|
| 1 | Pre-entrenamiento | El diseño sí (MLM al 30 % y SOP); resultados casi no | El paper elige la configuración por una «perplejidad total» que mezcla MLM (0,7) y SOP (0,3), con valores de 1,83 a 2,04 (Tabla A5), no comparables con nuestros 4,73. No compara con una línea de base de frecuencias, no reporta el AUC del SOP ni la variación entre semillas |
| 2 | Espacio de conceptos | Sí | Coincide: los 100 niveles de ingreso forman una curva ordenada, los años de nacimiento vecinos son contiguos y los tokens se agrupan por clasificación (ICD-10 e ISCO-08 allá, ramas y sectores acá). Robustez: el paper usa 3 modelos de 10 épocas con inicializaciones y subconjuntos distintos y un test de aleatorización (p ≈ 3·10⁻⁴); nosotros, 3 semillas de 150 épocas y Spearman 0,72. **Diferencia:** el paper no encuentra *hubs* (el token más conectado es vecino de menos del 2 % de los tokens); nosotros sí (los `RAMA2_xx`, y más con más entrenamiento) |
| 3 | Predicción contra métodos simples | Sí, el diseño | Mismas líneas de base (mayoría, azar, tabla de vida, logística, FFNN, RNN-GRU) y métricas parecidas (C-MCC, AUL, exactitud y F1 corregidas, bootstrap). En mortalidad: C-MCC 0,413 contra 0,369 (GRU) y 0,340 (FFNN), la ventaja del «11 %», que es relativa; en AUL la diferencia absoluta es chica: 0,845 contra 0,834 y 0,822. Nosotros no tenemos ventaja en T1 y T2 y sí en T3 y T4. Para personalidad el paper dice que un RNN extrae casi la misma señal y que «buena parte de lo que hace versátil a life2vec es el dataset mismo» |
| 4 | Pre-entrenar o no | **No** | El paper no compara contra un transformer sin pre-entrenar |
| 5 | Espacio de personas, TCAV y saliency | Sí, con resultados distintos | Mismos métodos (DensMAP, TCAV, SmoothGrad). El mapa del paper se organiza como un continuo de probabilidad de mortalidad; el nuestro, por quiénes son (salario, rama, edad) y casi no por el resultado. Su TCAV funciona (directivo, ingreso alto, varón y diagnóstico mental se separan de las direcciones al azar, Mann-Whitney); el nuestro quedó degenerado. Su saliency es un ejemplo token por token; el análisis agregado por antigüedad del evento es nuestro |
| 6 | Tiempo destruido | **No** | El paper usa ruido temporal de ±5 días solo como aumentación; no mide cuánto aporta el tiempo |
| 7 | Sin atributos de último valor | **No** | No hay control de fuga de información |
| 8 | Por continuidad | **No** | |
| 9 | Variantes de las preguntas | **No** | Lo más cercano: el paper trata la personalidad como tarea ordinal con kappa cuadrático (nuestro T2 usa QWK) y combina tres pérdidas (entropía cruzada ponderada por distancia, focal y suavizado); nosotros probamos esa combinación en T2 y T4 y no aportó |
| 10 | Simples con variables de tiempo | **No** | La logística del paper usa un vector de conteos de tokens en un intervalo de un año, sin variables de tiempo resumidas. El paper dice que el transformer permite evitar variables hechas a mano, pero no lo contrasta con una línea de base que las tenga |

**Resumen:** 6 de las 10 pruebas (4, 6, 7, 8, 9 y 10) no están en el paper y son contribuciones nuestras; las pruebas 2, 3 y 5 siguen su diseño, y en la 5 los resultados difieren.

**Lo que hace el paper y no hicimos:**
- Tarea de personalidad y tarea de emigración (SI).
- Búsqueda de hiperparámetros bayesiana para el modelo y las líneas de base (nosotros solo ajustamos el learning rate de T1 y usamos la configuración de la FFNN original).
- Análisis por largo de secuencia, por cantidad de eventos de salud y por grupos cruzados de edad y sexo (Fig. 2B-D); nosotros solo sexo y edad.
- Recall según la distancia al evento (Fig. A.6); lo más cercano es nuestra prueba de T1 a 2021.
- Model card y mención explícita de que hace falta una auditoría de equidad antes de usarlo.

**Para el paper:** el original muestra que life2vec gana; este trabajo muestra bajo qué condiciones esa ventaja se sostiene. Las diferencias en hubs, TCAV y organización del mapa de personas merecen una nota explícita en la discusión, porque no coinciden con lo que reporta el original.

## 4. Limitaciones

- Un solo pre-entrenamiento por configuración en las tareas (semilla 0); las 3 semillas son sólo de ajuste fino. El modelo sin último valor también es de una semilla.
- Sin pre-entrenar, GRU y logística tienen una semilla.
- Un solo corte temporal (2015-12); sólo T1 se probó contra 2021.
- Las variables de tiempo del control 3.6 son una elección entre muchas (una semilla en la logística, FFNN sin ajuste propio de hiperparámetros): una línea de base mejor afinada podría achicar más la ventaja en T4.
- No hay lectura de equidad específica (sí hay subgrupos por sexo y edad).
- No son comparables los números con los del paper original (otros datos, otras tareas); sólo el orden entre modelos.
- Datos: muestra del 3 % de empleo asalariado registrado privado, con ruido de ±3 % en salarios por anonimización. No vale para empleo no registrado.
- **Uso:** no usar para decisiones sobre individuos ni para lecturas causales.

## 5. Qué conviene hacer después

1. Medir qué parte de la ventaja que sobrevive en T4 es secuencia propiamente dicha (por ejemplo, una línea de base con más variables de tiempo o un GRU con tiempo destruido).
2. Más semillas de pre-entrenamiento y de las líneas de base secuenciales.
3. Validación con cortes posteriores (2017, 2019).
4. Comparar el espacio de conceptos con el del paper original.
5. Lectura de equidad por sexo, edad y provincia.
6. Fuera de alcance por ahora: variante B de eventos compactos, `r32` contra `r34`, cortes múltiples.

## 6. Cómo reproducir

- Entorno: `uv`, Python 3.13 local (`.venv/bin/python`), torch cu126 en Mendieta. Tests: `.venv/bin/python -m pytest tests -q` (123).
- Datos: `python -m labor2vec.datos`; vocabulario y corpus: `python -m labor2vec.tokens.vocabulario` y `.corpus`; tareas: `python -m labor2vec.tareas.t1` y `.derivadas`.
- Cluster: `scripts/cluster/*.sbatch` y `sincronizar_mendieta.sh` (`subir`, `datos`, `bajar`). Todos los jobs llevan aviso por mail.
- Evaluación: `python -m labor2vec.tareas.evaluar --tarea tN --bootstrap 1000` y `python -m labor2vec.tareas.ablaciones --bootstrap 1000`.
- Corridas en `runs/` (sin checkpoints, gitignored); modelos pre-entrenados: `pre_150ep` (el recomendado), `_s1`, `_s2`, `pre_30ep`, `diag_sop` y `pre_150ep_sinuv`. Los checkpoints están en el cluster (~100 MB cada uno).
