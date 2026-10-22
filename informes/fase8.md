# Fase 8: controles temporales, variantes de T2-T4 y ablación sin atributos de último valor

Plan: [plans/PLAN_fase_8.md](../plans/PLAN_fase_8.md). Tablas completas en [fase8/](fase8/) (`comparaciones.md` y `.json`, y una carpeta por evaluación: `t2c`, `t3d`, `t4c`, `t1_uv` … `t4_uv`). Las comparaciones salen de `python -m labor2vec.tareas.ablaciones --bootstrap 1000` (17 min en local); la sección 5, de `python -m labor2vec.tareas.ablaciones --solo tiempo` y `python -m labor2vec.tareas.tabulares --tarea tN --tiempo completo|sin_ultimo_valor`.

## Qué preguntas responde

Las fases 6 y 7 dejaron dos dudas sobre por qué el transformer gana en T3 y T4:

1. **¿Usa el orden y el momento de los eventos?** (hipótesis: una bolsa de tokens los pierde).
2. **¿Hay fuga de información?** Rama, provincia y antigüedad son atributos de «último valor»: se toman del último registro de cada relación laboral y pueden traer datos posteriores al corte (2015-12).

Además se probaron tres variantes de T2, T3 y T4 para ver si el patrón de resultados depende de cómo se formula la pregunta.

Todas las diferencias son pareadas (mismas personas del test, media de 3 semillas) con IC al 95 % por bootstrap de 1.000 remuestreos. `*` = el IC no incluye el 0.

## 1. Control temporal

Se alteró la entrada del transformer pre-entrenado:
- **tiempo**: se permutan las fechas y edades entre los eventos de la persona (los tokens quedan en su sitio).
- **orden**: se permuta el orden de los eventos llevándose su fecha.

Hay que saber que el modelo no tiene posición por índice: el orden sólo le llega por la fecha (`abspos`) y la edad.

**Sin reajustar** (el modelo ajustado se evalúa con la entrada alterada; mide cuánto depende del tiempo, pero lo exagera, porque nunca vio entradas así). Pérdida de la métrica principal:

| | tiempo | orden |
|---|---|---|
| T1 (AUL) | −0,031 | −0,041 |
| T2 (QWK) | −0,065 | −0,026 |
| T3 (AUL) | −0,017 | −0,013 |
| T4 (AUC macro) | −0,071 | −0,034 |

**Reajustando** con el tiempo destruido (el modelo aprende a vivir sin él; sólo T3 y T4):

| | original | tiempo destruido | diferencia |
|---|---|---|---|
| T3 (AUL) | 0,725 | 0,710 | −0,015 [0,011; 0,019] * |
| T4 (AUC macro) | 0,772 | 0,738 | −0,035 [0,030; 0,040] * |

Lectura:
- **En T3 el tiempo importa poco.** Sin tiempo el transformer (0,710) sigue por encima del GRU (0,702) y de la FFNN (0,687). La hipótesis de que gana por usar el orden y el momento no se sostiene para T3.
- **En T4 importa bastante.** Sin tiempo (0,738) queda por debajo del GRU (0,746), aunque sobre la logística (0,691). Acá la hipótesis sí tiene respaldo.
- En T1 y T2 sólo se midió sin reajustar, así que no se puede cuantificar el aporte real.

## 2. Variantes de T2, T3 y T4

Tres preguntas distintas sobre la misma cohorte (transformer pre-entrenado, 3 semillas, test):

| Variante | Qué predice | Transformer | GRU | Mejor simple |
|---|---|---|---|---|
| T2c | baja / igual / sube de quintil salarial (AUC macro) | **0,741** | 0,707 | 0,664 (logística) |
| T3d | destino de la rama entre quienes cambian, 6 clases (AUC macro) | 0,659 | 0,652 | 0,661 (logística con estado actual) |
| T4c | cambia el tamaño de empresa, sí/no (AUL) | **0,702** | 0,682 | 0,658 (logística) |

- **Patrón:** el transformer gana en preguntas de «se movió algo» (T2c, T3, T4, T4c); empata en niveles (T1, T2) y en «a cuál rama» (T3d, sólo 1.683 personas en test).
- **El transformer sin pre-entrenar colapsó en T2c** a la clase mayoritaria (QWK 0,000). La ventaja de pre-entrenar en T2c (0,741 contra 0,562) está inflada por ese colapso; no hay que citarla como efecto limpio.
- En T4c pre-entrenar aporta +0,045 de AUL (0,702 contra 0,657).

## 3. Sin atributos de último valor

Se pre-entrenó de nuevo un modelo sin rama, provincia ni antigüedad (`pre_150ep_sinuv`: perplejidad de validación 6,54, AUC de SOP 0,847; no es comparable con la perplejidad 4,73 del modelo completo, porque la tarea de enmascarado es otra) y se reajustaron los 20 modelos de T1 a T4 (3 semillas del transformer pre-entrenado, 1 del sin pre-entrenar y 1 del GRU, más las líneas de base tabulares sin esas variables). Todos los jobs terminaron bien y con predicciones completas.

Pérdida de la métrica principal al sacar esos atributos (con todos − sin último valor; positivo = caen):

| | T1 (AUL) | T2 (QWK) | T3 (AUL) | T4 (AUC macro) |
|---|---|---|---|---|
| Transformer pre-entrenado | 0,013 * | 0,007 * | 0,064 * | 0,023 * |
| GRU | 0,030 * | 0,006 * | 0,105 * | 0,047 * |
| FFNN | 0,025 * | 0,012 * | 0,064 * | 0,018 * |
| Regresión logística | 0,032 * | 0,012 * | 0,061 * | 0,050 * |
| Transformer sin pre-entrenar | 0,024 * | −0,073 * (ver abajo) | 0,049 * | 0,013 * |

Resultados sin esos atributos (test; transformer pre-entrenado, mejor método simple):
- **T1:** AUL 0,602 contra 0,585 (GRU) y 0,582 (logística). Sin ellos el transformer pasa a quedar levemente arriba (antes empataba).
- **T2:** QWK 0,822; empata con el GRU (0,822), el sin pre-entrenar (0,818), la logística con estado actual y la persistencia (0,822 las dos).
- **T3:** AUL 0,661 contra 0,656 de la logística con estado actual y 0,624 de la FFNN y la logística. El transformer cae 0,064 y sigue arriba de la FFNN, la logística y el GRU (0,597); la logística con estado actual queda cerca (0,656, IC que se superponen).
- **T4:** AUC macro 0,749 contra 0,699 (GRU) y 0,684 (logística con estado actual). La ventaja se mantiene e incluso crece.

Lecturas:
- **Ningún modelo se hunde, y el transformer no es el que más cae.** Cae menos que todos en T1; en T2 cae casi igual que el GRU; en T3 empata con FFNN y logística y cae mucho menos que el GRU; en T4 cae más que la FFNN (0,023 contra 0,018) pero menos que el GRU y la logística (0,047 y 0,050). Si T3 y T4 estuvieran ganando por la fuga, el transformer debería haber perdido más que los demás; no ocurrió.
- **El pre-entrenamiento sigue aportando sin esos atributos** y más en T3 (0,661 contra 0,589 sin pre-entrenar) y T4 (0,749 contra 0,673).
- **T3 es la tarea más sensible a esos atributos** (−0,06 a −0,10 en todos los modelos). Pero esta caída es una **cota superior** del efecto de la fuga: la rama que tenía la persona en 2015 es información legítima para predecir si cambiará de rama.

### La anomalía de T2 sin pre-entrenar (+0,073 al sacar atributos)

Parece raro que el transformer sin pre-entrenar *mejore* de 0,745 a 0,818. Se revisaron las curvas de validación de ambas corridas (`runs/t2_scratch_s0` y `runs/t2_uv_scratch_s0`):
- Sin los atributos, el modelo da un salto entre las épocas 3 y 6 (QWK de validación 0,33 a 0,81) y después se estabiliza.
- Con todos los atributos sube lento y sigue creciendo hasta el tope de 35 épocas (QWK de validación 0,74), sin llegar a converger.

Interpretación: es un problema de optimización de una sola semilla (el informe de la fase 6b ya advertía que esa corrida llegó al tope de épocas), no un efecto real de los atributos. **Consecuencia importante:** la ventaja de pre-entrenar en T2 (0,830 contra 0,745) estaba sobrestimada. Con una corrida que convergió, el modelo sin pre-entrenar iguala al pre-entrenado (0,818 contra 0,822). En T2 no hay evidencia de que pre-entrenar ayude.

## 4. Por continuidad de la relación principal

El 96,2 % de la cohorte tiene la relación principal vigente después de 2015-12; ahí la fuga «de último valor» es posible. Quienes terminaron antes no pueden filtrar datos posteriores, pero son pocos: 484 en T3/T4 y 481 en T2 en test, 726 en T1 (no son lo mismo que salir del empleo en T1: de ese grupo sale el doble que del resto, 31-33 % contra 15-16 %).

| Tarea y modelo | continúa (diferencia) | terminó antes (diferencia) |
|---|---|---|
| T1 transformer | 0,011 * | 0,041 * |
| T3 transformer | 0,063 * | 0,067 * |
| T3 FFNN | 0,066 * | 0,015 (IC incluye 0) |
| T4 transformer | 0,025 * | 0,009 (IC incluye 0) |
| T4 FFNN | 0,020 * | −0,019 (borde) |

(Detalle de T1 a T4 con la FFNN en `fase8/comparaciones.md`.)

Qué se puede y qué no se puede concluir:
- En T3 el transformer cae igual en quienes no pueden tener fuga (0,067), así que **la caída de T3 no parece venir de la fuga**: es la rama de 2015 usada como información legítima, o al menos no se distingue.
- Con 480-730 personas los IC son anchos y no hay nada concluyente en T4 ni en T1. No tomar estas filas como evidencia sólida.
- No se puede separar limpiamente fuga de información legítima. Esta ablación y la continuidad acotan el problema, no lo resuelven.

## 5. Líneas de base con variables de tiempo (fase 8b)

**Para qué:** en T3 y T4 el transformer gana sobre métodos que reciben solo proporciones de tokens, es decir, sin saber *cuándo* pasó cada cosa. Había que ver cuánto de esa ventaja es solo «tiempo resumido» y cuánto es algo que exige la secuencia completa. Se le dieron esas variables a los métodos simples.

**Qué se agregó** (`src/labor2vec/tareas/tiempo.py`, solo eventos hasta 2015-12; la relación principal se define igual que en `derivadas.py`): antigüedad en el empleo principal; meses desde el último cambio de empresa, de tamaño y de rama; cantidad de cambios en los últimos 12, 24 y 60 meses; meses sin registro en los últimos 24 y 60; y variación del percentil salarial a 12 y 36 meses. La versión «sin último valor» quita la antigüedad y los cambios de rama para ser comparable con los modelos sin esos atributos. Tres modelos: logística y FFNN sobre conteos de tokens más tiempo, y logística con el estado de 2015 más tiempo. Corridas en `runs/<tarea>_tabt/` y `runs/<tarea>_uv_tabt/`; tablas completas en [fase8/comparaciones_tiempo.md](fase8/comparaciones_tiempo.md).

Dos detalles de la definición: la rama es de «último valor» y no cambia dentro de una relación, así que un cambio de rama es que la relación nueva tenga otra rama que la anterior (la primera versión daba siempre 0 y se corrigió antes de correr nada). Y en la cohorte `meses_desde_ultimo_registro` es siempre 0, porque todos tienen registro en 2015: no aporta.

Diferencia = transformer menos la línea de base, en la métrica principal (positiva: gana el transformer; `*` = el IC no incluye el 0):

| | transformer | logística + tiempo | FFNN + tiempo | logística con estado + tiempo |
|---|---|---|---|---|
| T1 (AUL) | 0,619 | 0,634: **−0,016 \*** | 0,619: −0,000 | |
| T1 sin último valor | 0,606 | 0,618: **−0,011 \*** | 0,620: **−0,013 \*** | |
| T2 (QWK) | 0,830 | +0,053 \* | +0,070 \* | 0,821: +0,008 \* |
| T3 (AUL) | 0,725 | 0,731: −0,006 (empate) | 0,709: +0,016 \* | 0,719: +0,005 (empate) |
| T3 sin último valor | 0,661 | 0,692: **−0,031 \*** | 0,686: **−0,025 \*** | 0,708: **−0,047 \*** (ver nota) |
| T4 (AUC macro) | 0,772 | 0,722: +0,051 \* | 0,699: +0,074 \* | 0,741: +0,031 \* |
| T4 sin último valor | 0,749 | 0,697: +0,052 \* | 0,706: +0,043 \* | 0,738: +0,011 \* |

Variantes (con todos los atributos): T2c +0,065 a +0,112 \* (gana el transformer); T3d empate con la logística con estado (−0,008, IC incluye 0) y gana a los otros; T4c +0,014 a +0,027 \*.

**Qué cambia:**
- **T1: una logística con variables de tiempo supera al transformer** (0,634 contra 0,619; sin último valor 0,618 contra 0,606). Antes se decía que empataban. Los métodos simples, con tiempo resumido, ya no solo empatan: ganan.
- **T3: la ventaja del transformer desaparece.** Con todos los atributos empata con la logística con tiempo (−0,006) y con la logística con estado y tiempo (+0,005); solo le gana a la FFNN con tiempo (+0,016). Sin los atributos de último valor, los métodos simples con tiempo lo superan por 0,03 a 0,05. Nota: la «logística con estado + tiempo» sigue usando la rama de 2015 (es el estado de la tarea) y el transformer sin último valor no, así que ese −0,047 no es una comparación pareja; la de «logística + tiempo» (−0,031) sí lo es, porque tampoco usa rama.
- **T4: la ventaja sobrevive pero se achica.** Contra la mejor línea de base con tiempo (logística con estado + tiempo) pasa de +0,089 a +0,031 con todos los atributos y a +0,011 sin los de último valor. Sigue habiendo ventaja en todas las filas, así que parte de lo que aporta el transformer en T4 no queda cubierto por estas variables.
- **T2 y las variantes T2c y T4c no cambian el cuadro**: ahí el transformer ya ganaba o el estado actual explica todo.
- **El rival es fuerte por construcción:** las variables de tiempo las eligió quien conocía las tareas y se entrenaron con los mismos hiperparámetros de la FFNN original, sin ajuste propio. Eso favorece al transformer en la comparación más que a las líneas de base; que aun así gane o empate la logística con tiempo en T1 y T3 es una señal clara.

**Lectura:** buena parte de lo que el transformer aprovecha del tiempo en T3 y T1 son magnitudes simples (antigüedad, cuánto hace que cambió algo, cuántos cambios hubo) que un modelo lineal usa igual de bien. Lo que el transformer hace y esas variables no capturan solo se ve de forma consistente en T4 (y en T2c, T4c). No se probó qué parte es secuencia propiamente dicha.

## Resumen de lo que cambia respecto de la fase 6

| Afirmación anterior | Estado ahora |
|---|---|
| El transformer gana en T3 y T4 | Frente a métodos sin tiempo se mantiene, también sin atributos de último valor. **Frente a métodos simples con variables de tiempo (sección 5): en T4 se mantiene pero se achica (+0,01 a +0,03 contra el mejor); en T3 desaparece; en T1 pierde (−0,01 a −0,02)** |
| Gana por usar el orden y el momento | **Sólo en T4** (−0,035 al destruir el tiempo); en T3 poco (−0,015) |
| T3 puede estar inflado por fuga | **No hay indicios**: cae como los demás y también cae en el grupo sin fuga posible |
| Pre-entrenar aporta +0,06 a +0,09 en todas las tareas | Se mantiene en T1, T3, T4; **en T2 no se sostiene** (el control no había convergido) |
| T2 (quintil salarial): empate con la persistencia | Se mantiene |

## Limitaciones

- Una sola corrida de pre-entrenamiento (semilla 0) para todas las tareas; el modelo sin último valor también es de una semilla.
- El transformer sin pre-entrenar, el GRU y la logística tienen una sola semilla; la dispersión entre semillas del ajuste fino del transformer pre-entrenado es chica (IC angostos), pero no se midió en esos otros.
- Un solo corte temporal (2015-12).
- T3d tiene pocas personas en test (1.683).
- Las variables de tiempo de la sección 5 son una elección posible entre muchas, con una sola semilla en la logística y sin ajuste de hiperparámetros de la FFNN; una línea de base mejor afinada podría achicar más la ventaja en T4. No se midió cuánto de lo que queda es secuencia propiamente dicha.

## Corridas

En `runs/` local (sin checkpoints) y en `~/labor2vec/runs/` del cluster: `ctrl_{ninguno,tiempo,orden}/`, `t{3,4}_tiempo_l2v_s{0,1,2}`, `t2c_*`, `t3d_*` y `t4c_*` (variantes), `pre_150ep_sinuv` y `t{1..4}_uv_{l2v_s0,l2v_s1,l2v_s2,scratch_s0,gru_s0,tab}`. Jobs de los reajustes sin último valor: 1181780 a 1181799; cadena del pre-entrenamiento: 1181720 a 1181726.

## Síntesis: qué se puede decir del transformer

Resumen de todas las pruebas de las fases 4 a 8 y de lo que se desprende de ellas.

### Todas las pruebas

| # | Prueba | Qué se hizo | Qué salió |
|---|---|---|---|
| 1 | Pre-entrenamiento | Entrenar sin etiquetas (completar tokens y ordenar fragmentos) | Funciona y es estable: perplejidad 4,73 contra 81,4 de predecir por frecuencias |
| 2 | Espacio de conceptos | Ver cómo ordena los 519 tokens | Estructura real: salarios ordenados, ramas vecinas del mismo sector, provincias vecinas de la misma región; reproducible entre semillas |
| 3 | Predicción T1-T4 | Ajustar a cuatro tareas y comparar con métodos simples | Empata en T1 y T2; gana en T3 y T4 (frente a métodos sin información de tiempo) |
| 4 | Pre-entrenar o no | El mismo transformer sin pre-entrenar | Suma +0,06 a +0,09 en T1, T3 y T4; en T2 no hay evidencia (la corrida de comparación no había convergido) |
| 5 | Espacio de personas | Mapas, TCAV, saliency | El mapa ordena por quiénes son y no por lo que les pasa; usa sobre todo la historia reciente; TCAV no sirve |
| 6 | Tiempo destruido | Permutar fechas y orden y reajustar | T3 casi no cambia (0,725 a 0,710); T4 cae bastante (0,772 a 0,738) |
| 7 | Sin atributos de último valor | Sacar rama, provincia y antigüedad por posible fuga | Todos los modelos caen; el transformer no es el que más cae |
| 8 | Por continuidad | Separar a quienes no pueden tener fuga | Pocos casos (~500): no concluyente |
| 9 | Variantes de las preguntas | T2c, T3d, T4c | Gana en «se movió algo» (T2c, T4c); empata en «a qué rama» (T3d) |
| 10 | Métodos simples con variables de tiempo | Darles antigüedad, cambios recientes, huecos y tendencia salarial | T1: gana el simple; T3: empate; T4: el transformer sigue ganando pero chico (+0,031); T2: sin cambios |

### Qué se puede decir

**Con respaldo firme**
- El pre-entrenamiento funciona, es estable y aprende estructura real de salarios, ramas y provincias.
- Pre-entrenar mejora al transformer respecto de entrenarlo desde cero en T1, T3 y T4.
- El modelo organiza a las personas por sus características y no por el resultado que se predice.

**Con cuidado**
- Predice mejor que métodos simples sin información de tiempo en T3 y T4, y eso no se debe a fuga de información.
- Frente a métodos simples con información de tiempo, solo en T4 queda una ventaja consistente (+0,011 a +0,031 contra el mejor). Su origen exacto no se midió.

**No se puede decir**
- Que sea mejor en general: en T1, una logística con variables de tiempo lo supera.
- Que su ventaja sea por usar la secuencia: en T1 y T3 se explica con variables de tiempo resumidas.
- Que el pre-entrenamiento ayude en T2.

### Cómo plantearlo

La contribución es el recurso y la evidencia: una implementación abierta de life2vec para registros administrativos de un país de ingreso medio, con un resultado mixto documentado con todos sus controles. La representación aprende estructura real; la ventaja predictiva es de tamaño moderado y queda concentrada en una tarea (tamaño de empresa). Ese planteo es más defendible que afirmar que el transformer gana.

## Comparación con el paper original

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
