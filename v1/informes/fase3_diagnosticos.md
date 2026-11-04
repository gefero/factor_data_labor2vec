# Fase 3 de v1: lógica de los diagnósticos del pre-entrenamiento

Generado el 2026-11-03. Complementa a `v1/informes/fase3.md` (los resultados) explicando **cómo** se obtuvo cada número: qué hace cada prueba, qué pregunta responde, qué supone y qué arrojó. Sirve de guía para repetir el mismo esquema sobre otro modelo (otra semilla, un control) y compararlo con v0 y con la `pre_150ep` de v1.

Al revisar el código para escribirlo aparecieron dos cosas que `fase3.md` no dice y que cambian la lectura de dos resultados (secciones 3.1 y 3.2). Están marcadas como observaciones nuevas, sin verificar con datos.

## Resumen

| # | Prueba | Pregunta que responde | Resultado de `pre_150ep` v1 | Lo que **no** dice |
|---|---|---|---|---|
| P1 | Salud del entrenamiento | ¿Corrió completo, sin errores, en el tiempo previsto? | 56.250 pasos, sin NaN, 18,0 GiB, 5 h 28 min | Nada sobre la calidad del modelo |
| P2 | Curva de validación | ¿Sigue aprendiendo? ¿Se sobreajusta? | Mejora hasta el final, se aplana; sin sobreajuste | Si lo aprendido sirve para las tareas |
| P3 | Perplejidad y exactitud del MLM | ¿Predice los tokens ocultos mejor que el azar informado? | ppl 3,95 contra 59,7 del unigrama; acierto 55 % | No se compara con v0 (otros objetivos) |
| P4 | SOP: pérdida, AUC, recalls | ¿Detecta documentos con el orden alterado? | AUC 0,913 (v0 0,865) | Que v1 entienda mejor el orden: la tarea cambió |
| P5 | Comparación global con v0 | ¿Qué métricas se pueden poner lado a lado? | Sólo el SOP, con reservas | — |
| P6 | MLM por categoría (un modelo) | ¿De dónde viene el error del MLM? | Salario = 44 % de la pérdida; SINREG casi gratis | Comparación limpia con v0 (otros trabajadores) |
| P7 | MLM por categoría pareado con v0 | ¿v1 predice peor las variables de empleo que v0, con los mismos trabajadores? | Sí: +0,18 nats [+0,175, +0,188] | Por qué; ni si v1 rinde peor en T1-T4 |

## 1. Lo mínimo para entender las pruebas

**Qué aprende el modelo.** El pre-entrenamiento tiene dos tareas a la vez:

- **MLM** (*masked language model*, modelo de lenguaje enmascarado): se tapa el 30 % de los tokens de cada documento y el modelo tiene que adivinar cuál era cada uno mirando el resto. Es la tarea principal.
- **SOP** (*sequence order prediction*, predicción del orden): al 10 % de los documentos se les altera el orden de los eventos (5 % invertidos, 5 % mezclados) y el modelo tiene que decir si el documento está normal, invertido o mezclado. Es una tarea auxiliar que obliga a prestar atención al orden temporal.

**Cómo es un documento de v1.** Ejemplo inventado (no es una persona real): una mujer nacida en 1980 con empleo registrado de enero a marzo de 2015, sin registro en abril, empleo en mayo y nada más hasta diciembre de 2015 (el fin de la ventana).

```
[CLS] SEXO_M NAC_1980 [SEP]                       ← fondo (4 tokens)
REM_042 TAM_3 RAMA_521 PROV_02 EMPINI_4 [SEP]     ← ene-2015 (6 tokens)
REM_043 TAM_3 RAMA_521 PROV_02 EMPINI_4 [SEP]     ← feb-2015
REM_041 TAM_3 RAMA_521 PROV_02 EMPINI_4 [SEP]     ← mar-2015
SINREG_INT [SEP]                                  ← abr-2015 (2 tokens: hay un registro después)
REM_045 TAM_2 RAMA_473 PROV_02 EMPINI_2 [SEP]     ← may-2015
SINREG_SAL [SEP]                                  ← jun-2015 (no hay registro posterior en la ventana)
... SINREG_SAL [SEP] hasta dic-2015
```

Cada token lleva además la fecha (día desde el origen), la edad y un segmento. En v0 el mismo documento no tendría las líneas `SINREG`: el hueco de abril y la salida de junio sólo se verían por el salto de fecha entre eventos.

**Qué se enmascara.** Se eligen al azar `floor(0,3 × largo del documento)` posiciones **entre los tokens no especiales** (los `[CLS]` y `[SEP]` nunca se tapan). De las elegidas, 80 % se reemplazan por `[MASK]`, 10 % por un token al azar y 10 % quedan igual (`tokens/documento.py`, `enmascarar`). En validación el azar es **determinista por trabajador** (semilla `[0, k]`), así que el mismo modelo ve siempre el mismo enmascarado.

**Con qué datos se mide.** El split de validación tiene 86.347 trabajadores, de los cuales unos 73.900 tienen al menos un evento hasta 2015-12. Nunca se usaron para entrenar.

## 2. Las pruebas, una por una

### P1. Salud del entrenamiento

- **Qué hace.** Lee los logs de los 7 jobs (`v1/runs/pre_150ep/*.out`) y el CSV de métricas (`logs/version_*/metrics.csv`): pasos completados, memoria pico de la GPU, secuencias y tokens por segundo, presencia de NaN, pérdida de entrenamiento al principio (media de los primeros 50 pasos) y al final (últimos 50).
- **Qué testea.** Que la corrida terminó lo pedido (150 épocas × 375 pasos = 56.250) y que no hubo problemas numéricos ni de recursos. Es un control operativo, no estadístico.
- **Supuestos.** Que el último `modelo_final.pt` corresponde al paso 56.250 (verificado: el tramo 7 restauró `last.ckpt`, vio `max_epochs` cumplido y salió sin entrenar).
- **Resultado.** Completo; sin NaN; 18,0 GiB de 24; 209 seq/s en `ivb12`, 185 en `ivb06`; pérdida de entrenamiento 3,19 → 1,41; 5 h 28 min de reloj.

### P2. Curva de validación y sobreajuste

- **Qué hace.** Cada 3 épocas (51 veces en total) el entrenamiento se detiene y evalúa el modelo en **los mismos 200 lotes de 32 trabajadores de validación** (6.400 documentos): sin augmentación, con dropout apagado y con el enmascarado determinista. Registra pérdida y perplejidad del MLM, exactitud, pérdida, AUC y recalls del SOP.
- **Qué testea.** (a) Si el modelo sigue mejorando o ya se estancó: si al final la curva sigue bajando, más entrenamiento podría ayudar. (b) Si hay **sobreajuste**: el modelo memoriza el entrenamiento y empeora en datos nuevos, lo que se vería como una validación que sube después de un mínimo.
- **Supuestos.**
  - Que 6.400 documentos fijos son representativos de la validación. El orden de los lotes es aleatorio pero fijo, así que siempre son los mismos.
  - Que comparar la validación consigo misma a lo largo del tiempo alcanza para detectar sobreajuste. **No** sirve comparar con la pérdida de entrenamiento: la de entrenamiento es más alta (MLM 1,51 contra 1,37 en validación) porque se mide con augmentación (eventos borrados, fechas con ruido) y dropout, es decir, sobre documentos más difíciles.
- **Resultado.** Todas las métricas mejoran hasta el final y se aplanan en los últimos ~5.000 pasos (la tasa de aprendizaje baja a cero por el calendario). Ninguna empeora después de un mínimo: no hay sobreajuste.

### P3. Perplejidad y exactitud del MLM contra la línea de base unigrama

- **Qué hace.** Para cada token oculto calcula la **NLL** (*negative log-likelihood*, error en nats): menos el logaritmo de la probabilidad que el modelo le dio al token correcto. Si le dio probabilidad 1, el error es 0; si le dio 0,5, es 0,69; si le dio 0,01, es 4,6. Promedia sobre todos los tokens ocultos y reporta:
  - **Perplejidad** = exp(NLL media). Se lee como «entre cuántos tokens equiprobables está dudando el modelo, en promedio». 3,95 ≈ duda entre unos 4.
  - **Exactitud**: fracción de tokens ocultos en que el token con mayor probabilidad es el correcto.
  - **Perplejidad unigrama**: la de un «modelo» que ignora el contexto y predice siempre según la frecuencia de cada token en 20.000 documentos de entrenamiento (con suavizado de Laplace; `tokens/linea_base.py`). Se calcula sobre los mismos tokens ocultos.
- **Qué testea.** Que el modelo usa el contexto: si su perplejidad no fuera mucho menor que la del unigrama, no estaría aprendiendo nada de la secuencia.
- **Supuestos.**
  - Que el unigrama es una referencia razonable de «no saber nada». Es una línea de base muy débil: superarla es necesario, no suficiente.
  - La perplejidad depende de **qué tokens** se ocultan. Un token fácil (como `SINREG_SAL`, que se deduce casi siempre de los vecinos) baja la media sin que el modelo sepa más de empleo. Por eso **no se puede comparar** la perplejidad de v1 con la de v0: el 23 % de los objetivos de v1 son `SINREG`.
- **Resultado.** ppl 3,95 contra 59,7 del unigrama (v0: 4,73 contra 81,4); exactitud 0,553 (v0: 0,478). El modelo usa el contexto. La comparación con v0 no es válida; la que sí lo es está en P7.

### P4. SOP: pérdida, AUC y recalls

- **Qué hace.** En validación, el 10 % de los documentos tiene el orden alterado (con el mismo azar determinista). Para v1 la alteración intercambia sólo bloques del mismo largo: meses con empleo entre sí y meses sin registro entre sí; fechas, edades y segmentos quedan en su lugar (así los `[SEP]` no delatan la alteración). El modelo da tres probabilidades (normal, invertido, mezclado) y se mide:
  - **Pérdida SOP**: entropía cruzada con pesos de clase 1,11 / 10 / 10 (para que las clases raras importen) y suavizado de etiquetas 0,1. Por los pesos y el suavizado no es una probabilidad interpretable; sirve para comparar corridas con los mismos pesos.
  - **AUC** de «alterado contra normal» usando 1 − P(normal) como puntaje. Es la probabilidad de que, tomando un documento alterado y uno normal al azar, el modelo le dé más puntaje al alterado. 0,5 = azar, 1 = perfecto. No depende de ningún umbral.
  - **Recall por clase**: de los documentos de cada clase, qué fracción clasificó bien (eligiendo la clase de mayor probabilidad).
- **Qué testea.** Si el modelo aprendió algo del orden temporal de los eventos: por ejemplo, que el salario relativo y el tamaño de la empresa no saltan sin sentido de un mes a otro.
- **Supuestos.**
  - Que la tarea no tiene atajos (pistas que delatan la alteración sin entender el orden). Ya se arregló uno en la fase 2 (los `[SEP]` desalineados). **Ver 3.2: queda otro posible, propio de v1.**
  - Hay pocos casos: de 6.400 documentos, unos 320 invertidos y 320 mezclados. Los recalls de esas clases tienen un error de muestreo de unos ±0,05.
- **Resultado.** AUC 0,913 (v0 0,865); pérdida 1,091 (v0 1,196); recalls normal / invertido / mezclado 0,95 / 0,57 / 0,67 (v0 0,91 / 0,39 / 0,63). La AUC sube de 0,62 a 0,91 a lo largo del entrenamiento.

### P5. Comparación global con v0

- **Qué hace.** Pone lado a lado las métricas finales de validación de v0 y v1 (misma semilla, mismo calendario, mismos hiperparámetros).
- **Qué testea.** Si el cambio de codificación (la grilla con `SINREG`) cambió lo que el modelo aprende.
- **Supuestos.** Que la métrica mide lo mismo en las dos versiones. Ahí está el problema:
  - **Perplejidad y exactitud del MLM:** no comparables (P3).
  - **SOP:** mismas clases y pesos, pero la tarea cambió (permutación por clase de largo, y los bloques `SINREG` se mueven también). Comparable en la forma, no necesariamente en la dificultad.
  - Además, los 6.400 documentos **no son los mismos trabajadores** en v0 y v1: el armado de lotes ordena por largo dentro de grupos de 1.600 y los largos cambian con la grilla, así que los lotes que caen entre los 200 primeros son otros.
- **Resultado.** Sólo el SOP se pone lado a lado, con la reserva de que v1 resuelve «una tarea parecida», no la misma. Esto motivó P7.

### P6. Pérdida del MLM por categoría (un solo modelo)

- **Qué hace.** `v1/scripts/mlm_por_categoria.py runs/pre_150ep --lotes 80` recorre los primeros 80 lotes de validación (2.560 documentos, un subconjunto de los de P2) y separa la NLL y el acierto según la categoría del token oculto: salario, rama, tamaño, provincia, inicio de la empresa, sexo, nacimiento, `SINREG_INT`, `SINREG_SAL`. Guarda `v1/runs/pre_150ep/mlm_por_categoria.json`.
- **Qué testea.** De dónde viene el error del MLM: qué variables son difíciles, cuáles son casi gratis y cuánto de la pérdida total aporta cada una (cuántos objetivos × su NLL media).
- **Supuestos.** Que 80 lotes alcanzan para estimar la media de cada categoría frecuente (hay ~58.000 objetivos de cada variable de empleo; sí alcanza). Para sexo y nacimiento hay ~1.100 objetivos (un token por documento), así que esas medias son ruidosas.
- **Resultado.**

| Categoría | % de los objetivos | NLL media | Acierto | % de la pérdida |
|---|---|---|---|---|
| salario | 15,2 | 3,957 | 0,108 | 43,7 |
| rama | 15,3 | 1,276 | 0,539 | 14,2 |
| tamaño | 15,3 | 1,213 | 0,482 | 13,4 |
| provincia | 15,3 | 1,155 | 0,511 | 12,8 |
| inicio de la empresa | 15,3 | 1,148 | 0,524 | 12,7 |
| `SINREG_SAL` | 16,2 | 0,064 | 0,980 | 0,7 |
| `SINREG_INT` | 6,8 | 0,375 | 0,890 | 1,8 |
| sexo, nacimiento | 0,6 | 0,73 / 1,70 | 0,73 / 0,39 | 0,5 |

El salario (101 percentiles posibles) es casi la mitad de la dificultad del MLM. Los `SINREG` son casi un cuarto de los objetivos y menos del 3 % de la pérdida: confirman que la perplejidad de v1 está «inflada para bien» por tokens fáciles.

Con la misma prueba sobre v0 (copia del script en el scratchpad de la sesión anterior) aparecía que v1 predecía peor cada variable de empleo. Pero eran lotes con trabajadores distintos (ver P5), así que podía ser composición de la muestra. De ahí P7.

### P7. MLM por categoría pareado con v0, con intervalo por remuestreo

- **Qué hace.**
  1. Toma **los mismos 3.696 trabajadores** de validación en los dos modelos: uno de cada 20 (índices 0, 20, 40…).
  2. Arma el documento de cada uno con la codificación de su versión (v0 sin grilla, v1 con grilla), lo enmascara con el azar determinista del trabajador y pasa los dos modelos.
  3. Para cada trabajador y categoría guarda cuántos tokens ocultos hubo y la suma de sus NLL (`pareado.py`, salidas `par_v0.npz` y `par_v1.npz`).
  4. NLL media de cada categoría = suma de NLL / cantidad de objetivos, en cada versión; diferencia v1 − v0.
  5. **Intervalo de confianza por remuestreo de trabajadores** (*bootstrap*): se sortean 3.696 trabajadores con reposición, se recalcula la diferencia y se repite miles de veces; el intervalo del 95 % son los percentiles 2,5 y 97,5 de esas diferencias. Se remuestrea al trabajador entero (con todos sus tokens) porque los tokens de una misma persona no son independientes entre sí.
- **Qué testea.** Si, a igualdad de personas, el modelo de v1 adivina peor las variables de empleo que el de v0. Es la única comparación de calidad del MLM que se sostiene entre versiones, porque deja afuera los `SINREG` y fija la muestra.
- **Supuestos.**
  - **El intervalo mide sólo la incertidumbre por la muestra de trabajadores evaluados**, con los dos modelos fijos. No mide cuánto cambiaría el resultado si se volviera a entrenar con otra semilla. Hay un solo modelo por versión, así que una parte de la diferencia podría ser azar del entrenamiento. Ver 3.3: hay forma de acotarlo sin entrenar nada.
  - Que «la misma variable» es la misma tarea en las dos versiones. Es la misma variable y el mismo trabajador, pero **no los mismos tokens ocultos ni el mismo contexto visible**: el documento de v1 tiene otros tokens alrededor y, sobre todo, se tapa una fracción mayor de los tokens de empleo. **Ver 3.1: esto puede explicar parte de la diferencia.**
  - Que los 3.696 trabajadores representan la validación. Es una muestra sistemática (uno de cada 20) de un split armado al azar; razonable.
  - La media pondera a cada trabajador por sus tokens ocultos: las historias largas pesan más.
- **Resultado.** (Reproducido hoy desde `par_v0.npz` y `par_v1.npz`, con 2.000 remuestreos: coincide con `fase3.md`.)

| Variable | NLL v0 | NLL v1 | v1 − v0 [IC 95 %] |
|---|---|---|---|
| salario | 3,744 | 3,941 | +0,197 [+0,183, +0,211] |
| rama | 1,065 | 1,265 | +0,200 [+0,190, +0,212] |
| tamaño | 1,032 | 1,199 | +0,167 [+0,160, +0,174] |
| provincia | 0,993 | 1,147 | +0,154 [+0,146, +0,161] |
| inicio de la empresa | 0,955 | 1,141 | +0,187 [+0,180, +0,193] |
| año de nacimiento | 1,384 | 1,840 | +0,456 [+0,403, +0,505] |
| sexo | 0,675 | 0,707 | +0,032 [+0,001, +0,062] |
| **todas sin `SINREG`** | **1,556** | **1,737** | **+0,181 [+0,175, +0,188]** |

Dato nuevo, calculado hoy con los mismos archivos: **el 84,7 % de los 3.696 trabajadores** tiene NLL media de empleo más alta con v1. No es un efecto de unos pocos casos extremos.

## 3. Observaciones nuevas al revisar el código

> **Actualización (misma fecha, más tarde):** 3.1 y 3.2 se midieron en `fase3_evaluacion.md`. El atajo del SOP (3.2) existe: aparece en el 35 % de los documentos alterados. El enmascarado más denso (3.1) explica, como mucho, una parte: en 463 trabajadores con documento y máscara idénticos en las dos versiones, v1 igual predice peor el empleo (+0,107 nats).

### 3.1 v1 tapa una fracción mayor de los tokens de empleo (posible explicación parcial de P7)

En `fase3.md` se ofrecieron dos hipótesis para el +0,18 (capacidad del modelo y ventana local). Hay una tercera, más mecánica, que no se mencionó: **el MLM de v1 es más difícil por construcción**.

- La cantidad de tokens que se tapan es el 30 % del **largo total** del documento, `[SEP]` incluidos, pero sólo se eligen entre los no especiales.
- En v0 cada evento tiene 6 tokens y 5 se pueden tapar: cada token de empleo queda oculto con probabilidad ≈ 0,30 × 6/5 = **0,36**.
- En v1 cada mes sin registro aporta 2 tokens (`SINREG_x [SEP]`) pero sólo 1 se puede tapar. Los `[SEP]` de esos meses suman largo (y por lo tanto cupo de máscara) sin sumar candidatos, así que la probabilidad de tapar cada token sube.
- Medido en la muestra pareada: v1 tiene **7,5 % más objetivos de empleo** que v0 con las mismas personas y los mismos eventos (499.920 contra 465.204; salario 99.557 contra 92.626). O sea, cada token de empleo se tapa con probabilidad ≈ **0,39** en lugar de 0,36.

Consecuencia: al predecir un token de empleo, v1 ve menos vecinos de la misma relación sin tapar (≈ 61 % contra 64 %). Eso pasa en la evaluación de P7 y también pasó durante todo el entrenamiento. No sabemos cuánto de los +0,18 nats explica; podría ser poco o bastante.

**Cómo probarlo sin entrenar:** repetir P7 con el modelo de v1 pero fijando la cantidad de tokens tapados para que la probabilidad por token de empleo sea 0,36, como en v0 (`enmascarar` ya acepta `n_mask`). Es sólo inferencia sobre 3.696 documentos. Si la diferencia baja mucho, buena parte del «v1 predice peor» es «a v1 se le toma un examen más difícil». Para el entrenamiento, el control equivalente sería calcular `mask_ratio` sobre los candidatos y no sobre el largo total, pero eso sí requiere re-entrenar.

### 3.2 El SOP de v1 puede tener otro atajo, a través de los tipos de `SINREG`

En un documento normal de v1 los `SINREG_SAL` están siempre al final y todos seguidos (después del último empleo), y los `SINREG_INT` siempre antes de algún empleo. Cuando el SOP invierte o mezcla los bloques de 2 tokens entre sí, en una persona que tiene meses de los dos tipos un `SINREG_SAL` puede quedar antes de un empleo o un `SINREG_INT` después del último. Eso **no ocurre nunca** en un documento normal: el modelo puede detectar la alteración mirando sólo esa regla, sin entender nada del orden de las trayectorias.

Si esto pesa, la AUC 0,913 (contra 0,865 de v0) no indicaría que v1 entiende mejor el orden. En la población de entrenamiento el 62,5 % termina en salida y el 58,7 % tiene algún hueco intermedio, así que la combinación debe ser frecuente.

**Cómo probarlo sin entrenar:** separar la AUC del SOP de validación entre documentos con y sin meses de los dos tipos (o con y sin alguna «violación» de la regla después de la alteración). Si la AUC de los que no pueden tener esa pista queda cerca de la de v0, el aumento era el atajo. No afecta al MLM ni a P7.

### 3.3 La variación por semilla de entrenamiento no está medida, pero se puede acotar

El intervalo de P7 no incluye el azar del entrenamiento (P7, supuestos). En v0 existen dos modelos más, iguales salvo la semilla (`pre_150ep_s1` y `pre_150ep_s2`, copias locales en `v0/runs/*_vista/`). Corriendo P7 entre v0 semilla 0 y v0 semilla 1 y 2 se obtiene **cuánto difieren dos modelos que sólo cambian la semilla**. Si esa diferencia es de 0,01-0,02 nats, el +0,18 de v1 está muy por encima; si fuera de 0,1, habría que entrenar más semillas de v1 antes de concluir.

### 3.4 Reproducibilidad: parte del análisis no está en el repo

`pareado.py`, la copia de `mlm_por_categoria.py` para v0, el resumen de métricas (`resumen_humo.py`) y el cálculo del intervalo (que se hizo en línea y no quedó guardado) viven en el scratchpad de la sesión anterior, en `/tmp`, y pueden desaparecer. Para repetir el esquema conviene pasarlos a `v1/scripts/` como un solo script con argumentos (modelo A, modelo B, cada cuántos trabajadores, número de remuestreos, opción de igualar la tasa de enmascarado).

## 4. Cómo repetir el esquema sobre otro modelo

Para un modelo nuevo `M` (otra semilla, un control) comparado con v1 `pre_150ep` y con v0 `pre_150ep`:

1. **P1-P2:** resumir logs y métricas de validación de `M` (pérdida inicial/final, NaN, memoria, velocidad; curva de las 51 validaciones).
2. **P3-P4:** métricas finales de validación. La perplejidad sólo se compara entre modelos con la **misma codificación**; el SOP sólo entre modelos con la misma regla de alteración.
3. **P6:** `mlm_por_categoria.py <corrida de M> --lotes 80`.
4. **P7:** pareado de `M` contra v0 y contra v1 sobre los mismos 3.696 trabajadores, con intervalo por remuestreo; agregar la fracción de trabajadores en que cada uno es peor.
5. **Nuevos (3.1-3.3):** P7 con tasa de enmascarado igualada; AUC del SOP separada por documentos con posible atajo; P7 entre las semillas de v0 como referencia de ruido.

Las decisiones abiertas son cuál es el modelo `M` (o si se repite sólo sobre la `pre_150ep` de v1 agregando 3.1-3.3) y si los scripts se pasan al repo antes (3.4). Ninguno de los pasos de arriba necesita GPU ni cluster: son inferencia sobre unos pocos miles de documentos.
