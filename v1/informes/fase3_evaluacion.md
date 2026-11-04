# Fase 3 de v1: evaluación de los diagnósticos del pre-entrenamiento y propuesta de rediseño

Generado el 2026-11-03. Se lee junto con `v1/informes/fase3_diagnosticos.md`, que explica **qué hace** cada prueba (P1 a P7), qué pregunta responde, qué supone y qué dio. Acá se evalúa **si las pruebas están bien diseñadas**, se agregan tres mediciones nuevas hechas hoy y se propone un esquema corregido para repetir el análisis.

Las mediciones nuevas **no entrenan ni corren ningún modelo**: usan los archivos del pareado ya guardados (`par_v0.npz`, `par_v1.npz`, con la NLL por trabajador y categoría de los dos modelos) y arman documentos de validación sin pasar por la red (unos segundos de CPU). Scripts en el scratchpad de esta sesión (`identicos.py`, `sop_atajo.py`).

## Resumen

> **Resultados del bloque A en la sección 6** (corrido el 2026-11-03 a la noche). Confirman que la diferencia en el MLM es real (+0,11 nats, diez veces el ruido de semilla) y muestran que la mejora del SOP de v1 era entera el atajo. Lo de abajo es la evaluación previa a esas corridas.

| # | Prueba | ¿Bien diseñada? | Problema principal | Qué cambia en la conclusión |
|---|---|---|---|---|
| P1 | Salud del entrenamiento | Sí | — | Nada |
| P2 | Curva de validación | Sí | Mide la pérdida total; no se ve por categoría | Nada |
| P3 | Perplejidad contra unigrama | Sí, pero débil | Línea de base demasiado fácil; ppl mezclada con `SINREG` | Nada |
| P4 | SOP (pérdida, AUC, recalls) | **No, para comparar con v0** | La alteración de v1 deja un **atajo** en el 35 % de los documentos alterados y no cambia nada en otro 4,6 %; sin intervalo | **«v1 resuelve mejor el SOP» no se sostiene**: el atajo solo alcanza para explicar casi toda la mejora |
| P5 | Comparación global con v0 | Sí (su conclusión era correcta) | — | Nada |
| P6 | MLM por categoría (un modelo) | Sí, como descripción | Incluye documentos alterados por el SOP | Nada |
| P7 | MLM pareado con v0 | **A medias** | Empareja trabajadores pero no los tokens ocultos; no mide el azar del entrenamiento | El hallazgo (v1 predice peor el empleo) **se sostiene y se precisa**: aparece aun con documentos idénticos, así que está en el modelo, no en la codificación |

Lo más importante:

1. **En 463 trabajadores el documento y la máscara son idénticos en v0 y v1** (no tienen ningún mes sin registro). Ahí la entrada al modelo es exactamente la misma y aun así v1 predice peor el empleo: **+0,107 nats [+0,100, +0,113]**, peor en el 93 % de esos trabajadores. Esa parte de la diferencia no puede venir de la codificación ni del enmascarado: está en lo que aprendió el modelo.
2. Por eso, de mis hipótesis anteriores, **el enmascarado más denso y la ventana local de atención explican, como mucho, lo que pasa por encima de ese +0,107**, y no la diferencia entera (me corrijo: en `fase3_diagnosticos.md`, 3.1, lo presenté como posible explicación de una parte sin acotar cuánta).
3. Falta la pieza que decide si +0,107 es mucho o poco: **cuánto difieren dos modelos de v0 que sólo cambian la semilla**. Hay dos (`pre_150ep_s1` y `_s2`) y medirlo no requiere entrenar.
4. **El SOP de v1 tiene un atajo medido**: en el 35,1 % de los documentos alterados aparece un `SINREG` en un lugar donde nunca está en un documento normal. Si el modelo acierta esos casos y en el resto rinde como v0, la AUC daría 0,91, prácticamente la observada (0,913).

## 1. Con qué criterio se evaluó cada prueba

Una prueba de diagnóstico está bien diseñada si cumple seis condiciones:

1. **Pregunta clara**: se sabe qué respuesta llevaría a qué decisión.
2. **La métrica mide lo que se pregunta** y no otra cosa que se le parece.
3. **Comparación justa**: si compara dos modelos, lo hace en las mismas condiciones (mismas personas, mismos casos, misma dificultad).
4. **Sin atajos**: el modelo no puede sacar buena nota por una pista ajena a lo que se quiere medir.
5. **Incertidumbre completa**: el intervalo incluye las fuentes de azar que pueden cambiar la conclusión (qué personas se evaluaron, pero también la semilla del entrenamiento).
6. **Reproducible**: el código y los datos intermedios están en el repo.

## 2. Evaluación prueba por prueba

### P1. Salud del entrenamiento — correcta

Es un control operativo (pasos completados, NaN, memoria, velocidad). Cumple su función y no pretende más. Sin cambios.

### P2. Curva de validación — correcta, con una mejora útil

- **Bien:** usa siempre los mismos 6.400 documentos de validación, sin augmentación y con enmascarado fijo; compara la validación consigo misma (no con la pérdida de entrenamiento, que se mide sobre documentos más difíciles). Así se detecta sobreajuste correctamente.
- **Limitación:** registra sólo la pérdida total del MLM. En v1 esa pérdida mezcla los `SINREG` (casi gratis) con el empleo, así que la curva puede bajar porque mejoran los `SINREG` sin que mejore el empleo.
- **Arreglo (para próximos entrenamientos):** registrar además la pérdida del MLM **sólo de empleo** y **sólo de `SINREG`** en cada validación. Es un cambio chico en `pretrain.py`; no se puede aplicar retroactivamente a la `pre_150ep`.

### P3. Perplejidad contra unigrama — correcta pero débil

- **Bien:** comprueba que el modelo usa el contexto (3,95 contra 59,7).
- **Limitación 1:** el unigrama es una referencia muy fácil de superar. Gran parte del MLM sobre atributos se resuelve **copiando el valor del mes vecino de la misma relación** (rama, provincia, tamaño e inicio de la empresa casi no cambian dentro de una relación). Superar al unigrama no dice si el modelo hace algo más que copiar.
- **Limitación 2:** la perplejidad de v1 está bajada por los `SINREG` (ya dicho en `fase3_diagnosticos.md`).
- **Arreglo:** (a) reportar la perplejidad **sólo de empleo**; (b) agregar una línea de base **«copiar al vecino»**: para cada token oculto, predecir el mismo campo del evento visible más cercano. Es una regla sin entrenamiento que mide cuánto del MLM es copia. Sirve además para la comparación entre versiones (ver P7): si la regla también rinde peor en v1, la dificultad es de la tarea y no del modelo.

### P4. SOP — no sirve para comparar v1 con v0

**Qué está bien:** la AUC no depende de umbrales; la pérdida usa los mismos pesos en las dos versiones; el arreglo de la fase 2 (permutar sólo bloques del mismo largo) eliminó el atajo de los `[SEP]` desalineados.

**Qué está mal**, medido hoy sobre 3.696 trabajadores de validación (uno de cada 20), alterando cada documento a propósito y comparándolo con su versión normal:

| | v0 | v1 |
|---|---|---|
| Documentos alterables | 96,0 % | 99,7 % |
| Alterados que **quedan idénticos** al normal | 1,0 % | **4,6 %** |
| Alterados con un `SINREG` **fuera de lugar** (atajo) | — | **35,1 %** |
| Normales con un `SINREG` fuera de lugar | — | 0 % |

(Iguales para «invertido» y «mezclado».)

Explicación sencilla:

- **Atajo.** En un documento normal los `SINREG_SAL` siempre están todos al final, después del último empleo, y los `SINREG_INT` siempre antes de algún empleo. La alteración de v1 intercambia los bloques de 2 tokens entre sí, sin mirar su tipo. Si la persona tiene meses de los dos tipos, un `SAL` termina antes de un empleo o un `INT` después del último. Eso **nunca** pasa en un documento normal: basta mirar esa regla para saber que el documento fue alterado, sin entender nada de la trayectoria.
- **Alteraciones invisibles.** Todos los `SINREG_SAL` son el mismo token, igual que todos los `SINREG_INT`. Intercambiarlos entre sí no cambia nada. Si además la persona tiene un solo mes de empleo, o todos sus meses de empleo son iguales, el documento «alterado» es idéntico al normal pero tiene la etiqueta «alterado». Nadie puede acertar esos casos.
- **Cuentas.** La AUC es, en promedio, la probabilidad de ordenar bien un alterado frente a un normal. Si el modelo ordena bien todos los casos con atajo (35,1 %), tira la moneda en los invisibles (4,6 %) y en el resto rinde como v0 (0,865), la AUC total da 0,351 + 0,046 × 0,5 + 0,603 × 0,865 ≈ **0,90**; con rendimiento perfecto sólo en el atajo y 0,894 en el resto se llega exactamente a 0,913. O sea, **casi toda la mejora de v1 en el SOP se puede explicar por el atajo**. No está demostrado que el modelo lo use; está demostrado que puede.
- **Sin intervalo.** Hay unos 640 documentos alterados en la validación; una diferencia de AUC de 0,05 entre dos corridas puede tener un error de ±0,02 o más, y nunca se calculó.

**Arreglos.**
- Para evaluar la `pre_150ep` actual (sin entrenar): calcular la AUC **por separado** en los alterados con atajo, los invisibles y el resto (cada grupo contra los normales), con intervalo por remuestreo. La comparable con v0 es la del **resto**.
- Para próximos entrenamientos (requiere re-entrenar): **dejar fijos los bloques `SINREG`** y permutar sólo los de empleo. Los `SINREG` no tienen contenido propio que permutar: intercambiarlos sólo produce el atajo o nada. Y descartar (volver a sortear o marcar como normal) las alteraciones que no cambian el documento.

### P5. Comparación global con v0 — correcta

Su conclusión («sólo el SOP es comparable, y con reservas») era correcta; con lo de P4, ni siquiera el SOP es comparable tal como se mide.

### P6. MLM por categoría de un modelo — correcta como descripción

- **Bien:** muestra de dónde sale la pérdida (salario 44 %, `SINREG` < 3 %).
- **Detalles:** (a) los 80 primeros lotes son lotes enteros de largo parecido, así que la muestra está agrupada (los intervalos, si se calcularan, deberían remuestrear lotes o trabajadores, no tokens); (b) incluye el ~10 % de documentos alterados por el SOP, donde predecir un atributo es más difícil. Ninguno cambia las conclusiones; para comparar versiones conviene excluir los alterados (ver P7).

### P7. MLM pareado con v0 — el diseño empareja personas pero no casos

**Lo que hace bien:** usa las mismas 3.696 personas en los dos modelos, deja afuera los `SINREG`, calcula el intervalo remuestreando personas enteras.

**Cuatro problemas de diseño:**

1. **Empareja trabajadores, no tokens.** El enmascarado se sortea sobre cada documento, y los documentos de v1 son más largos: en v0 y v1 se tapan **otros** tokens de la misma persona, y en v1 se tapan **más** (la cantidad depende del largo total, incluidos los `SINREG [SEP]`). Medido hoy, la proporción de tokens de empleo tapados crece con los huecos de la persona:

| Fracción de `SINREG` entre los objetivos de v1 (aprox. del peso de los huecos) | Trabajadores | Objetivos de empleo v1 / v0 | NLL v0 | NLL v1 | v1 − v0 (media por trabajador) | % peor con v1 |
|---|---|---|---|---|---|---|
| 0 % | 537 | 1,00 | 1,508 | 1,616 | +0,105 | 92 % |
| 0-10 % | 813 | 1,03 | 1,550 | 1,695 | +0,151 | 88 % |
| 10-30 % | 764 | 1,12 | 1,598 | 1,800 | +0,236 | 86 % |
| 30-50 % | 485 | 1,25 | 1,589 | 1,869 | +0,281 | 82 % |
| 50-70 % | 450 | 1,38 | 1,643 | 1,998 | +0,412 | 81 % |
| 70-100 % | 645 | 1,50 | 1,709 | 2,337 | +0,746 | 78 % |

   En las personas con muchos huecos, v1 tapa hasta un 50 % más de sus tokens de empleo (la probabilidad pasa de ~0,36 a ~0,54), y ahí la diferencia es mucho mayor. El «+7,5 % de objetivos» que di en `fase3_diagnosticos.md` (3.1) es un promedio que esconde esta variación.

2. **La ponderación cambia entre versiones.** El promedio de cada versión pondera a cada persona por sus tokens tapados; como v1 tapa más a quienes tienen huecos (y son más difíciles de predecir), su promedio les da más peso. Con los pesos de v0 en las dos versiones, la diferencia baja de +0,181 a **+0,169**: poco, pero el diseño debería fijar los pesos. El promedio simple por persona da **+0,315**, porque pesa igual a las muchas personas con pocos eventos y muchos huecos. Son tres preguntas distintas; hay que elegir una de antemano.

3. **No mide el azar del entrenamiento.** El intervalo sólo dice cuánto cambiaría el resultado con otras personas evaluadas, con los dos modelos fijos. Con un solo modelo por versión no se sabe cuánto de la diferencia es la semilla. Después del hallazgo de abajo, esto pasa a ser **el punto decisivo**.

4. **Incluye documentos alterados por el SOP** (10,3 % en v0, 10,7 % en v1), alterados de forma distinta en cada versión.

**El hallazgo nuevo: documentos idénticos.** Se reconstruyeron los documentos de los 3.696 trabajadores con las dos codificaciones. En **463** el documento de v0 y el de v1 son idénticos token por token, con la misma máscara (no tienen meses sin registro; al tener el mismo largo, el sorteo de la máscara y del SOP sale igual). Para ellos el modelo recibe exactamente lo mismo, así que cualquier diferencia es del modelo:

| Variable | NLL v0 | NLL v1 | v1 − v0 [IC 95 %] |
|---|---|---|---|
| salario | 3,628 | 3,743 | +0,115 [+0,103, +0,126] |
| rama | 0,984 | 1,094 | +0,110 [+0,099, +0,122] |
| tamaño | 1,005 | 1,109 | +0,104 [+0,094, +0,113] |
| provincia | 1,001 | 1,090 | +0,089 [+0,079, +0,102] |
| inicio de la empresa | 0,951 | 1,067 | +0,116 [+0,107, +0,125] |
| año de nacimiento (178 casos) | 1,338 | 1,638 | +0,300 [+0,212, +0,393] |
| sexo (173 casos) | 0,726 | 0,692 | −0,034 [−0,071, +0,001] |
| **empleo (90.047 objetivos)** | **1,514** | **1,621** | **+0,107 [+0,100, +0,113]**; peor en el 93 % |

Lectura:

- **Unos 0,11 nats de la diferencia son del modelo**, no de la entrada: el modelo de v1 es peor prediciendo empleo aun en documentos que se ven exactamente como los de v0. Eso descarta, para esta parte, el enmascarado más denso (3.1 del informe anterior) y la ventana local (hipótesis 2 de `fase3.md`): ninguna de las dos actúa en estos documentos.
- **Lo que excede ese +0,11 en las personas con huecos** (hasta +0,75) sí puede deberse a la entrada: más tokens tapados, contexto distinto, o simplemente que son personas más difíciles para un modelo que ya es algo peor. Separar eso requiere el diseño corregido de la sección 4.
- Un matiz: estos 463 son personas sin huecos (12,5 % de la muestra), que en el entrenamiento de v1 eran minoría. Que el modelo de v1 rinda peor justo con ellas es compatible con que se haya especializado en documentos con huecos.

## 3. Qué hipótesis quedan en pie para el +0,11 «del modelo»

Ninguna está probada; las cuatro son compatibles con los datos.

1. **Azar del entrenamiento (semilla).** v0 y v1 usaron la misma semilla, pero con datos distintos la trayectoria del entrenamiento es otra: en la práctica son dos sorteos distintos. Si dos semillas de v0 difieren en ~0,10, no hay nada que explicar. **Es la primera que hay que descartar, y se puede sin entrenar.**
2. **Dilución del gradiente del empleo.** La pérdida del MLM es el **promedio** sobre todos los tokens tapados. En v1, ~23 % son `SINREG`, que el modelo acierta casi siempre y casi no aportan gradiente. El empleo pesa entonces ~0,77 de lo que pesaba en v0 dentro de la pérdida del MLM, mientras que el SOP sigue con su 0,2 fijo (y el recorte del gradiente a norma 1 es sensible a esa proporción). En la práctica, v1 le dio relativamente más peso al SOP. Es coherente con lo observado: SOP mejor (en parte por el atajo, que además es una señal fácil de aprender) y empleo peor.
3. **Capacidad repartida.** El mismo modelo de 8,1 millones de parámetros tiene que representar además la estructura de los huecos, que ocupan más de la mitad de las posiciones.
4. **Especialización en documentos con huecos** (ver el matiz de arriba).

Las 2, 3 y 4 sólo se distinguen re-entrenando (por ejemplo, con `SINREG` como entrada pero no como objetivo, que ya figura entre los controles de `fase3.md` y ataca la 2 y en parte la 3).

## 4. Rediseño propuesto del esquema de diagnóstico

### A. Para analizar la `pre_150ep` actual (sin entrenar; CPU local)

| # | Prueba | Qué responde | Supuestos y cuidados | Costo aprox. |
|---|---|---|---|---|
| D1 | **Referencia de ruido por semilla**: el pareado entre v0 semilla 0 y v0 semillas 1 y 2, con el mismo diseño que D2 | ¿Cuánto difieren dos modelos que sólo cambian la semilla? | Con 3 semillas la banda es gruesa; alcanza para distinguir 0,01-0,03 de 0,10 | 2 inferencias de ~6 min |
| D2 | **Pareado por casos** (reemplaza a P7): se eligen los tokens a tapar **en coordenadas de evento** (persona, mes, campo) y se tapan **los mismos** en los dos documentos; los `SINREG` no se tapan; sin alteración del SOP; tasa de v0 (0,36 de los tokens de empleo) | ¿A igualdad de personas, casos y contexto visible de empleo, v1 predice mejor o peor? (v1 además ve los huecos: es justamente lo que agrega) | Fija de antemano la ponderación (cociente de sumas con los mismos casos en las dos versiones; el promedio por persona como secundario) | 2 inferencias (v0, v1) |
| D3 | **Subconjunto de documentos idénticos** (ya hecho, arriba) | ¿Cuánto de la diferencia es del modelo y no de la entrada? | Son personas sin huecos: no representan a todos | Hecho |
| D4 | D2 **por estratos**: peso de los huecos de la persona, y posición del token (dentro de un tramo de empleo / primer mes después de un hueco / regreso al mismo empleador) | ¿Dónde se concentra la diferencia? Prueba la hipótesis de la ventana local | Estratos definidos antes de mirar los resultados | Sin costo extra (sale de D2) |
| D5 | **Línea de base «copiar al vecino»** sobre los mismos casos de D2 | ¿Cuánto del MLM es copia? ¿La tarea es más difícil en v1 para cualquier predictor? | Regla fija, sin parámetros | Segundos |
| D6 | **SOP desarmado**: AUC con intervalo, por separado para alterados con atajo, invisibles y resto (cada grupo contra los normales), en v1 y en v0 | ¿Mejoró v1 en el SOP sin el atajo? ¿Usa el modelo el atajo? | Sólo la AUC del «resto» se compara con v0 | 2 inferencias (sólo la cabeza del SOP) |

**Regla de decisión, fijada antes de correr:**
- Si la diferencia de D2 está **dentro de la banda de D1**, «v1 predice peor el empleo» no se sostiene: es azar del entrenamiento. Se pasa a la fase 4 sin controles.
- Si está **por encima de la banda**, la diferencia es real. D4 dice si se concentra en las personas con huecos (entonces importa la codificación) o es pareja (entonces importa el entrenamiento: hipótesis 2 a 4). La fase 4 igual decide si eso afecta a las tareas.
- En el SOP, si la AUC del «resto» en v1 queda cerca de 0,865, se reporta que v1 **no** mejoró el SOP y que la cifra 0,913 estaba inflada.

### B. Para próximos pre-entrenamientos (requieren re-entrenar; sólo si D1-D6 lo justifican)

1. **SOP sin atajo:** bloques `SINREG` fijos, se permutan sólo los de empleo; alteraciones que no cambian el documento se vuelven a sortear o se etiquetan como normales.
2. **Validación desagregada:** pérdida del MLM de empleo y de `SINREG` por separado en cada validación (P2), y métricas del MLM sobre documentos sin alterar.
3. **Control de la dilución:** `SINREG` como entrada pero no como objetivo (o ponderar la pérdida para que el empleo pese lo mismo que en v0). Es el control que mejor separa las hipótesis 2-3.
4. **Enmascarado proporcional a los candidatos**, no al largo total, para que la probabilidad de tapar un token de empleo no dependa de los huecos de la persona.
5. Más de una semilla del modelo elegido, si el resultado de D1 muestra que el ruido es comparable a las diferencias que importan.

### C. Reproducibilidad

`pareado.py`, la copia de `mlm_por_categoria.py` para v0 y los scripts de hoy viven en scratchpads de `/tmp`. Propuesta: un solo `v1/scripts/diagnostico_pre.py` con subcomandos (`pareado`, `identicos`, `sop`, `copia`), argumentos para los dos modelos, el muestreo y el número de remuestreos, y salidas en `v1/runs/<corrida>/diagnostico/`.

## 5. Decisiones tomadas

El usuario aprobó el bloque A y el paso de los scripts al repo; el bloque B y la fase 4 quedan en suspenso. Resultados en la sección 6.

## 6. Resultados del bloque A (corrido el 2026-11-03, 19:56 a 21:15 hora de Argentina)

**Cómo se corrió.** `v1/scripts/diagnostico_pre.py` (pasos `preparar`, `inferir`, `resumir`), en la CPU local, sin GPU. Mismos 3.696 trabajadores de validación que P7 (uno de cada 20). En cada persona se taparon **los mismos 462.590 casos de empleo** (persona, mes, variable) en los documentos de v0 y de v1, con la tasa de v0, sin tapar los `SINREG` y sin alteración del SOP. Cuatro modelos: v1 semilla 0 y v0 semillas 0, 1 y 2. Intervalos del 95 % con 2.000 remuestreos de trabajadores. Salidas en `v1/runs/diagnostico_pre/` (`resumen.txt`, `resumen.json`; ignoradas por git). Los números de esta sección reemplazan a los de P7 para comparar v0 con v1.

### D1 y D2. La diferencia es real: diez veces el ruido de semilla

NLL sobre los mismos casos. «Semilla» = diferencia entre el modelo de v0 de semilla 1 o 2 y el de semilla 0, o sea, cuánto cambia el resultado con sólo re-entrenar.

| Variable | NLL v0 (semilla 0) | v1 − v0 [IC 95 %] | Semilla 1 − 0 | Semilla 2 − 0 |
|---|---|---|---|---|
| salario | 3,718 | +0,101 [+0,096, +0,106] | +0,018 | +0,015 |
| tamaño | 1,017 | +0,120 [+0,116, +0,124] | +0,006 | +0,007 |
| rama | 1,026 | +0,114 [+0,108, +0,119] | +0,014 | +0,044 |
| provincia | 0,988 | +0,096 [+0,092, +0,100] | −0,005 | +0,005 |
| inicio de la empresa | 0,947 | +0,123 [+0,120, +0,127] | +0,013 | +0,014 |
| sexo | 0,661 | +0,026 [+0,012, +0,040] | +0,015 | +0,009 |
| año de nacimiento | 1,342 | +0,524 [+0,491, +0,558] | +0,063 | +0,037 |
| **empleo (todas las variables)** | **1,539** | **+0,111 [+0,109, +0,113]**; peor en el 89 % | **+0,009** | **+0,017** |

Dos modelos de v0 que sólo cambian la semilla difieren en 0,01-0,02 nats; v1 difiere en 0,11. **La regla de decisión fijada en la sección 4 se cumple: la diferencia está muy por encima de la banda de semilla y es real.** (Con el enmascarado propio de cada versión, P7 daba +0,18; unos 0,07 de eso eran el examen más difícil que se le tomaba a v1, ver D4.)

### D3. Documentos idénticos

En los 463 trabajadores sin meses sin registro, donde los dos modelos reciben exactamente la misma entrada: **+0,104 [+0,098, +0,110]**, peor en el 93 % (semillas de v0: +0,009 y +0,017). Prácticamente la misma diferencia que en el total.

### D4. La diferencia es pareja: no depende de los huecos de la persona

| Fracción de meses sin registro de la persona | Trabajadores | v1 − v0 | Semilla 1 − 0 | Semilla 2 − 0 |
|---|---|---|---|---|
| ninguno | 460 | +0,108 | +0,009 | +0,016 |
| hasta 10 % | 299 | +0,107 | +0,007 | +0,014 |
| 10-30 % | 452 | +0,110 | +0,009 | +0,015 |
| 30-50 % | 436 | +0,113 | +0,010 | +0,017 |
| 50-70 % | 483 | +0,110 | +0,011 | +0,019 |
| 70-100 % | 1.566 | +0,121 | +0,014 | +0,029 |

Con los mismos casos tapados, **la diferencia es casi igual para todos** (+0,11 a +0,12). La escalera de +0,10 a +0,75 de la sección 2 (P7) era efecto del enmascarado: a las personas con muchos huecos v1 les tapaba hasta un 50 % más de tokens. Mi observación 3.1 de `fase3_diagnosticos.md` explicaba entonces la **escalera** y no el nivel de base.

Según dónde está el caso dentro de la trayectoria:

| Posición del mes | Casos | v1 − v0 | Semilla 1 − 0 | Semilla 2 − 0 |
|---|---|---|---|---|
| dentro de un tramo de empleo (continuidad) | 437.836 | +0,108 | +0,008 | +0,016 |
| primer mes de la historia | 6.681 | +0,103 | +0,017 | +0,023 |
| nueva relación sin hueco previo | 6.248 | +0,129 | +0,035 | +0,049 |
| **primer mes después de un hueco** | 11.825 | **+0,212** | +0,029 | +0,045 |

El único lugar con un déficit extra es el primer mes después de un hueco (2,6 % de los casos): v1 rinde ahí unos 0,10 nats peor que en el resto. Paradójicamente, es donde v1 tiene más información (ve el hueco). Es el único dato que apoya la hipótesis de la ventana local: en v1 los meses sin registro se interponen entre ese mes y los empleos anteriores.

### D5 y un hallazgo nuevo: el modelo no sabe qué variable se le pregunta

La regla «copiar el mismo campo del mes vecino visible» acierta mucho más que los modelos en todo menos el salario:

| Variable | Acierto de la regla | v0 semilla 0 | v1 |
|---|---|---|---|
| salario | 0,152 | 0,115 | 0,107 |
| tamaño | 0,944 | 0,537 | 0,504 |
| rama | 0,949 | 0,629 | 0,579 |
| provincia | 0,978 | 0,519 | 0,535 |
| inicio de la empresa | 0,970 | 0,594 | 0,555 |

La causa está en la codificación, y es la misma en v0 y v1. Los cinco tokens de un mes de empleo comparten fecha, edad y segmento, y **no hay nada que indique la posición de cada token dentro del evento**. Si en un mes se tapa un token, el modelo ve un `[MASK]` con la fecha de ese mes, pero no sabe si se le pregunta por la rama, el tamaño o la provincia. Sólo puede deducirlo por descarte, mirando qué variables del mismo mes quedaron visibles, y si se taparon dos o más tokens del mismo mes ni siquiera eso. La regla de copia, en cambio, sabe qué campo busca. Medido:

| Tokens tapados en el mismo mes | Casos | Acierto v0 | Acierto v1 | Predice un valor de **otra variable**: v0 | v1 |
|---|---|---|---|---|---|
| 1 (deducible por descarte) | 77.518 | 0,669 | 0,615 | 0,261 | **0,344** |
| 2 | 175.546 | 0,506 | 0,481 | 0,477 | 0,506 |
| 3 o más | 209.526 | 0,386 | 0,377 | 0,607 | 0,617 |

(Las semillas 1 y 2 de v0 dan lo mismo que la 0, con diferencias de 0,005 o menos. v1 casi nunca predice un `SINREG` en un mes de empleo: 0,2 %. El año de nacimiento se responde con un token de sexo en el 27 % de los casos, en las dos versiones.)

Consecuencias:

- **Gran parte del error del MLM, en las dos versiones, es «no sé qué variable me preguntan»**, no «no sé el valor». La NLL por categoría (P6, P7) mezcla las dos cosas; leída como «cuánto sabe el modelo de rama» es engañosa.
- **El déficit de v1 está sobre todo ahí**: es máximo cuando la variable se puede deducir por descarte (+0,19 con un solo token tapado; predice otra variable 34 % contra 26 %) y mínimo cuando no se puede (+0,07 con tres o más). v1 aprendió peor a reconocer qué variable falta en un mes.
- Viene del diseño tomado de life2vec: no es un error de v1. Darle a cada token un indicador de su campo (por ejemplo, un embedding de «posición dentro del evento», sumado como la fecha y la edad) eliminaría la ambigüedad. Es un cambio de diseño que afecta a v0 y v1 por igual y exige re-entrenar.

### D6. SOP: la mejora de v1 es entera el atajo

La primera versión de esta prueba comparaba cada alterado contra **todos** los normales. Resultó inadecuada: los alterados «invisibles», idénticos a su normal, daban AUC 0,77-0,82 en lugar de 0,5, porque la comparación mezcla el tipo de persona (el modelo les da menos P(normal) a ciertos documentos aunque no estén alterados). Por eso la medida que vale es la **pareada**: la probabilidad de que el modelo le dé menos P(normal) al alterado que al documento normal **de la misma persona**. Con ella los invisibles dan ~0,5, como corresponde.

| | v0 semilla 0 | v0 semilla 1 | v0 semilla 2 | v1 |
|---|---|---|---|---|
| Todos los alterados | 0,757 | 0,766 | 0,773 | 0,843 [0,831, 0,855] |
| Con atajo (35 % en v1) | — | — | — | **0,998** [0,995, 1,000] |
| Invisibles | 0,63 (n = 35) | 0,40 (n = 35) | 0,46 (n = 35) | 0,476 [0,41, 0,55] (n = 170) |
| **Resto** (comparable) | **0,759** | **0,770** | **0,776** | **0,782** [0,764, 0,798] |

- v1 **usa el atajo**: acierta prácticamente todos esos casos.
- Sin el atajo, v1 queda en 0,782, apenas por encima de las tres semillas de v0 (0,759-0,776) y con un intervalo que se superpone con ellas. **v1 no aprendió el orden de las trayectorias mejor que v0.** La AUC 0,913 de `fase3.md` y la conclusión «v1 resuelve mejor el SOP» quedan **retiradas**.
- (Los niveles son menores que los de la validación del entrenamiento, 0,865 y 0,913, porque acá el enmascarado y el sorteo de alteraciones son otros. Lo que importa es la comparación dentro de esta tabla.)

### Qué cambia en la lectura de la fase 3

1. **v1 predice peor el empleo que v0, de verdad**: +0,11 nats sobre los mismos casos, diez veces el ruido de semilla, parejo para todas las personas y presente aun con documentos idénticos. No es la codificación de la persona, es el modelo entrenado.
2. El déficit está sobre todo en **reconocer qué variable falta** dentro de un mes, y hay un extra en el **primer mes después de un hueco**.
3. **El SOP no mejoró**: la ganancia aparente era el atajo.
4. Entre las hipótesis de la sección 3, la semilla queda descartada. La especialización en documentos con huecos queda debilitada: el déficit es igual en personas sin huecos. La que mejor encaja es la **dilución del gradiente**: con ~23 % de objetivos `SINREG` casi gratis, el empleo pesa menos en el MLM, mientras el SOP mantiene su peso fijo y además ofrecía un atajo fácil de aprender. Esta hipótesis sólo se prueba re-entrenando.
5. Esto **no dice todavía que v1 sea peor en las tareas**. Lo decide la fase 4.

## 7. Qué sigue (para decidir)

1. **Fase 4 con la `pre_150ep` actual de v1**, tal como estaba planeada (ajuste fino de T1 a T4 con 3 semillas y GRU). Sigue siendo la prueba que importa. El déficit del MLM es un aviso, no un veredicto.
2. **Re-entrenar v1 con los arreglos del bloque B** antes o en paralelo: SOP con los `SINREG` fijos y `SINREG` sólo como entrada, o con peso. Así se prueba la dilución y se obtiene un v1 sin atajo. Cuesta otras ~5,5 h de GPU en Mendieta.
3. **Indicador de campo dentro del evento** (D5): es un cambio de diseño para las dos versiones y, por lo tanto, una decisión de alcance: obliga a re-entrenar también v0 si se quiere comparar.

Mi recomendación: correr primero la 1, que no requiere re-entrenar. Si T1 a T4 empeoran respecto de v0, pasar a la 2. La 3 conviene discutirla aparte, porque toca la comparación con life2vec.
