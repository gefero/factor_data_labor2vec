# Fase 7: espacio de personas (mapas, TCAV y saliency)

Plan: [plans/PLAN_fase_7.md](../plans/PLAN_fase_7.md). Tablas completas y figuras en [fase7/](fase7/) (`resultados.md`, mapas y historias). Todo sale de `python -m labor2vec.analisis.personas analizar` sobre lo que dejó `analisis.extraer` en Mendieta (jobs 1181624, 1181625 y 1181626, de 3 a 12 minutos cada uno).

## Qué se hizo

La fase 5 miró el espacio de **conceptos** (los 519 tokens). Esta mira el espacio de **personas**: qué representa el modelo para cada trabajador, y qué partes de su historia usa para predecir.

Se usaron tres modelos, siempre sobre el test:
- **T1 ajustado** (`t1_l2v_lr0001`, semilla 0): salida del empleo registrado, 18.337 personas.
- **T3 ajustado** (`t3_l2v_s0`, semilla 0): cambio de rama, 15.316 personas.
- **Pre-entrenado sin ajustar** (`pre_150ep`): sirve de control para saber qué organización ya existía antes del fine-tuning.

Chequeo previo: los logits que se extrajeron reproducen la AUL de las predicciones guardadas (T1: 0,6127 contra 0,6128; T3: 0,7263 contra 0,7266). Las diferencias de probabilidad de hasta 0,10 (T1) y 0,21 (T3) vienen de que el entrenamiento usó bf16 y la extracción usa fp32.

Los «conceptos» de personas se construyeron del panel, con el estado a 2015: sexo, rama (`letra`, las 6 más frecuentes), tamaño de la empresa principal, tercil de salario, tercil de antigüedad en el empleo registrado y tercil de edad (cortes con train).

## 1. Mapa de personas

Cada persona se resume en un vector (la salida del *pooling* de la cabeza; en el pre-entrenado, el promedio de la última capa) y se proyecta a 2D con DensMAP y PaCMAP. Figuras: `fase7/mapa_{t1,t3,pre}_{densmap,pacmap}.png`.

Para no depender de la proyección se midió también la **pureza de los 5 vecinos más cercanos**: de cada persona, qué fracción de sus 5 vecinos comparte el atributo (entre paréntesis, lo esperable al azar; submuestra de 4.000 personas del test).

| modelo | y | sexo | rama | tamaño | salario | antigüedad | edad |
|---|---|---|---|---|---|---|---|
| T1 ajustado | 0,742 (0,729) | 0,641 (0,556) | 0,424 (0,205) | 0,614 (0,364) | **0,731** (0,334) | 0,570 (0,333) | 0,559 (0,335) |
| T3 ajustado | 0,827 (0,812) | 0,657 (0,557) | **0,615** (0,203) | 0,597 (0,367) | 0,601 (0,333) | 0,564 (0,333) | 0,573 (0,336) |
| pre-entrenado | 0,734 (0,716) | 0,659 (0,552) | 0,537 (0,205) | **0,679** (0,362) | 0,657 (0,334) | 0,429 (0,333) | 0,416 (0,335) |

Lo que se ve:
- **El espacio está organizado por rasgos de la persona, no por el resultado.** En todos los modelos, personas parecidas en salario, rama o tamaño quedan cerca. Pero los vecinos casi no comparten la clase real `y` (0,742 contra 0,729 al azar en T1): el mapa de T1 muestra un gradiente claro del logit, con los dos colores de `y` mezclados. Es coherente con que T1 es una tarea difícil (AUL 0,61).
- **El salario domina en T1** (0,731 contra 0,334 al azar) y **la rama domina en T3** (0,615 contra 0,203). Cada ajuste fino reorganiza el espacio hacia lo que le sirve a su tarea.
- **Antigüedad y edad casi no están en el pre-entrenado** (0,429 y 0,416, cerca del 0,333 del azar) y **aparecen con el ajuste** (0,56 a 0,57). El fine-tuning las incorpora.
- **Tamaño y rama estaban más marcados antes del ajuste que después** en T1 (tamaño 0,679 contra 0,614; rama 0,537 contra 0,424). El ajuste de T1 las diluye en favor del salario.

La pureza es una medida sobre una submuestra de 4.000 personas, con 100 permutaciones para el azar, y no se repitió con otra semilla ni con otro valor de k.

## 2. TCAV

Para cada concepto se entrena una dirección (CAV) que separa a las personas del concepto de las demás. El puntaje TCAV es la fracción de personas para las que mover el estado de todos sus tokens en esa dirección **sube** el logit de la clase 1. Si el concepto no importa, el puntaje es parecido al de direcciones al azar (el «nulo», que da ~0,5). Se probó en la última y en la penúltima capa del encoder. Tablas completas en `fase7/resultados.md`.

**Resultado, con una advertencia grande.** En T1 casi todos los puntajes son 0,000 o 1,000, y son **iguales para las personas con `y = 1` y con `y = 0`**. Eso significa que el gradiente apunta casi en la misma dirección para todas las personas (la suma de gradientes sobre los tokens es muy parecida de una persona a otra). El TCAV entonces no dice «para qué personas importa el concepto»: dice, para todo el mundo a la vez, **si esa dirección sube o baja el logit**. Es una lectura legítima, pero mucho más pobre que la que se esperaba. En T3 pasa lo mismo con muchos conceptos (rama, antigüedad, salario alto), pero otros tienen puntajes intermedios, con mucha dispersión entre CAVs (desvíos de 0,2 a 0,4) y a veces distintos según la clase (por ejemplo, «mayor» da 0,54 con `y = 0` y 0,83 con `y = 1`).

Con esa salvedad, en T1 (última capa) la dirección sube la probabilidad de salir del empleo registrado para: mujer, empresa chica, salario bajo, antigüedad alta, edad mayor, ramas 11 y 14. Y la baja para: empresa grande, salario alto, antigüedad baja, joven, ramas 9 y 12. En T3 los signos cambian en varios conceptos (por ejemplo, salario alto sube la probabilidad de cambiar de rama y salario bajo la baja), y en la última capa de T3 varios conceptos no se distinguen del nulo (mujer, empresa chica, joven; empresa grande y mayor sólo se distinguen en las personas con `y = 1`).

Cuidados:
- **Los signos no son estables entre capas.** Por ejemplo, en T3 «empresa chica» da 0,43 a 0,63 en la última capa y 0,97 a 0,99 en la penúltima; «salario bajo» da 0,13 a 0,22 en la última y 0,52 a 0,64 en la penúltima. Tomar la última capa como «la» respuesta sería una elección arbitraria.
- **Los p-valores con desvío cero no son confiables.** Cuando los 50 puntajes de un concepto son todos 1,000 (o todos 0,000), el test de Welch queda mal condicionado (scipy avisa de «precision loss»). Los p de 1e-9 hay que leerlos como «claramente distinto del nulo», no como un número exacto.
- **La exactitud de los CAVs** (0,73 a 0,98) dice que los conceptos son legibles de forma lineal en el espacio: la edad se separa mejor (0,94 a 0,97) que el sexo (0,73). Eso es independiente de si influyen en la predicción.
- Se probó un solo modelo por tarea (semilla 0).

## 3. Saliency (SmoothGrad)

Para 2.000 personas de T1 y 1.795 de T3 (muestra estratificada por `y` y por quintil del logit; en T3 algunas celdas tenían menos gente) se promedió el gradiente del logit sobre 25 copias con ruido en los embeddings de entrada. El puntaje de cada token es qué tan sensible es la predicción a él. El cociente de la tabla es «% del puntaje / % de los tokens».

| categoría de token | T1 (y = 1) | T3 (y = 1) |
|---|---|---|
| `[SEP]` y especiales | 1,76 | 1,84 |
| antigüedad de la empresa | 1,43 | 1,41 |
| tamaño | 0,79 | 0,91 |
| provincia | 0,75 | 0,76 |
| salario | 0,72 | 0,52 |
| rama | 0,54 | 0,56 |

Lo que se ve:
- **La historia reciente pesa mucho más que la lejana.** Los eventos de los últimos 24 meses tienen un cociente de 1,8 a 2,2 en los dos modelos y en las dos clases; los de más de 5 años, 0,5 a 0,6. Es el resultado más claro del informe.
- **Los `[SEP]` pesan casi el doble de lo que les toca por cantidad** (cada evento termina en un `[SEP]`, y eso es el 17 % de los tokens). Es una conducta conocida de los transformers: el separador actúa como lugar donde se resume el evento. No quiere decir que el separador «signifique» algo.
- **El token de antigüedad pesa de más (1,4) y los de rama y salario pesan de menos.** Atención: la antigüedad (con la rama y la provincia) es un atributo de «último valor», justamente el que el plan marcó como posible fuga de información posterior al corte. Este análisis no permite decir si el modelo usa esa antigüedad por una fuga o porque es genuinamente informativa. El control para eso es la ablación de la fase 8.
- **Los perfiles de `y = 1` y de `y = 0` son casi idénticos**, tanto en T1 como en T3. El modelo mira los mismos tipos de token en ambos grupos.
- **La hipótesis sobre T3 no se confirma con este análisis.** Se esperaba que el modelo prestara más atención a los eventos donde la persona cambia de empresa. El puntaje medio por token es prácticamente igual en eventos con cambio y sin cambio (T1: 1,48e-3 contra 1,47e-3; T3: 1,48e-3 contra 1,49e-3). Pero el detector de «cambio» es burdo: marca un evento cuando cambia la rama o el tamaño respecto del evento anterior. Así que lo correcto es decir «no se encontró evidencia con esta medida», no «el modelo no mira los cambios». El control del plan de la fase 6b, que mezcla el orden de los eventos, sigue pendiente.

Figuras con la línea de tiempo de 4 personas por modelo: `fase7/historias_{t1,t3}.png`. Se ve el mismo patrón: puntajes bajos y parejos hasta unos 2 años antes del corte, y los picos (sobre todo `[SEP]` y antigüedad) concentrados en los últimos meses.

## Limitaciones

- Un modelo (semilla 0) por tarea; no se mide cuánto varían estos resultados entre semillas.
- TCAV degenerado (puntajes en 0 o 1, iguales en ambas clases) y con signos que dependen de la capa.
- La medida de saliency es de gradiente × entrada, con 25 copias; no se comparó con otro método de atribución.
- El detector de eventos de cambio de empresa es una aproximación.
- No hay comparación con el paper: sus mapas y TCAV son de otra tarea y otros datos.

## Qué sigue

- Fase 8: ablación con y sin los atributos de «último valor» (antigüedad, rama, provincia), que es lo que responde la duda de fuga de información, y control que mezcla el orden de los eventos para la hipótesis de T3.
- Si se quiere un TCAV más informativo, calcularlo sobre otra cantidad (por ejemplo, derivada por token y no sumada) o sobre una capa intermedia; hoy no está planificado.
