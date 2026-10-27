# Fase 9: el espacio de personas, para interpretarlo y para usarlo

Plan: [plans/PLAN_fase_9.md](../plans/PLAN_fase_9.md). Tablas completas con intervalos de confianza en [fase9/](fase9/) (`resultados_sondas.md`, `resultados_ejes.md`). Todo sale de `python -m labor2vec.analisis.espacio {proporciones,sondas,ejes}` sobre los vectores que dejó `analisis.extraer` en la fase 7 (`runs/fase7/{pre,t1,t3}/personas.npz`).

**Estado (2026-10-26).** Hecha la etapa A1 (sondas lineales) y A2 (ejes). Pendientes: A3 (tipologías de trayectorias y comparación con optimal matching) y la etapa B (estabilidad entre semillas, trayectorias en el tiempo, intervenciones; requieren el cluster). Este informe se completa a medida que se cierren.

---

## 1. Por qué esta fase

La fase 7 dejó dos resultados débiles sobre el espacio de personas:

1. **El TCAV salió degenerado.** El puntaje es la fracción de personas para las que mover el estado en la dirección de un concepto sube el logit. Pero el gradiente del logit es casi el mismo para todas las personas (coseno 0,98 entre dos personas cualesquiera en T1), porque la cabeza del modelo es casi lineal sobre el resumen de la persona. Entonces el puntaje da 0 o 1 para todos a la vez y no distingue personas. No es un error de implementación: es una limitación de aplicar TCAV a una cabeza así.
2. **Los mapas no tenían con qué compararse.** Vimos que personas cercanas se parecen en salario o en rama, pero no sabemos si un simple conteo de tokens daría el mismo orden.

Esta fase ataca el espacio de dos maneras, que responden a dos preguntas distintas:

- **Interpretarlo:** ¿qué información guarda el vector de cada persona, y es algo más que «qué tokens tuvo»? En particular, ¿guarda la *dinámica* de la historia laboral (cuántos cambios, hace cuánto, huecos), que es lo que justificaría usar un modelo de secuencias?
- **Usarlo:** ¿se pueden armar tipologías de trayectorias laborales desde el espacio del modelo, y cómo se comparan con el método clásico de la sociología (análisis de secuencias con optimal matching)? Esta parte (A3) todavía no está hecha.

Este informe desarrolla en detalle la primera parte.

---

## 2. Los insumos

### 2.1 Los espacios que se comparan

Cada persona del test (y de val) tiene un vector. Se usan cuatro, más las referencias:

| nombre | qué es | dimensión |
|---|---|---|
| `pre` | encoder pre-entrenado `pre_150ep` (semilla 0) **sin ajustar**; promedio, sobre los tokens reales de la persona, de la salida de la **última** capa | 280 |
| `pre_pen` | lo mismo con la **penúltima** capa | 280 |
| `t1` | encoder ajustado a T1 (`t1_l2v_lr0001`, semilla 0); salida del *pooling* con atención de la cabeza (lo que entra a la capa lineal final) | 280 |
| `t3` | ídem para T3 (`t3_l2v_s0`) | 280 |
| `tokens` | **referencia**: proporción de cada uno de los 509 tokens (sin especiales ni fondo) en el documento que ve el transformer, más sexo y edad | 511 |
| `tokens + X` | la referencia con el vector `X` concatenado | 511 + 280 |

Notas:

- La fila `tokens` es la misma representación que usa la regresión logística de conteos de las fases 6 a 8 (la del paper original, salvo que el paper cuenta por intervalos de un año; ver la sección 7).
- Las cohortes son: T1 y `pre`, 18.303 personas en val y 18.337 en test; T3, 15.391 y 15.316. Las proporciones de tokens se recortan a la cohorte de cada espacio para que la comparación sea sobre las mismas personas.
- `pre` y `pre_pen` no pasan por ninguna cabeza de tarea; los vectores de `t1` y `t3` sí, porque están ajustados.

### 2.2 Lo que se les pide adivinar

Son características de cada persona que **no fueron etiquetas de entrenamiento de ningún modelo**: se calcularon del panel después.

**Estáticas** (estado en 2015-12, de `conceptos_personas.parquet`):

- *categóricas:* sexo, rama (`letra`, las 6 más frecuentes), tamaño de la empresa principal (grande, mediana, chica), tercil de salario, tercil de antigüedad, tercil de edad (los cortes se calculan con train);
- *continuas:* edad en años, percentil salarial de 2015, antigüedad en años (años desde el primer mes de la persona en el panel).

**De dinámica** (las 20 variables de `tareas/tiempo.py`, las mismas de la fase 8b). Son cosas que le *pasaron* a la persona y cuándo:

- antigüedad;
- meses desde el último cambio de empresa, de tamaño y de rama;
- cantidad de cambios de empresa, tamaño y rama en los últimos 12, 24 y 60 meses;
- cantidad de huecos (meses sin registro) en 24 y 60 meses;
- diferencia de percentil salarial a 12 y 36 meses, y si esa diferencia existe (`hay_dif_12`, `hay_dif_36`);
- meses desde el último registro.

Preparación: a los meses y conteos se les aplica $\log(1+x)$ y todas las variables se estandarizan con la media y el desvío de **val** (no de test). La última variable, «meses desde el último registro», es **constante (0) en las 18.337 personas del test**, porque la cohorte exige estar registrado en 2015-12. No hay nada que predecir y se descarta del análisis; ver la sección 5.1.

---

## 3. La lógica de una sonda

### 3.1 La idea

Una *sonda* es un modelo simple que intenta adivinar una característica de la persona mirando únicamente su vector. Si lo logra, esa información estaba guardada en el vector de forma accesible. Si no, no.

La sonda es **lineal a propósito**. Si hiciera falta un modelo complejo para recuperar la información, no sabríamos si la tenía el vector o la construyó la sonda. Con una sonda lineal, lo que se mide es información que se lee con una sola operación (una suma ponderada de las 280 coordenadas).

Se entrena con las personas de **val** y se evalúa con las de **test**, que la sonda no vio. Las medias y desvíos para estandarizar salen de val.

### 3.2 Variables continuas: regresión ridge

Sea $\mathbf z_i$ el vector estandarizado de la persona $i$ (coordenada a coordenada: $z_{ij} = (x_{ij}-\mu_j)/\sigma_j$, con $\mu_j,\sigma_j$ de val) e $y_i$ el valor a adivinar. La sonda predice

$$\hat y_i = \boldsymbol\beta^\top \mathbf z_i + b ,$$

y los pesos $\boldsymbol\beta$ se eligen con las personas de val minimizando

$$\hat{\boldsymbol\beta} = \arg\min_{\boldsymbol\beta}\ \sum_{i\in val}\bigl(y_i - \boldsymbol\beta^\top \mathbf z_i - b\bigr)^2 + \alpha\,\lVert\boldsymbol\beta\rVert^2 ,$$

con solución cerrada $\hat{\boldsymbol\beta} = (Z^\top Z + \alpha I)^{-1} Z^\top y$ (con $Z$ y $y$ centrados). El término $\alpha\lVert\boldsymbol\beta\rVert^2$ encoge los pesos hacia cero. Hace falta porque las 280 coordenadas están muy correlacionadas entre sí, y sin penalización la sonda ajustaría ruido. El parámetro $\alpha$ se elige entre $\{0{,}1;\,1;\,10;\,10^2;\,10^3;\,10^4\}$ por validación cruzada dentro de val.

**Medida:** con los pesos ya fijos, se predice en test y se calcula

$$R^2 = 1-\frac{\sum_i (y_i-\hat y_i)^2}{\sum_i (y_i-\bar y)^2}.$$

Se lee como la fracción de la variación de $y$ que el vector explica con una combinación lineal. $R^2=1$ es predicción perfecta; $R^2=0$ es no mejorar a decir siempre el promedio; puede ser negativo. El intervalo de confianza al 95 % sale de remuestrear las personas del test con reposición 200 veces.

*Ejemplo mínimo, para fijar ideas.* Con una sola coordenada y 4 personas en val, $z=(-1{,}5;\,-0{,}5;\,0{,}5;\,1{,}5)$ e $y=(-1;\,-1;\,1;\,1)$, se tiene $\sum zy=4$ y $\sum z^2=5$, de modo que $\hat\beta = 4/(5+\alpha)$: vale $0{,}8$ con $\alpha=0$ y $0{,}4$ con $\alpha=5$. En un test con $z=(-1;1)$ e $y=(-0{,}8;1{,}2)$, el primero da $R^2=0{,}92$ y el segundo $0{,}60$. Con 280 coordenadas ruidosas suele convenir un $\alpha>0$, y por eso se lo elige con datos.

### 3.3 Variables categóricas: logística multinomial

Para rama, tamaño, y los terciles de salario, antigüedad y edad no hay un número sino una clase $k\in\{1,\dots,K\}$:

$$P(k\mid\mathbf z)=\frac{e^{\mathbf w_k^\top\mathbf z}}{\sum_l e^{\mathbf w_l^\top\mathbf z}} .$$

Se usan pesos de clase balanceados para que las clases chicas cuenten igual que las grandes. La penalización $C$ (inverso de $\alpha$) se elige entre $\{0{,}001;\,0{,}01;\,0{,}1\}$ en una porción de val (70 % para entrenar, 30 % para elegir), y después se reentrena con todo val. Para acelerar, se usan como máximo 8.000 personas de val.

**Medida:** la *exactitud balanceada* en test, el promedio del acierto por clase:

$$\text{exactitud balanceada}=\frac1K\sum_{k=1}^K\frac{\text{aciertos en la clase }k}{\text{personas de la clase }k}.$$

El valor esperado al azar es $1/K$: 0,50 para sexo, 0,33 para los terciles y tamaño, 0,17 para las 6 ramas. La exactitud balanceada, a diferencia de la común, no se infla por predecir siempre la clase más frecuente.

### 3.4 Qué prueba una sonda y qué no

- **Prueba** que la información está en el vector y se puede leer linealmente. Una sonda que acierta dice que el transformer la guardó.
- **No prueba** que el modelo la *use* para predecir.
- **La falla de una sonda lineal no prueba ausencia.** La información podría estar codificada de forma no lineal y una sonda más potente la encontraría. Esto es relevante más abajo, en el caso de la evolución del salario.
- **Un número aislado no se puede interpretar.** Un $R^2$ de 0,5 puede ser mucho o poco. Lo informativo es la comparación entre filas, y por eso están las referencias.

---

## 4. El diseño de las comparaciones

Cada espacio se sondea con los mismos 6 objetivos categóricos, 3 continuos estáticos y 19 de dinámica. Cada contraste responde una pregunta distinta:

| contraste | qué responde |
|---|---|
| `pre` contra `tokens` | ¿el vector aprendido guarda algo que el conteo de tokens no tiene? |
| `tokens + pre` contra `tokens` y contra `pre` | si el vector aporta información *nueva*, concatenarlo mejora al conteo; si el vector ya contiene al conteo, concatenar el conteo no mejora al vector |
| `pre` contra `t1` y `t3` | ¿qué cambia con el ajuste fino a una tarea? |
| `pre` contra `pre_pen` | ¿depende de la capa? |

La referencia clave es `tokens`, porque **no tiene información de tiempo**: sólo cuenta cuántas veces aparece cada token, sin fechas ni orden. Por eso, las diferencias entre el vector y los tokens en las variables de dinámica miden información temporal que el conteo no puede tener. Más abajo se discute que esto es *tiempo*, no necesariamente *orden*.

---

## 5. Resultados de las sondas

Cifras en test. Entre corchetes, el IC al 95 % por bootstrap, cuando se cita una cifra suelta; las tablas completas con todos los intervalos están en `fase9/resultados_sondas.md`.

### 5.1 Una corrección sobre los datos

La columna «meses desde el último registro» muestra $R^2=1{,}000$ en todos los espacios en `resultados_sondas.md`. **Es un artefacto:** la variable es constante en test, así que el denominador de $R^2$ es casi cero y mi fórmula devolvía 1. Ya está corregido en el código (devuelve NaN si la variable no varía) y hay una prueba que lo cubre, pero las tablas guardadas fueron generadas antes de la corrección. Hay que ignorar esa columna; se arregla la próxima vez que se regeneren las sondas. No afecta a ninguna otra cifra.

La edad con tokens ($R^2=1{,}000$) y el sexo con tokens (exactitud 0,999) sí son correctos pero **triviales**: sexo y edad entran directamente como columnas de la representación de tokens, así que ahí la sonda sólo recupera una entrada. No se usan para comparar.

### 5.2 Variables estáticas

Exactitud balanceada de las categóricas (entre paréntesis, el azar):

| espacio | sexo (0,50) | rama (0,17) | tamaño (0,33) | salario (0,33) | antigüedad (0,33) | edad (0,33) |
|---|---|---|---|---|---|---|
| `pre` | 0,767 | 0,856 | 0,874 | 0,835 | **0,843** | 0,969 |
| `tokens` | (trivial) | **0,888** | 0,855 | 0,824 | 0,626 | 0,955 |
| `tokens + pre` | (trivial) | 0,889 | 0,861 | 0,833 | 0,826 | 0,972 |
| `t1` | 0,748 | 0,844 | 0,908 | 0,887 | 0,842 | 0,942 |
| `t3` | 0,747 | 0,901 | 0,936 | 0,897 | 0,834 | 0,961 |

Y en las continuas ($R^2$): salario `pre` 0,836 contra `tokens` 0,801; antigüedad en años `pre` **0,906** [0,903; 0,909] contra `tokens` **0,506** [0,491; 0,516].

**Interpretación.**

1. **En rama, tamaño, salario y edad, el vector pre-entrenado y el conteo de tokens llegan a lo mismo.** La diferencia es de uno a tres puntos, en ambos sentidos: el conteo gana en rama (0,888 contra 0,856), el vector en tamaño (0,874 contra 0,855). Esto es lo esperable. Esas características se leen directamente de qué tokens aparecen (hay tokens de rama, de tamaño y de salario), así que un buen resumen de la persona tiene que contenerlas, y el conteo también las tiene. Que el vector las conserve es una buena noticia —no perdió lo que sabe un conteo—, pero no es un hallazgo.
2. **La excepción es la antigüedad, donde el vector gana por mucho** (0,843 contra 0,626 en terciles; $R^2$ 0,906 contra 0,506). La antigüedad es el largo de la historia de la persona en el panel. Un conteo de proporciones no sabe cuánto dura la historia: dos personas con la misma mezcla de tokens pero una con 5 años y otra con 15 tienen la misma proporción. El transformer, que ve las fechas (su codificación de tiempo), sí. Es la primera señal de que el vector guarda información temporal.
3. **El vector pierde el sexo.** Se lee con exactitud 0,767 [0,759; 0,772] contra 0,50 al azar: está por encima del azar pero lejos de lo que un conteo recupera sin esfuerzo (el sexo es una columna). Una explicación plausible, **no verificada**: el sexo entra al documento en un único token del fondo (`SEXO`) entre cientos o miles de tokens, y un promedio sobre todos los tokens (`pre`) lo diluye. Que `t1` y `t3` lo lean un poco peor (0,75) es compatible con que el ajuste fino no necesita el sexo. Vale la pena anotarlo para la parte de equidad (sección 8 del informe final): el modelo sí lo conserva, aunque no sea una dirección fuerte del espacio.
4. **El ajuste fino reorienta el espacio hacia lo que importa a la tarea.** `t1` mejora salario (0,887 contra 0,835) y tamaño (0,908 contra 0,874); `t3` mejora tamaño (0,936), salario (0,897) y rama (0,901 contra 0,856, y supera al conteo de tokens 0,890). Es coherente con que T1 (salida del empleo) y T3 (cambio de rama) dependen de dónde trabaja la persona.
5. **La penúltima capa se parece a la última** (`pre_pen` contra `pre`: diferencias de un punto o dos, con la edad 0,986 contra 0,969 y la rama 0,877 contra 0,856 un poco mejores). La elección de capa no cambia las conclusiones.

### 5.3 Variables de dinámica

$R^2$ de la sonda sobre cada vector. `tokens` es la referencia sin tiempo; en negrita, donde el vector supera claramente al conteo.

| variable | `pre` | `tokens` | `t1` | `t3` |
|---|---|---|---|---|
| antigüedad | **0,567** | 0,309 | 0,575 | 0,654 |
| meses desde cambio de empresa | **0,492** [0,479; 0,504] | 0,261 | 0,511 | 0,682 |
| meses desde cambio de tamaño | **0,546** | 0,270 | 0,544 | 0,634 |
| meses desde cambio de rama | **0,505** | 0,272 | 0,514 | 0,692 |
| cambios de empresa, 12 meses | **0,195** | 0,071 | 0,221 | 0,380 |
| cambios de empresa, 24 meses | **0,365** | 0,139 | 0,413 | 0,523 |
| cambios de empresa, 60 meses | **0,548** [0,535; 0,559] | 0,220 | 0,549 | 0,625 |
| cambios de tamaño, 60 meses | **0,238** | 0,043 | 0,213 | 0,229 |
| cambios de rama, 12 meses | **0,141** [0,119; 0,161] | 0,049 | 0,175 | 0,340 |
| cambios de rama, 60 meses | **0,495** [0,481; 0,509] | 0,171 | 0,486 | 0,612 |
| huecos, 24 meses | **0,284** | 0,054 | 0,338 | 0,244 |
| huecos, 60 meses | **0,492** [0,480; 0,504] | 0,130 | 0,500 | 0,422 |
| ¿hay diferencia salarial a 12 meses? | **0,428** | 0,002 | 0,259 | 0,180 |
| ¿hay diferencia salarial a 36 meses? | **0,613** [0,593; 0,631] | 0,071 | 0,539 | 0,535 |
| diferencia salarial, 12 meses | 0,022 | 0,009 | 0,031 | 0,032 |
| diferencia salarial, 36 meses | 0,039 | 0,017 | 0,116 | 0,092 |

**Interpretación.**

1. **El vector guarda bastante más dinámica que el conteo de tokens, y la diferencia no es marginal.** Casi todas las variables de dinámica tienen en `pre` entre dos y seis veces el $R^2$ de `tokens` (por ejemplo, cambios de empresa en 60 meses, 0,55 contra 0,22; huecos en 60 meses, 0,49 contra 0,13). Los intervalos de confianza no se tocan. Esto responde la pregunta central de la fase: el espacio contiene información sobre *cuándo* pasaron las cosas, que no se obtiene contando tokens.
2. **Las variables que más ganan son las que dependen de saber cuándo pasó algo.** «Meses desde el último cambio», «cuántos cambios en los últimos 60 meses», «cuántos huecos» y «si hay registro de hace 12 o 36 meses» exigen ubicar eventos en el tiempo. El conteo de proporciones no puede, por construcción. `hay_dif_12` y `hay_dif_36` son el caso extremo: valen 1 si la persona tiene un salario registrado 12 o 36 meses antes del corte, es decir, si su historia empezó hace al menos ese tiempo; es esencialmente una pregunta de cobertura temporal. Con tokens dan $R^2$ 0,002 y 0,071; con el vector, 0,428 y 0,613.
3. **Concatenar los tokens al vector casi no mejora.** `tokens + pre` da, por ejemplo, 0,565 en cambios de empresa a 60 meses (contra 0,548 de `pre`) y 0,533 en cambios de rama a 60 meses (contra 0,495). Lo único que mejora algo es `hay_dif_12` (0,58 contra 0,43) y los huecos (0,54 contra 0,49). Es decir: **el vector contiene casi todo lo que sabe el conteo, y además información temporal que el conteo no tiene.** No es que ambos sean complementarios; uno incluye al otro.
4. **El ajuste a T3 refuerza justo la dinámica de cambios.** Con `t3`, el $R^2$ de «meses desde el último cambio de empresa» pasa de 0,49 a 0,68 y el de «cambios de rama en 12 meses» de 0,14 a 0,34. Es coherente con la naturaleza de T3 (predecir si la persona cambia de rama): para eso el modelo necesita representar bien cuándo y cuánto cambió. T1 casi no cambia la dinámica (sus números están a un punto o dos de `pre`), consistente con que T1 depende más del nivel que de los cambios (la fase 8 ya mostró que T1 es el caso donde una logística con tiempo iguala al transformer).
5. **El ajuste fino también *olvida* cosas que la tarea no pide.** `hay_dif_12` baja de 0,43 (`pre`) a 0,26 (`t1`) y 0,18 (`t3`); los huecos bajan en `t3` (0,42 contra 0,49 en 60 meses). El pre-entrenamiento deja un espacio más general que el ajustado: el ajuste fino orienta el espacio hacia la tarea a costa de otras direcciones. Esto sugiere que, para armar tipologías generales, conviene usar el espacio pre-entrenado y no el ajustado (decisión que ya tomamos para A3).
6. **La evolución del salario casi no se lee linealmente en ningún espacio.** La diferencia de percentil salarial a 12 y 36 meses tiene $R^2$ entre 0,02 y 0,12 en todos los espacios; sólo en `t1` y `t3` sube a ~0,1 en 36 meses. Esto contrasta con que «existe una diferencia» sí se lee. Hay que ser cuidadoso con la lectura: una sonda lineal sobre un promedio no puede recuperar una *diferencia* entre el salario de hoy y el de hace 36 meses, porque eso exige emparejar tokens con sus fechas y restarlos, una operación no lineal. Que no se lea linealmente **no prueba que el transformer no la represente**: la atención puede hacerlo y el promedio perder el resultado. Para decidirlo haría falta una sonda no lineal (por ejemplo, un perceptrón chico) o intervenciones directas (etapa B3).

---

## 6. Ejes del espacio (A2)

### 6.1 Método

Un análisis de componentes principales (PCA) del vector `pre` del test busca la dirección en la que las personas más difieren entre sí (componente 1), después la siguiente, ortogonal a la anterior, y así. Para cada componente se calcula:

- la varianza que explica;
- la **correlación** con cada variable continua (edad, salario, antigüedad y las de dinámica);
- el **$\eta^2$** con cada característica categórica, es decir, la fracción de la varianza del componente que explican los grupos:
  $$\eta^2=\frac{\sum_g n_g(\bar x_g-\bar x)^2}{\sum_i (x_i-\bar x)^2}.$$

Es un análisis descriptivo que reemplaza al TCAV: dice qué organiza el espacio sin pasar por gradientes.

### 6.2 Resultados

| componente | varianza | continuas más correlacionadas | categóricas con mayor $\eta^2$ |
|---|---|---|---|
| PC1 | 22,3 % | meses desde cambio de tamaño $+0{,}34$; antigüedad $+0{,}27$; cambios de tamaño 60 meses $-0{,}23$ | tamaño 0,07; rama 0,02 |
| PC2 | 7,5 % | salario $+0{,}78$; antigüedad en años $+0{,}31$ | salario **0,56**; tamaño 0,37 |
| PC3 | 5,2 % | (todas $\lvert r\rvert<0{,}2$) | rama 0,22; salario 0,11 |
| PC4 | 4,5 % | (todas $\lvert r\rvert<0{,}1$) | salario 0,24; rama 0,07 |
| PC7 | 3,5 % | (todas $\lvert r\rvert<0{,}1$) | rama 0,31; tamaño 0,09 |

Los diez primeros componentes explican el 59,8 % de la varianza.

### 6.3 Interpretación

- **El componente 2 es claramente el nivel salarial** ($r=+0{,}78$, $\eta^2=0{,}56$), con una parte importante de tamaño de empresa. Es el único eje que se deja resumir con una etiqueta limpia.
- **Los componentes 3 y 7 se asocian con la rama** ($\eta^2$ 0,22 y 0,31), y el 4 con el salario. La rama no ocupa un único eje sino varios.
- **El componente 1, el más grande, no se explica bien con nuestras variables.** Explica el 22 % de la varianza pero su mayor correlación es 0,34 (es decir, $R^2\approx0{,}12$), con una mezcla de antigüedad y tiempo desde el último cambio de tamaño. Hay una señal de dinámica, pero débil. **Retiro lo que dije al ver estos números por primera vez** («el eje principal es de dinámica»): lo correcto es que el eje principal tiene una componente temporal modesta y el resto de lo que lo organiza no está capturado por ninguna de las variables que probamos. Una hipótesis a verificar es que sea un efecto de largo o de composición de la historia (el promedio sobre tokens refleja cuántos tokens de cada tipo hay), pero no está probado.
- **Importante:** los ejes principales son direcciones de máxima *varianza*, no de máximo *significado*. Que el eje 2 sea el salario dice que el salario es una fuente grande de variación entre personas, no que sea la dimensión más importante para el modelo.

---

## 7. Qué se puede concluir y qué no

**Se puede concluir, con respaldo en los datos:**

1. El espacio pre-entrenado guarda información de la historia laboral que un conteo de tokens no tiene. La evidencia es consistente: antigüedad y 17 de las 19 variables de dinámica (las dos excepciones son las diferencias salariales, que no se leen linealmente en ningún espacio), con intervalos de confianza que no se solapan, en dos capas distintas y en los tres modelos.
2. Esa información es de naturaleza temporal (cuándo, cuánto hace, cuántas veces en una ventana, huecos).
3. Lo que sabe un conteo está contenido en el vector: concatenar los tokens no agrega casi nada.
4. El ajuste fino reorienta el espacio hacia la tarea: T3 refuerza los cambios; T1, el nivel; ambos pierden direcciones que no necesitan.

**No se puede concluir, y por qué:**

1. **Que el modelo use la secuencia (el *orden* de los eventos).** El conteo no tiene fechas, así que la ventaja del vector prueba información de *tiempo*. Para mostrar dependencia del *orden* hay que correr los controles de la fase 8 (fechas permutadas, orden permutado) sobre los vectores y volver a sondear. Está en la etapa B3.
2. **Que el pre-entrenamiento sea la causa.** Falta el control de un encoder con pesos al azar o entrenado sólo en la tarea. Un encoder sin entrenar que promedia embeddings ya organiza a las personas por algo; sin esa referencia no sé cuánto aportan los 150 epochs de pre-entrenamiento.
3. **Que estas diferencias sean estables.** Todo sale de la semilla 0. Falta la etapa B1 (otras semillas).
4. **Que el modelo no represente la evolución del salario.** Sólo que no se lee con una sonda lineal sobre un promedio.

**Amenazas a la validez que conviene tener presentes:**

- **La comparación con el conteo no es del todo justa.** Las proporciones dividen por el total de tokens, así que pierden los conteos absolutos («cuántos cambios hubo»), algo que el paper original sí conserva al contar por intervalo de un año. Parte de la ventaja en `cambios_*` puede venir de ahí y no de la información temporal. Se puede cerrar con conteos por año; es barato pero no está hecho.
- **Dos variables de dinámica pueden leerse de atributos de «último valor».** La antigüedad y los cambios de rama usan información (rama y antigüedad de la relación) que los documentos traen como atributos de cada evento, y que la fase 8 señaló como posible fuga de información posterior al corte. Los tokens la tienen también, así que no favorece a un espacio sobre el otro, pero conviene repetir la sonda con el modelo `pre_150ep_sinuv` (sin esos atributos).
- **Comparaciones múltiples.** Son unos 25 objetivos por 12 espacios. Los intervalos de confianza son marginales, sin corrección. Las diferencias citadas son grandes (de decenas de puntos y con intervalos que no se tocan), así que no dependen de ello, pero los pares con diferencias de un punto o dos (por ejemplo, rama con `pre` contra `tokens`) no deben leerse como superioridad.
- **Las cohortes de T1 y T3 no son las mismas** (T3 es un subconjunto: quienes siguen en 2019). Las comparaciones entre `t1` y `t3` son entre cohortes distintas; las comparaciones contra `tokens` son siempre sobre la misma cohorte.

---

## 8. Qué sigue

Etapa A:

- **A3: tipologías.** Armar tipos de trayectoria desde el vector `pre` y compararlos con optimal matching sobre secuencias mensuales y con el espacio de proporciones de tokens. Antes se mide cuánto tarda el optimal matching con 500 personas.

Etapa B (Mendieta, con OK previo):

- **B1** estabilidad entre semillas; **B2** trayectorias en el espacio con ventanas que terminan en distintos años; **B3** intervenciones en la historia (que también permite separar *tiempo* de *orden*).

Ajustes pendientes en lo ya hecho:

- Regenerar las sondas para sacar el artefacto de «meses desde el último registro» (unos 30 minutos).
- Sumar como referencia un encoder sin pre-entrenar y la sonda con `pre_150ep_sinuv`.
- Opcional: conteos por año para igualar al paper.
