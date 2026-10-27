# Fase 9: el espacio de personas, para interpretarlo y para usarlo

Plan: [plans/PLAN_fase_9.md](../plans/PLAN_fase_9.md). Tablas completas en [fase9/](fase9/) (`resultados_{sondas,ejes,semillas,extras,controles,conteos,cortes,intervenciones}.md`). Las etapas A y de conteos salen de `python -m labor2vec.analisis.espacio {proporciones,sondas,ejes,conteos}` sobre los vectores que dejó `analisis.extraer` en la fase 7 (`runs/fase7/{pre,t1,t3}/personas.npz`); la etapa B, de `python -m labor2vec.analisis.etapa_b {semillas,cortes,intervenciones,controles,extras}` sobre las extracciones de `runs/fase9/` (jobs de Mendieta).

**Estado (2026-10-27).** Hechas las sondas (A1), los ejes (A2) y toda la etapa B (semillas, cortes, intervenciones, controles) más una línea de base con conteos absolutos. **No está hecha la parte de tipologías (A3):** el optimal matching de un solo canal se descartó y el análisis multicanal (MCSA) quedó planificado y suspendido (sección 8). La sección 9 reúne las conclusiones y **corrige varias de las que figuraban en la primera versión de este informe**; si una cifra de las secciones 1 a 6 contradice la sección 9, vale la sección 9.

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

> **Tres cosas distintas que en este proyecto se llaman «antigüedad».** (1) La **estática**, `antig_anios`: años entre el primer mes en que aparece la persona en el panel (cualquier relación, desde 1996-01) y 2015-12; es el largo de su historia registrada, no el tiempo en un empleo. (2) La de **dinámica**, `antiguedad` de `tareas/tiempo.py`: meses desde el primer mes de la *relación principal* (la de más meses registrados en 2015), es decir, el tiempo en el empleo actual. (3) El **token `ANT`**: el tramo del año de inicio de la *empresa*; es un atributo de la empresa, no de la persona. En las tablas, «antigüedad (años)» y los terciles de antigüedad son la (1); «antigüedad (dinámica)» es la (2); y «antigüedad de la empresa» es la (3). Una revisión de este informe (2026-10-27) encontró que no lo distinguía.

**De dinámica** (las 20 variables de `tareas/tiempo.py`, las mismas de la fase 8b). Son cosas que le *pasaron* a la persona y cuándo:

- antigüedad **en el empleo actual** (meses desde el primer mes de la relación principal; es otra variable que la «antigüedad en años» de las estáticas, ver el recuadro de abajo);
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
2. **La excepción es la antigüedad, donde el vector gana por mucho** (0,843 contra 0,626 en terciles; $R^2$ 0,906 contra 0,506). Esta antigüedad es la (1) del recuadro de la sección 2.2: el largo de la historia registrada de la persona en el panel, no el tiempo en un empleo ni la edad de la empresa. Un conteo de proporciones no sabe cuánto dura la historia: dos personas con la misma mezcla de tokens pero una con 5 años y otra con 15 tienen la misma proporción. En la primera versión escribí que el transformer lo sabe «porque ve las fechas» y que era la primera señal de información temporal. **Eso no se sostiene:** con las fechas permutadas la sonda sube (0,906 → 0,914, sección 7.3), y con conteos más el tamaño de la historia llega a 0,73. Lo más probable es que el vector conserve el *largo* de la historia (cuántos eventos hay), que las proporciones normalizan, y no las fechas.
3. **El vector pierde el sexo.** Se lee con exactitud 0,767 [0,759; 0,772] contra 0,50 al azar: está por encima del azar pero lejos de lo que un conteo recupera sin esfuerzo (el sexo es una columna). Una explicación plausible, **no verificada**: el sexo entra al documento en un único token del fondo (`SEXO`) entre cientos o miles de tokens, y un promedio sobre todos los tokens (`pre`) lo diluye. Que `t1` y `t3` lo lean un poco peor (0,75) es compatible con que el ajuste fino no necesita el sexo. Vale la pena anotarlo para la parte de equidad (sección 8 del informe final): el modelo sí lo conserva, aunque no sea una dirección fuerte del espacio.
4. **El ajuste fino reorienta el espacio hacia lo que importa a la tarea.** `t1` mejora salario (0,887 contra 0,835) y tamaño (0,908 contra 0,874); `t3` mejora tamaño (0,936), salario (0,897) y rama (0,901 contra 0,856, y supera al conteo de tokens 0,890). Es coherente con que T1 (salida del empleo) y T3 (cambio de rama) dependen de dónde trabaja la persona.
5. **La penúltima capa se parece a la última** (`pre_pen` contra `pre`: diferencias de un punto o dos, con la edad 0,986 contra 0,969 y la rama 0,877 contra 0,856 un poco mejores). La elección de capa no cambia las conclusiones.

### 5.3 Variables de dinámica

$R^2$ de la sonda sobre cada vector. `tokens` es la referencia sin tiempo; en negrita, donde el vector supera claramente al conteo.

| variable | `pre` | `tokens` | `t1` | `t3` |
|---|---|---|---|---|
| antigüedad en el empleo actual (2) | **0,567** | 0,309 | 0,575 | 0,654 |
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

1. **El vector guarda bastante más dinámica que el conteo de tokens, y la diferencia no es marginal.** Casi todas las variables de dinámica tienen en `pre` entre dos y seis veces el $R^2$ de `tokens` (por ejemplo, cambios de empresa en 60 meses, 0,55 contra 0,22; huecos en 60 meses, 0,49 contra 0,13). Los intervalos de confianza no se tocan. **Esta lectura se corrigió después** (secciones 7.3 y 7.4): la referencia de proporciones era demasiado débil, porque dividir por el total de tokens borra el largo de la historia. Con conteos en escala logarítmica más el tamaño de la historia, la ventaja desaparece o se achica mucho en casi todas las variables, y los controles muestran que no depende de las fechas ni del orden. Lo que queda de ventaja está en los huecos y en la existencia de cambios salariales recientes.
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

## 7. Etapa B: estabilidad, controles, trayectorias e intervenciones

Todo lo de esta sección sale de extracciones hechas en Mendieta (16 jobs cortos, todos `COMPLETED`, 28 carpetas en `runs/fase9/`), analizadas en local con `labor2vec.analisis.etapa_b`. Salvo que se indique otra cosa, las cifras son en test (18.337 personas; 15.316 en la cohorte de T3).

### 7.1 Estabilidad entre semillas (B1)

**Qué se hizo.** Se repitió el entrenamiento con dos semillas más: el pre-entrenado completo (`pre_150ep_s1` y `_s2`) y el ajuste fino de T1 y de T3 (`t1_l2v_lr0001_s1/_s2`, `t3_l2v_s1/_s2`; en estos sólo cambia el ajuste, a partir del mismo pre-entrenado). Sobre las mismas personas se compararon los espacios de dos formas, y se repitieron las sondas.

**Lo que se mide.**

- **CKA lineal:** similitud global entre dos espacios, invariante a rotaciones y a la escala. 1 significa idénticos salvo rotación; 0, nada en común.
- **Superposición de vecinos:** para cada persona, qué fracción de sus 10 vecinos más cercanos (coseno) es la misma en los dos espacios. Lo esperable al azar es 0,25 % (0,0025 en `resultados_semillas.md`).
- **ARI de las tipologías:** se arma un Ward de k = 8 en cada espacio y se mide el acuerdo entre particiones (1 = idénticas; 0 = lo esperable al azar).

**Resultados** (rango sobre los tres pares de semillas):

| espacio | CKA | vecinos en común | ARI de tipologías (k = 8) |
|---|---|---|---|
| `pre` | 0,45 a 0,64 | 41 a 48 % | 0,12 a 0,15 |
| `t1` | 0,69 a 0,84 | 12 a 18 % | 0,20 a 0,22 |
| `t3` | 0,63 a 0,74 | 15 a 22 % | 0,17 a 0,22 |

**Las sondas sí son estables.** Entre las tres semillas, el rango (máximo menos mínimo) de las sondas de `pre` es de 0,002 a 0,011 en las estáticas y de 0,004 a 0,071 en las de dinámica. Para las variables que más importan: cambios de empresa a 60 meses 0,548, 0,553 y 0,569; huecos a 60 meses 0,492, 0,522 y 0,508; $R^2$ de antigüedad en años 0,906, 0,903 y 0,906. La ventaja sobre las proporciones (por ejemplo 0,22 en cambios de empresa a 60 meses) se mantiene en cada semilla. Los modelos ajustados varían algo más en la letra de rama (hasta 0,05) pero no en lo demás.

**La geometría no lo es.** Entre semillas coincide menos de la mitad de los vecinos en el pre-entrenado (y menos de uno de cada cinco en los ajustados). El ARI de 0,12 a 0,22 dice que dos personas que quedan juntas en una tipología de una semilla tienen poca probabilidad de quedar juntas en otra.

**Cómo lo interpreto** (coherente con los números, no probado con un experimento aparte). Una sonda sólo necesita que *exista* una dirección del espacio que codifique la variable; esa dirección puede estar en cualquier parte y girar con la semilla. Los vecinos y los grupos, en cambio, dependen de las distancias finas entre todas las personas, y eso sí cambia. Sumado a la silueta baja de las tipologías del `pre` que medí en la etapa A (0,03 a 0,05: los datos forman un continuo y no nubes separadas), cabría pensar que cortar el espacio en k tipos es sensible a movimientos chicos. **La sección 7.1b mostró que eso no alcanza para explicarlo.**

**Consecuencias.**

1. **Las tipologías del espacio del modelo no se pueden presentar como un resultado.** Los 8 tipos de una corrida no los reproduce otra.
2. **Techo para cualquier comparación entre tipologías.** Un ARI de 0,12 a 0,22 entre semillas es el máximo acuerdo que se le puede pedir a otro método. El acuerdo de 0,04 que medí en la etapa A entre el modelo y el optimal matching simple (descartado) queda dentro del ruido: no permite afirmar que el método clásico «vea otra cosa». Lo mismo valdrá para el MCSA si se retoma.
3. **Lo que se midió después (7.1b):** el techo de ruido por muestreo, con tipologías de proporciones de tokens y del `pre`. Quedan sin medir si promediar las tres semillas estabiliza los tipos.

### 7.1b Techo de ruido por muestreo (tokens contra `pre`)

**Pregunta.** En 7.1 el ARI entre semillas dio 0,12 a 0,22 y yo sugerí que parte de eso era propio de cortar un continuo en k tipos. Para separarlo del efecto de la semilla, se armó el Ward de k = 8 sobre una mitad al azar de las personas del test (5 repeticiones con mitades distintas), se asignó a todas al centroide más cercano y se comparó entre mitades (`etapa_b techo`, `resultados_techo.md`). Las proporciones de tokens no dependen de ninguna semilla de entrenamiento, así que su ARI mide sólo el ruido de muestreo.

| comparación | ARI medio | mín a máx |
|---|---|---|
| tokens: mitad A contra mitad B (sólo muestreo) | 0,686 | 0,588 a 0,825 |
| `pre` semilla 0: mitad A contra B (sólo muestreo) | 0,448 | 0,395 a 0,525 |
| `pre`: semillas 0 y 1, misma mitad (sólo semilla) | 0,199 | 0,178 a 0,216 |
| `pre`: semillas 0 y 2, misma mitad (sólo semilla) | 0,118 | 0,092 a 0,136 |
| tokens contra `pre` semilla 0, misma mitad (otro espacio) | 0,028 | 0,018 a 0,049 |

**Lectura.**

1. **Cortar un continuo no explica la inestabilidad entre semillas.** El muestreo solo da ARI de 0,45 a 0,69, bastante más que el 0,12 a 0,20 entre semillas. Mi explicación de 7.1 («cortar el espacio en k tipos es sensible a movimientos chicos») era insuficiente: la mayor parte del desacuerdo entre semillas es propia de los vectores del modelo, que cambian en cada entrenamiento.
2. **El espacio `pre` es menos estable que el de tokens aun frente al muestreo** (0,45 contra 0,69), lo que es coherente con una geometría con menos estructura de grupos.
3. **Tokens y `pre` casi no coinciden** (0,03), por debajo incluso del acuerdo entre semillas del `pre` (0,12 a 0,20). Las dos tipologías organizan a la gente por criterios distintos, aunque ese contraste es de tamaño modesto frente al ruido semilla a semilla; sirve como descripción, no como prueba de que uno «vea más».
4. **Consecuencia para el MCSA y cualquier comparación con el modelo:** el techo que importa es el acuerdo entre semillas del propio modelo (0,12 a 0,20), no el de muestreo. Un método que coincida con el `pre` en ese rango no podría distinguirse de otra semilla del mismo modelo.

**Reservas.** Cinco repeticiones; Ward sobre la mitad de las personas, no sobre todas (el ARI entre semillas de 7.1 con todas las personas, 0,12 a 0,15 en `pre`, es coherente con el 0,12 a 0,20 de aquí); sólo k = 8; y para tokens se usó la representación de proporciones con sexo y edad que usó A3.


### 7.2 Qué encoder lee qué (controles de B1 y B3)

Se sondaron tres encoders con las mismas sondas (`resultados_extras.md`):

| encoder | rama (letra) | antigüedad (años) | cambios de empresa 60 m | huecos 60 m | hay_dif_36 |
|---|---|---|---|---|---|
| pre-entrenado | 0,856 | 0,906 | 0,548 | 0,492 | 0,613 |
| `pre_150ep_sinuv` (sin rama, provincia ni antigüedad de la empresa) | 0,422 | 0,927 | 0,510 | 0,634 | 0,719 |
| pesos sin entrenar | 0,890 | 0,306 | 0,206 | 0,126 | 0,216 |

1. **El pre-entrenamiento es el origen de la ventaja en dinámica.** Un encoder con los pesos al azar lee lo estático igual o mejor que el entrenado (rama 0,890, tamaño 0,861, salario 0,812: un promedio de embeddings aleatorios ya conserva qué tokens hubo), pero en dinámica queda al nivel de las proporciones (0,21 en cambios de empresa, 0,31 en antigüedad). Esto responde la objeción que quedó abierta en la primera versión de este informe («no sé cuánto aportan los 150 epochs»).
2. **No depende de los atributos de «último valor».** Sin rama, provincia ni antigüedad de la empresa, el modelo conserva la dinámica (0,510 en cambios de empresa; huecos e `hay_dif_36` incluso suben) y la letra de rama cae de 0,856 a 0,422, como debe ser. Esto responde la preocupación sobre la rama retroactiva y la comparabilidad: la ventaja del vector no viene de leer esos atributos. (Aun así, la rama sigue siendo un atributo de «último valor» que hay que tener presente al interpretar el modelo completo.)

### 7.3 Tiempo y orden (controles de B3)

Se repitieron las sondas con el modelo `pre` alimentado con historias alteradas: **fechas y edades permutadas** entre los eventos de la persona (mismo orden, sin saber cuándo pasó cada uno) y **orden permutado** (cada evento conserva su fecha pero cambia de lugar).

| entrada | cambios de empresa 60 m | meses desde cambio de empresa | meses desde cambio de tamaño | cambios de rama 60 m | huecos 60 m | hay_dif_36 |
|---|---|---|---|---|---|---|
| sin tocar | 0,548 | 0,492 | 0,546 | 0,495 | 0,492 | 0,613 |
| fechas permutadas | 0,547 | 0,488 | 0,546 | 0,501 | 0,441 | 0,615 |
| orden permutado | 0,545 | 0,542 | 0,639 | 0,529 | 0,402 | 0,603 |

**Lo que dicen.** Casi ninguna variable de dinámica cae. Las que más caen son los huecos (0,49 → 0,44 sin fechas y 0,40 sin orden), y algunas incluso suben con el orden permutado (meses desde el cambio de tamaño: 0,55 → 0,64).

**Corrección de lo que escribí antes.** Antes de estos controles sostuve que la ventaja del vector era información de *tiempo* aunque no necesariamente de *orden* (sección 7 de la versión previa de este informe, y mi lectura inicial de la etapa B). Los controles lo desmienten: la lectura de esas variables **no requiere ni fechas ni orden**. Los controles se corrieron sólo con la semilla 0; las sondas del `pre` sin tocar son estables entre semillas (7.1), pero no se repitió la permutación con las otras dos.

### 7.4 Línea de base con conteos absolutos

Las proporciones de tokens dividen por el total de tokens de la persona, así que borran cuánta historia hay y cuántos eventos de cada tipo ocurrieron. Se rehizo la comparación con una referencia que conserva esa información (`resultados_conteos.md`; $R^2$ en test, ridge entrenada en val):

| variable | proporciones | proporciones + tamaño | conteos | log(1+conteos) | log(1+conteos) + tamaño | `pre` |
|---|---|---|---|---|---|---|
| antigüedad en el empleo actual (2) | 0,309 | 0,503 | 0,424 | 0,505 | **0,644** | 0,567 |
| meses desde cambio de empresa | 0,261 | 0,385 | 0,316 | 0,397 | **0,498** | 0,492 |
| meses desde cambio de tamaño | 0,270 | 0,415 | 0,350 | 0,437 | 0,548 | 0,546 |
| meses desde cambio de rama | 0,272 | 0,426 | 0,342 | 0,434 | **0,537** | 0,505 |
| cambios de empresa 12 m | 0,071 | 0,081 | 0,065 | 0,109 | 0,112 | **0,195** |
| cambios de empresa 24 m | 0,139 | 0,181 | 0,119 | 0,223 | 0,249 | **0,365** |
| cambios de empresa 60 m | 0,220 | 0,320 | 0,225 | 0,440 | 0,484 | **0,548** |
| cambios de tamaño 60 m | 0,043 | 0,045 | 0,036 | 0,116 | 0,117 | **0,238** |
| cambios de rama 60 m | 0,171 | 0,272 | 0,160 | 0,420 | 0,451 | **0,495** |
| huecos 24 m | 0,054 | 0,108 | 0,057 | 0,077 | 0,116 | **0,284** |
| huecos 60 m | 0,130 | 0,232 | 0,167 | 0,231 | 0,294 | **0,492** |
| ¿hay dif. salarial 12 m? | 0,002 | 0,011 | 0,004 | 0,007 | 0,036 | **0,428** |
| ¿hay dif. salarial 36 m? | 0,071 | 0,316 | 0,137 | 0,204 | 0,466 | **0,613** |

(«tamaño de la historia» = log del total de eventos y número de tokens distintos. Las proporciones y los conteos no incluyen sexo ni edad en esta tabla; ignorar la fila de edad de `resultados_conteos.md`.)

**Lectura.**

1. **Una parte grande de la ventaja del vector era información de cantidad.** Con log(1+conteos) más el tamaño de la historia, los meses desde el último cambio (de empresa, tamaño y rama) y la antigüedad en el empleo actual igualan o superan al vector, y cambios de empresa y de rama a 60 meses quedan cerca (0,48 contra 0,55; 0,45 contra 0,50). La comparación contra las proporciones solas (secciones 5.3 y 7.2) sobreestimaba la ventaja.
2. **Lo que sigue por encima del conteo.** Los huecos (0,49 contra 0,29 a 60 meses; 0,28 contra 0,12 a 24 meses), los cambios en ventanas cortas (empresa a 12 meses 0,20 contra 0,11; tamaño a 60 meses 0,24 contra 0,12) y, sobre todo, la existencia de un cambio salarial reciente (0,43 contra 0,04 a 12 meses; 0,61 contra 0,47 a 36). Estas variables comparan momentos distintos de la historia (hubo un hueco, el salario cambió en el último año), cosa que un conteo de tokens no puede representar. Es la mejor evidencia que tenemos de que el vector guarda algo más que cantidades.
3. **Reservas.** (a) La línea de base tiene unas mil columnas y el vector bastante menos, lo que favorece a los conteos; la ventaja que queda es, en ese sentido, conservadora. (b) No se probó una línea de base con interacciones (por ejemplo, conteos por año), que podría achicar la brecha en huecos y salario. (c) Los controles de 7.3 muestran que esa ventaja restante tampoco necesita fechas exactas ni orden, así que no se debe leer como «el modelo usa la secuencia».

### 7.5 Trayectorias en el espacio (B2)

**Qué se hizo.** El mismo modelo pre-entrenado, con la historia de cada persona cortada en cuatro fechas: 2010-12, 2012-12, 2014-12 y 2015-12 (`fin_ventana` = 179, 203, 227 y 239). Sólo entran las personas con al menos 12 meses de historia en el corte, así que los cortes tempranos tienen menos gente: 16.185, 17.401, 18.296 y 18.337 personas con eventos. (La primera versión de este análisis decía «sin datos» porque exigía que todas las personas estuvieran en todos los cortes; se corrigió y hay una prueba.)

**Cuánto se mueve el vector.** Distancia coseno entre el vector del corte y el de 2015-12: mediana 0,080 (corte 2010-12), 0,034 (2012-12) y 0,005 (2014-12). Se mueve más cuanto más lejos está el corte, y menos cuanto más historia ya tenía la persona.

**Se mueve más quien cambió.** Entre el corte y 2015-12, las personas que cambiaron de empleador, de tamaño o de rama se movieron en el espacio de dos a cuatro veces más que quienes no cambiaron, en cada estrato de historia. Ejemplo, corte 2010-12 y cambio de empleador: 0,296 contra 0,098 con poca historia; 0,127 contra 0,050 con historia media; 0,070 contra 0,027 con mucha (la comparación por estratos controla que los cortes tempranos tienen menos información). Con la rama, los cocientes son de 3,1 (poca historia), 2,7 y 2,7. Que el espacio responda a los cambios de la manera esperada es una validación de sentido común, no un hallazgo sustantivo.

**Movilidad entre tipos.** Con un k-means de k = 8 sobre los vectores de 2015-12 (centrados y normalizados) y asignando los vectores de cada corte al centroide más cercano, la permanencia en el mismo tipo es 65 % a cinco años, 75 % a tres y 90 % a uno. Las matrices están en `resultados_cortes.md`; los flujos más grandes son entre pocos pares de tipos (por ejemplo, del tipo 0 al 2 con 16 % y del 6 al 2 con 19 % a cinco años).

**Advertencia.** Los tipos son los de una sola corrida, y 7.1 mostró que las tipologías del `pre` no se reproducen entre semillas. Estas matrices describen cómo se mueven las personas entre los tipos de *este* modelo; no son una medida de movilidad laboral. Rehacerlas con tipologías estables (por ejemplo, definidas sobre variables observables) es lo que haría de B2 un resultado utilizable.

### 7.6 Intervenciones sobre la historia (B3)

**Qué se hizo.** Se cambia una cosa de la historia de cada persona del test y se vuelve a correr el modelo (`resultados_intervenciones.md`): borrar los eventos del último año, que la última relación conserve la rama y el tamaño de la anterior, subir o bajar 10 puntos de percentil el salario de los eventos del último año, o atrasar todas las fechas dos años. Se mide el cambio del logit en los modelos ajustados. Es lo que el modelo hace con la historia modificada, no un efecto causal sobre las personas.

**Resultados** (cambio medio del logit; entre paréntesis, el porcentaje de personas cuyo logit sube):

| intervención | T1 | T3 |
|---|---|---|
| sin último año | +0,11 (87 %) | +0,22 (82 %) |
| sin cambios recientes | +0,08 (54 %) | −0,03 (40 %) |
| salario +10 puntos | −0,04 (24 %) | −0,05 (34 %) |
| salario −10 puntos | +0,08 (95 %) | +0,07 (73 %) |
| atrasar 2 años | +0,19 (94 %) | +0,21 (75 %) |

Para ubicar las magnitudes: el desvío del logit entre personas es 0,69 en T1 y 1,55 en T3. Lo que cambia la AUL es poco, salvo en un caso: **sin cambios recientes en T3 baja la AUL de 0,726 a 0,666**, porque ahí se destruye justo la información de cambios de rama sobre la que se apoya la tarea. En T1 el resto de las intervenciones deja la AUL a ±0,008 (0,613 → 0,605 a 0,616).

**Lo que se lee.**

1. **El signo es el esperable.** Subir el salario baja la probabilidad de salida del empleo (T1) y de cambio de rama (T3); bajarlo la sube. Es una lectura de sentido común que el modelo respeta, no una prueba de que «entienda» el salario.
2. **La magnitud es chica frente a la variación entre personas.** Un movimiento de 10 puntos en el salario del último año mueve el logit unos 0,04 a 0,08, una décima parte del desvío entre personas en T1.
3. **El modelo reacciona a las fechas.** Atrasar dos años sube el logit de salida en 0,19 (T1) con casi nada de cambio en la AUL. Una lectura posible, no verificada, es que el modelo asocia una historia que termina antes con más probabilidad de salida. Esto convive con 7.3: *leer* variables de dinámica desde el vector no necesita fechas, pero *predecir* sí responde a ellas.
4. **El salario se registra poco en el vector.** La sonda lineal implícita (cuánto cambia el percentil de salario de 2015 que predice el vector) da +2,2 puntos con `salario_arriba` y −2,6 con `salario_abajo` en T1, contra los ±10 aplicados. En el `pre` es todavía menor (±1,2). Una sonda lineal sobre un promedio puede subestimar un efecto que la atención sí represente (ver 5.3, punto 6), pero tampoco hay evidencia de lo contrario.
5. **Cambio de tipo.** En el espacio `pre`, las intervenciones hacen que cambie de tipo entre el 6 y el 18 % de las personas (28 % de las afectadas por `sin_cambios_recientes`). Se aplica la misma advertencia de 7.5 sobre la inestabilidad de los tipos.

---

## 8. La parte de tipologías (A3): qué se hizo y por qué se frenó

El plan era armar tipologías de trayectorias en tres espacios (el vector `pre`, las proporciones de tokens y el optimal matching clásico) y compararlas. Quedó incompleto, y lo que pasó es parte del resultado de la fase:

1. **El optimal matching de un canal se descartó.** Lo construí con 10 estados (tercil de salario × tamaño de empresa) y 120 meses. La comparación era injusta: el modelo ve rama, provincia, edad, antigüedad de la empresa y varias relaciones a la vez, y ese OM ve dos cosas. Los tipos que salían separaban sobre todo por cuándo había entrado la persona al panel. Por eso se borraron sus resultados.
2. **El análisis multicanal (MCSA) quedó diseñado pero no implementado**, y se suspendió por decisión del proyecto. La referencia es López-Pernas et al. (2024, `data/docs/978-3-031-54464-4_13.pdf`): ocho canales con costos aditivos que reflejan el input del transformer (salario, tamaño, rama, provincia, antigüedad de la empresa, edad, sexo y relaciones simultáneas). El diseño completo está en `plans/PLAN_fase_9.md`, con cuatro decisiones pendientes.
3. **La etapa B cambia lo que se podría esperar del MCSA.** La sección 7.1 mostró que las tipologías del modelo no se reproducen entre semillas (ARI 0,12 a 0,22). Eso impone un techo a cualquier comparación contra otro método: no se podría concluir que dos tipologías «ven cosas distintas» si su acuerdo no supera el que tiene el modelo consigo mismo. Antes de invertir las ~9 horas de cálculo, convendría medir el ARI entre semillas de las tipologías de proporciones de tokens y decidir si el objetivo es probar una hipótesis (modelo contra MCSA) o describir lo que capta el transformer.
4. **Dos limitaciones de los datos que afectan cualquier tipología:** (a) la **rama** es la «última rama disponible» de cada relación (retroactiva): puede incorporar información posterior al momento del evento y hay problemas de comparabilidad en el tiempo según el manual de la muestra; (b) la **antigüedad (ANT)** es el tramo del año de inicio de la *empresa*, no la antigüedad del trabajador. Ninguna de las dos es ruido para el transformer (las ve igual que cualquier otro atributo), pero un análisis de secuencias que las use como estado hereda esas limitaciones.

---

## 9. Qué se puede concluir y qué no

**Se puede concluir, con respaldo en los datos:**

1. **Las sondas del espacio pre-entrenado son estables.** Dan lo mismo en tres entrenamientos (rangos de 0,00 a 0,07), por lo que los resultados de lectura del vector no son accidentes de una semilla.
2. **El vector conserva lo que sabe un conteo de tokens** (rama, tamaño, salario, edad) y **la antigüedad en años mucho mejor** (0,906 contra 0,506 con proporciones; con conteos más tamaño de la historia la brecha casi se cierra: 0,73 contra 0,91 para años, 0,64 contra 0,57 para la antigüedad en el empleo actual).
3. **La ventaja en dinámica sobre las proporciones era, en su mayor parte, información de cantidad.** Una referencia que conserve los conteos y el largo de la historia iguala o supera al vector en los meses desde el último cambio y en la antigüedad en el empleo actual, y se acerca en los cambios a 60 meses.
4. **Hay un resto genuino:** huecos, cambios en ventanas cortas y, sobre todo, la existencia de cambios salariales recientes (0,43 contra 0,04) se leen del vector y no de los conteos.
5. **El origen es el pre-entrenamiento**: un encoder sin entrenar lee lo estático pero no la dinámica; y no depende de rama, provincia ni antigüedad de la empresa.
6. **Ni las fechas ni el orden** son necesarios para leer esas variables del vector (los controles de 7.3).
7. **Los ajustes finos reorientan el espacio:** T3 refuerza los cambios; T1, el nivel; ambos pierden direcciones que no necesitan (sección 5.3, puntos 4 y 5).
8. **El modelo reacciona a las modificaciones de la historia con el signo esperado**, pero con magnitudes chicas (7.6).

**No se puede concluir, y por qué:**

1. **Que el modelo use el orden de los eventos para estas variables.** Los controles dicen que no lo necesita para leerlas; queda sin probar si lo usa para predecir (T1 a T4) más allá de lo que aportan los conteos.
2. **Que el vector guarde «dinámica» en sentido fuerte.** Sólo el resto de 7.4 (huecos, cambio salarial reciente) apunta en esa dirección, y falta una referencia con conteos por año para saber si es genuino.
3. **Que existan tipologías estables de trayectorias en el espacio del modelo.** Con ARI de 0,12 a 0,22 entre semillas, frente a 0,45 de ruido por muestreo del propio `pre` (7.1b), y silueta de 0,03 a 0,05, no.
4. **Que el modelo no represente la evolución del salario.** Sólo que no se lee linealmente de un promedio, y que la sonda implícita responde con un cuarto del cambio aplicado (7.6).
5. **Que las comparaciones con el método clásico favorezcan o desfavorezcan al transformer:** la parte A3 no se hizo.

**Amenazas a la validez que conviene tener presentes:**

- **La referencia de las sondas importa más de lo que parecía.** La primera versión del informe usaba proporciones y concluía que el vector tenía información temporal; con conteos más tamaño de la historia, la mayoría de esa ventaja desaparece. Cualquier comparación futura debe usar la referencia más fuerte (7.4), no la más simple.
- **Comparaciones múltiples.** Son unos 25 objetivos por 12 espacios, con intervalos marginales. Las diferencias citadas son de decenas de puntos y con intervalos que no se tocan; los pares con una diferencia de uno o dos puntos no deben leerse como superioridad.
- **Las cohortes de T1 y T3 no son las mismas** (T3 es un subconjunto: quienes siguen en 2019). Las comparaciones contra `tokens` son siempre sobre la misma cohorte.
- **Las tablas de sondas de la etapa A guardadas** (`resultados_sondas.md`) tienen la columna «meses desde el último registro» con $R^2$ = 1,000: es un artefacto (la variable es constante) corregido en el código y todavía no regenerado. Las tablas de la etapa B y de conteos muestran `nan` y están bien.

---

## 10. Qué sigue

Prioridad según lo aprendido:

1. ~~Medir el techo de ruido de las tipologías de proporciones de tokens~~ (hecho, sección 7.1b). Queda probar si promediar semillas estabiliza los tipos.
2. **Línea de base con conteos por año** (como el paper original) para saber si el resto de 7.4 (huecos, cambio salarial) es genuino.
3. **Decidir sobre el MCSA** (sección 8) a la luz de 7.1. Si se retoma, hay cuatro decisiones pendientes en `plans/PLAN_fase_9.md`.
4. **Regenerar las sondas de la etapa A** para sacar el artefacto (unos 30 minutos) y agregar una sonda no lineal para la evolución del salario.
5. **Repetir B2 con tipos estables** para que la movilidad entre tipos sea utilizable.

Pendiente de orden: los resultados de la etapa B, la línea de base de conteos y sus correcciones (`etapa_b.py`, `espacio.py`, los `resultados_*.md` y dos figuras sin usar, `index_pre_k8.png` e `index_tokens_k8.png`) están **sin commitear**; falta además una prueba para `conteos`/`seccion_conteos`.
