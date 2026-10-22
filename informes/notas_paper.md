# Notas para el paper: qué se puede decir según lo que sabemos

Actualizado el 2026-10-22, **con los resultados de la fase 8** (controles temporales, variantes de T2-T4 y ablación sin atributos de último valor). Es un punteo de trabajo, no un borrador. Cifras de `informes/fase4.md`, `fase5.md`, `fase6.md`, `fase6b.md`, `fase7.md` y `fase8.md`.

## Planteo posible

«Adaptamos life2vec a registros administrativos pobres en información de un país de ingreso medio. La representación aprende estructura real del mercado laboral, pero la ventaja predictiva sobre modelos simples depende de la tarea.» Un resultado mixto, bien documentado, tiene valor. Contribución de recurso: implementación abierta en castellano, con las diferencias respecto del código original documentadas.

## Se puede afirmar

- **El pre-entrenamiento funciona y es estable.** Perplejidad MLM 4,73 contra 81,4 de la línea de base de frecuencias (17 veces menor); AUC de SOP 0,865. Dispersión entre 3 semillas ~1,5 % en perplejidad. 150 épocas mejoran 26 % la perplejidad respecto de 30.
- **El espacio de conceptos tiene estructura real y reproducible.**
  - Los percentiles de salario forman una curva ordenada (vecinos a ≤ 5 percentiles: 1,00 contra 0,10 del azar).
  - Años de nacimiento ordenados a nivel local.
  - Ramas vecinas comparten sector (0,41 contra 0,22) y provincias vecinas comparten región (0,49 contra 0,15).
  - Distancias reproducibles entre semillas (Spearman 0,72), mucho más en salarios y años (0,74-0,86) que en ramas finas (0,37-0,40).
- **Pre-entrenar ayuda en T1, T3 y T4** (frente al mismo transformer sin pre-entrenar, no frente a métodos simples): de +0,06 a +0,09 sobre el mismo transformer sin pre-entrenar, también sin los atributos de último valor (T3 0,661 contra 0,589; T4 0,749 contra 0,673; T1 0,602 contra 0,547).
- **La ventaja sobre modelos simples depende de la tarea y de qué se le da al modelo simple.**
  - Contra métodos que solo ven proporciones de tokens: T1 y T2 empate (T1 0,619 contra 0,617; T2 QWK 0,830 contra 0,822 de repetir el quintil de 2015); T3 gana (AUL 0,725 contra 0,702 GRU y 0,687 FFNN); T4 gana (AUC macro 0,772 contra 0,746 GRU y 0,691 logística). Sin los atributos de último valor la ventaja de T3 y T4 sigue (T3 0,661 contra 0,624; T4 0,749 contra 0,699).
  - **Contra métodos simples con variables de tiempo resumidas** (fase 8b): T1 pierde (0,619 contra 0,634); T3 empata (0,725 contra 0,731); T2 sin cambios; **T4 gana pero más chico** (0,772 contra 0,741, y 0,749 contra 0,738 sin último valor). Es el resultado a reportar.
  - Variantes: gana en T2c (baja/igual/sube de quintil) y T4c (cambia el tamaño) incluso contra los simples con tiempo; empata en T3d.
  - Patrón: gana en preguntas de «se movió algo» sobre tamaño y salario; no en niveles ni en rama.
- **No hay indicios de que la ventaja de T3 o T4 sea por fuga de información.** Al sacar rama, provincia y antigüedad el transformer no cae más que los métodos simples (T3 −0,064 contra −0,061 a −0,105; T4 −0,023 contra −0,018 a −0,050), y en T3 cae igual en el grupo que no puede tener fuga (0,067 contra 0,063).
- **El espacio de personas se organiza por quiénes son, no por lo que les pasa.** Salario, rama, tamaño, edad y antigüedad ordenan el mapa; la salida del empleo casi no. El ajuste fino agrega edad y antigüedad, que el modelo sin ajustar casi no tenía.
- **El modelo se apoya sobre todo en la historia reciente.** Los últimos 24 meses pesan ~2 veces lo que les toca por cantidad de tokens; lo de hace más de 5 años, ~0,55.

## Se puede decir con cuidado

- **El tiempo.** Con la línea de base de la fase 8b, buena parte de lo que el transformer saca del tiempo en T1 y T3 son magnitudes simples que un modelo lineal usa igual. Además, el orden y el momento de los eventos importan en T4 (reajustando sin tiempo: 0,772 a 0,738, por debajo del GRU) pero poco en T3 (0,725 a 0,710, sigue sobre el GRU). En T1 y T2 sólo se midió sin reajustar (−0,03 a −0,07), que exagera la dependencia. Qué parte de la ventaja que queda en T4 es secuencia propiamente dicha no se midió.
- **La caída al sacar atributos de último valor** (T3 −0,06 a −0,10) es una cota superior de la fuga: la rama de 2015 es información legítima para predecir si cambia. El análisis por continuidad intenta separar los casos, pero el grupo sin fuga posible es chico (~480-730 personas en test, 3,8 % de la cohorte): sin conclusión limpia.
- **El efecto del pre-entrenamiento en T2.** Con la corrida de scratch que convergió (sin atributos de último valor), el modelo sin pre-entrenar iguala al pre-entrenado (0,818 contra 0,822). La cifra original (0,745) era una corrida que no convergió en 35 épocas. En T2c el sin pre-entrenar colapsó. Decir: «el pre-entrenamiento no ayuda de forma demostrable en T2».
- **Que vale para el mercado laboral argentino.** Es una muestra del 3 % de empleo asalariado registrado privado, con ruido de ±3 % en los salarios por anonimización.

## Evitar

- Comparar valores con el paper original (otros datos y otras tareas): sólo sirve el orden entre modelos.
- TCAV como resultado: quedó degenerado (puntajes 0 o 1, signos que cambian con la capa). Mencionarlo, a lo sumo, como limitación.
- Lecturas causales o uso para decisiones sobre individuos (el paper original restringe ese uso y pide auditoría de equidad).
- Decir que «el modelo entiende el mercado laboral» en general: lo mostrado es estructura de conceptos y rendimiento en cuatro tareas.
- Citar la ventaja de pre-entrenar en T2 o T2c como efecto limpio (ver arriba).
- Decir que el transformer «gana» en T1 o T3 sin aclarar contra qué línea de base (con variables de tiempo, T1 lo pierde y T3 empata).
- Interpretar las filas de continuidad con IC que incluyen el 0 o con menos de ~500 casos.

## Falta antes de escribirlo

- **Variación del pre-entrenamiento en las tareas.** Todas usan la misma corrida pre-entrenada (semilla 0); las 3 semillas son de ajuste fino. El modelo sin último valor también es de una semilla.
- **Más semillas de las líneas de base secuenciales** (sin pre-entrenar, GRU, logística): una sola.
- **Una línea de base tabular con más y mejores variables de tiempo y ajuste de hiperparámetros** (la de la fase 8b es una elección entre muchas).
- **Validación temporal:** hay un solo corte (2015); sólo T1 se probó contra 2021.
- **Lectura de equidad:** hay subgrupos por sexo y edad, pero no un análisis específico.
- **Comparación del espacio de conceptos con el del paper original** (pendiente de definir si contra lo que reporta o reproduciendo sus análisis).
