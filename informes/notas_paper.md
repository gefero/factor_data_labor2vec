# Notas para el paper: qué se puede decir según lo que sabemos

Anotado el 2026-10-21, **antes de tener los resultados de la fase 8** (controles temporales, ablación sin atributos de último valor y variantes de T2-T4). Es un punteo de trabajo, no un borrador. Cifras de `informes/fase4.md`, `fase5.md`, `fase6.md`, `fase6b.md` y `fase7.md`.

## Planteo posible

«Adaptamos life2vec a registros administrativos pobres en información de un país de ingreso medio. La representación aprende estructura real del mercado laboral, pero la ventaja predictiva sobre modelos simples depende de la tarea.» Un resultado mixto, bien documentado, tiene valor. Contribución de recurso: implementación abierta en castellano, con las diferencias respecto del código original documentadas.

## Se puede afirmar

- **El pre-entrenamiento funciona y es estable.** Perplejidad MLM 4,73 contra 81,4 de la línea de base de frecuencias (17 veces menor); AUC de SOP 0,865. Dispersión entre 3 semillas ~1,5 % en perplejidad. 150 épocas mejoran 26 % la perplejidad respecto de 30.
- **El espacio de conceptos tiene estructura real y reproducible.**
  - Los percentiles de salario forman una curva ordenada (vecinos a ≤ 5 percentiles: 1,00 contra 0,10 del azar).
  - Años de nacimiento ordenados a nivel local.
  - Ramas vecinas comparten sector (0,41 contra 0,22) y provincias vecinas comparten región (0,49 contra 0,15).
  - Distancias reproducibles entre semillas (Spearman 0,72), mucho más en salarios y años (0,74-0,86) que en ramas finas (0,37-0,40).
- **Pre-entrenar ayuda en las cuatro tareas:** de +0,06 a +0,09 sobre el mismo transformer sin pre-entrenar.
- **La ventaja sobre modelos simples depende de la tarea.**
  - T1 (salida del empleo registrado) y T2 (quintil salarial): empate. T1 0,619 contra 0,617 de la logística; T2 QWK 0,830 contra 0,822 de repetir el quintil de 2015.
  - T3 (cambio de rama): gana claro, AUL 0,725 contra 0,702 (GRU) y 0,687 (FFNN).
  - T4 (cambio de tamaño de empresa): gana claro, AUC macro 0,772 contra 0,746 (GRU) y 0,691 (logística).
- **El espacio de personas se organiza por quiénes son, no por lo que les pasa.** Salario, rama, tamaño, edad y antigüedad ordenan el mapa; la salida del empleo casi no. El ajuste fino agrega edad y antigüedad, que el modelo sin ajustar casi no tenía.
- **El modelo se apoya sobre todo en la historia reciente.** Los últimos 24 meses pesan ~2 veces lo que les toca por cantidad de tokens; lo de hace más de 5 años, ~0,55.

## Se puede decir con cuidado

- **Que T3 y T4 ganan por usar el orden y el momento de los eventos.** Hipótesis, no hallazgo. El saliency no encontró evidencia (detector de cambios de empresa burdo); lo prueba o lo descarta el control temporal de la fase 8.
- **Que T3 no está inflado por fuga de información.** Rama, provincia y antigüedad son atributos de «último valor» y pueden traer datos posteriores al corte; el token de antigüedad pesó 1,4 veces lo esperado. La ablación lo acota, pero el 96,2 % de la cohorte tiene la relación principal vigente después del corte, así que el grupo sin fuga posible es chico y particular: no se va a poder separar limpiamente fuga de información legítima.
- **Que vale para el mercado laboral argentino.** Es una muestra del 3 % de empleo asalariado registrado privado, con ruido de ±3 % en los salarios por anonimización.

## Evitar

- Comparar valores con el paper original (otros datos y otras tareas): sólo sirve el orden entre modelos.
- TCAV como resultado: quedó degenerado (puntajes 0 o 1, signos que cambian con la capa). Mencionarlo, a lo sumo, como limitación.
- Lecturas causales o uso para decisiones sobre individuos (el paper original restringe ese uso y pide auditoría de equidad).
- Decir que «el modelo entiende el mercado laboral» en general: lo mostrado es estructura de conceptos y rendimiento en cuatro tareas.

## Falta antes de escribirlo

- **Variación del pre-entrenamiento en las tareas.** Todas usan la misma corrida pre-entrenada (semilla 0); las 3 semillas son de ajuste fino.
- **La fase 8 completa:** controles temporales, ablación sin atributos de último valor y variantes de T2-T4.
- **Validación temporal:** hay un solo corte (2015); sólo T1 se probó contra 2021.
- **Lectura de equidad:** hay subgrupos por sexo y edad, pero no un análisis específico.
