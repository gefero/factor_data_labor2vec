# Fase 9, etapa B: controles (generado por `analisis.etapa_b`)

## Tiempo contra orden

Sondas (A1) sobre el vector del pre-entrenado cuando las historias llegan al modelo con las **fechas y edades permutadas** entre los eventos de la persona (mismo orden, sin saber cuándo pasó cada uno) o con el **orden permutado** (cada evento conserva su fecha, pero cambia de lugar). Si una variable de dinámica cae sin fechas, el modelo la lee del tiempo; si cae con el orden permutado, depende del orden de los eventos.

| entrada | letra | tamaño | salario | antigüedad | antigüedad (años) | dinámica: cambios_empresa_60 | dinámica: meses_desde_cambio_empresa | dinámica: meses_desde_cambio_tamano | dinámica: cambios_tamano_60 | dinámica: cambios_rama_60 | dinámica: huecos_60 | dinámica: hay_dif_36 | dinámica: dif_salario_36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre (sin tocar) | 0.856 | 0.874 | 0.835 | 0.843 | 0.906 | 0.548 | 0.492 | 0.546 | 0.238 | 0.495 | 0.492 | 0.613 | 0.039 |
| fechas permutadas | 0.858 | 0.869 | 0.831 | 0.858 | 0.914 | 0.547 | 0.488 | 0.546 | 0.225 | 0.501 | 0.441 | 0.615 | 0.037 |
| orden permutado | 0.861 | 0.895 | 0.841 | 0.859 | 0.915 | 0.545 | 0.542 | 0.639 | 0.252 | 0.529 | 0.402 | 0.603 | 0.046 |
