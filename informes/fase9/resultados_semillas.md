# Fase 9, etapa B: semillas (generado por `analisis.etapa_b`)

## B1. Estabilidad entre semillas

Cada semilla es un entrenamiento distinto (el pre-entrenado con otro azar; en `t1` y `t3`, sólo el ajuste fino con otro azar, a partir del mismo pre-entrenado). Sobre las mismas personas del test se mide: **CKA lineal** (similitud global de los espacios; 1 = idénticos salvo rotación o escala); **vecinos** (fracción de los 10 vecinos más cercanos que coinciden; entre paréntesis, lo esperable al azar); **ARI** de las tipologías de k = 8 tipos armadas con cada espacio.

### pre

| semillas | CKA | vecinos (azar) | ARI de tipologías |
|---|---|---|---|
| 0 y 1 | 0.504 | 0.447 (0.0025) | 0.153 |
| 0 y 2 | 0.446 | 0.408 (0.0025) | 0.118 |
| 1 y 2 | 0.635 | 0.480 (0.0025) | 0.142 |

### t1

| semillas | CKA | vecinos (azar) | ARI de tipologías |
|---|---|---|---|
| 0 y 1 | 0.687 | 0.117 (0.0025) | 0.196 |
| 0 y 2 | 0.738 | 0.163 (0.0025) | 0.222 |
| 1 y 2 | 0.840 | 0.184 (0.0025) | 0.212 |

### t3

| semillas | CKA | vecinos (azar) | ARI de tipologías |
|---|---|---|---|
| 0 y 1 | 0.739 | 0.169 (0.0025) | 0.221 |
| 0 y 2 | 0.694 | 0.215 (0.0025) | 0.180 |
| 1 y 2 | 0.625 | 0.146 (0.0025) | 0.173 |

### Sondas del vector por semilla (exactitud balanceada para las categóricas, R² para las continuas)

| espacio | letra | tamaño | salario | antigüedad | antigüedad (años) | dinámica: cambios_empresa_60 | dinámica: meses_desde_cambio_empresa | dinámica: meses_desde_cambio_tamano | dinámica: cambios_tamano_60 | dinámica: cambios_rama_60 | dinámica: huecos_60 | dinámica: hay_dif_36 | dinámica: dif_salario_36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre semilla 0 | 0.856 | 0.874 | 0.835 | 0.843 | 0.906 | 0.548 | 0.492 | 0.546 | 0.238 | 0.495 | 0.492 | 0.613 | 0.039 |
| pre semilla 1 | 0.856 | 0.875 | 0.834 | 0.843 | 0.903 | 0.553 | 0.500 | 0.527 | 0.195 | 0.510 | 0.522 | 0.647 | 0.040 |
| pre semilla 2 | 0.858 | 0.871 | 0.825 | 0.847 | 0.906 | 0.569 | 0.517 | 0.572 | 0.266 | 0.556 | 0.508 | 0.619 | 0.043 |
| t1 semilla 0 | 0.844 | 0.908 | 0.887 | 0.842 | 0.887 | 0.549 | 0.511 | 0.544 | 0.213 | 0.486 | 0.500 | 0.539 | 0.116 |
| t1 semilla 1 | 0.872 | 0.929 | 0.899 | 0.832 | 0.883 | 0.561 | 0.552 | 0.599 | 0.236 | 0.486 | 0.497 | 0.565 | 0.137 |
| t1 semilla 2 | 0.896 | 0.922 | 0.900 | 0.836 | 0.884 | 0.574 | 0.558 | 0.593 | 0.248 | 0.516 | 0.501 | 0.565 | 0.128 |
| t3 semilla 0 | 0.901 | 0.936 | 0.897 | 0.834 | 0.874 | 0.625 | 0.682 | 0.634 | 0.229 | 0.612 | 0.422 | 0.535 | 0.092 |
| t3 semilla 1 | 0.886 | 0.932 | 0.893 | 0.815 | 0.864 | 0.600 | 0.646 | 0.627 | 0.225 | 0.582 | 0.399 | 0.547 | 0.087 |
| t3 semilla 2 | 0.932 | 0.936 | 0.887 | 0.818 | 0.864 | 0.602 | 0.635 | 0.597 | 0.196 | 0.609 | 0.402 | 0.515 | 0.076 |

Rango de las sondas en `pre` (máx − mín entre semillas): letra 0.002; tamaño 0.004; salario 0.011; antigüedad 0.004; antigüedad (años) 0.004; dinámica: cambios_empresa_60 0.021; dinámica: meses_desde_cambio_empresa 0.025; dinámica: meses_desde_cambio_tamano 0.044; dinámica: cambios_tamano_60 0.071; dinámica: cambios_rama_60 0.061; dinámica: huecos_60 0.030; dinámica: hay_dif_36 0.034; dinámica: dif_salario_36 0.004

Rango de las sondas en `t1` (máx − mín entre semillas): letra 0.052; tamaño 0.021; salario 0.012; antigüedad 0.009; antigüedad (años) 0.004; dinámica: cambios_empresa_60 0.025; dinámica: meses_desde_cambio_empresa 0.047; dinámica: meses_desde_cambio_tamano 0.055; dinámica: cambios_tamano_60 0.036; dinámica: cambios_rama_60 0.030; dinámica: huecos_60 0.003; dinámica: hay_dif_36 0.026; dinámica: dif_salario_36 0.020

Rango de las sondas en `t3` (máx − mín entre semillas): letra 0.046; tamaño 0.005; salario 0.010; antigüedad 0.019; antigüedad (años) 0.010; dinámica: cambios_empresa_60 0.025; dinámica: meses_desde_cambio_empresa 0.047; dinámica: meses_desde_cambio_tamano 0.037; dinámica: cambios_tamano_60 0.033; dinámica: cambios_rama_60 0.029; dinámica: huecos_60 0.023; dinámica: hay_dif_36 0.032; dinámica: dif_salario_36 0.015

