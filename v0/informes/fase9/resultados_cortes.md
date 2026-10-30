# Fase 9, etapa B: cortes (generado por `analisis.etapa_b`)

## B2. Trayectorias en el espacio

Personas del test con eventos en cada corte: 2010-12: 16,185; 2012-12: 17,401; 2014-12: 18,296; 2015-12: 18,337.

El mismo modelo pre-entrenado, con la historia de cada persona cortada en distintos meses (sólo ve lo anterior al corte). Sólo se cuentan las personas con al menos 12 meses de historia en el corte. Los tipos son los de un k-means de k = 8 sobre los vectores de 2015-12 (centrados y normalizados); los vectores de los cortes anteriores se asignan al centroide más cercano con el mismo centrado. Los cortes más tempranos tienen menos historia, así que parte del movimiento es sólo falta de información: por eso se compara por estratos de historia.

### Corte 2010-12 contra 2015-12 (15,555 personas con historia)

Distancia coseno entre el vector del corte y el de 2015-12: mediana 0.080, p25 0.031, p75 0.213.

| cambió entre el corte y 2015-12 | estrato de historia en el corte | n (cambió) | n (no cambió) | mediana (cambió) | mediana (no cambió) | diferencia |
|---|---|---|---|---|---|---|
| empleador | poca historia | 2,899 | 2,246 | 0.296 | 0.098 | +0.198 |
| empleador | media | 2,390 | 2,761 | 0.127 | 0.050 | +0.077 |
| empleador | mucha | 1,824 | 3,435 | 0.070 | 0.027 | +0.043 |
| tamaño | poca historia | 2,908 | 2,237 | 0.294 | 0.098 | +0.196 |
| tamaño | media | 2,371 | 2,780 | 0.127 | 0.051 | +0.076 |
| tamaño | mucha | 1,880 | 3,379 | 0.066 | 0.027 | +0.038 |
| rama | poca historia | 2,513 | 2,632 | 0.326 | 0.106 | +0.220 |
| rama | media | 1,877 | 3,274 | 0.147 | 0.055 | +0.092 |
| rama | mucha | 1,367 | 3,892 | 0.078 | 0.029 | +0.050 |

Transición entre tipos, de 2010-12 (filas) a 2015-12 (columnas); proporción de cada fila. Permanencia (diagonal ponderada): 0.653.

| tipo | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.57 | 0.08 | 0.16 | 0.01 | 0.10 | 0.03 | 0.04 | 0.00 |
| 1 | 0.05 | 0.71 | 0.03 | 0.10 | 0.01 | 0.06 | 0.03 | 0.01 |
| 2 | 0.08 | 0.02 | 0.70 | 0.00 | 0.02 | 0.11 | 0.06 | 0.01 |
| 3 | 0.01 | 0.16 | 0.00 | 0.71 | 0.08 | 0.02 | 0.01 | 0.01 |
| 4 | 0.08 | 0.03 | 0.05 | 0.04 | 0.66 | 0.07 | 0.05 | 0.01 |
| 5 | 0.01 | 0.03 | 0.06 | 0.02 | 0.12 | 0.75 | 0.01 | 0.00 |
| 6 | 0.08 | 0.13 | 0.19 | 0.01 | 0.06 | 0.01 | 0.51 | 0.01 |
| 7 | 0.03 | 0.10 | 0.10 | 0.02 | 0.05 | 0.01 | 0.08 | 0.61 |

### Corte 2012-12 contra 2015-12 (16,920 personas con historia)

Distancia coseno entre el vector del corte y el de 2015-12: mediana 0.034, p25 0.013, p75 0.095.

| cambió entre el corte y 2015-12 | estrato de historia en el corte | n (cambió) | n (no cambió) | mediana (cambió) | mediana (no cambió) | diferencia |
|---|---|---|---|---|---|---|
| empleador | poca historia | 2,186 | 3,433 | 0.170 | 0.054 | +0.115 |
| empleador | media | 1,783 | 3,812 | 0.060 | 0.024 | +0.037 |
| empleador | mucha | 1,309 | 4,397 | 0.033 | 0.012 | +0.021 |
| tamaño | poca historia | 2,159 | 3,460 | 0.170 | 0.055 | +0.115 |
| tamaño | media | 1,749 | 3,846 | 0.059 | 0.024 | +0.035 |
| tamaño | mucha | 1,345 | 4,361 | 0.030 | 0.012 | +0.019 |
| rama | poca historia | 1,821 | 3,798 | 0.187 | 0.058 | +0.129 |
| rama | media | 1,318 | 4,277 | 0.069 | 0.025 | +0.043 |
| rama | mucha | 897 | 4,809 | 0.037 | 0.012 | +0.025 |

Transición entre tipos, de 2012-12 (filas) a 2015-12 (columnas); proporción de cada fila. Permanencia (diagonal ponderada): 0.753.

| tipo | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.69 | 0.06 | 0.12 | 0.01 | 0.07 | 0.02 | 0.03 | 0.00 |
| 1 | 0.03 | 0.79 | 0.02 | 0.07 | 0.00 | 0.04 | 0.03 | 0.01 |
| 2 | 0.06 | 0.02 | 0.78 | 0.00 | 0.01 | 0.08 | 0.05 | 0.01 |
| 3 | 0.01 | 0.11 | 0.00 | 0.78 | 0.07 | 0.02 | 0.01 | 0.01 |
| 4 | 0.06 | 0.02 | 0.02 | 0.03 | 0.77 | 0.06 | 0.04 | 0.01 |
| 5 | 0.00 | 0.02 | 0.05 | 0.01 | 0.09 | 0.82 | 0.00 | 0.00 |
| 6 | 0.06 | 0.09 | 0.14 | 0.01 | 0.04 | 0.01 | 0.65 | 0.01 |
| 7 | 0.02 | 0.07 | 0.07 | 0.01 | 0.03 | 0.01 | 0.05 | 0.75 |

### Corte 2014-12 contra 2015-12 (17,902 personas con historia)

Distancia coseno entre el vector del corte y el de 2015-12: mediana 0.005, p25 0.002, p75 0.015.

| cambió entre el corte y 2015-12 | estrato de historia en el corte | n (cambió) | n (no cambió) | mediana (cambió) | mediana (no cambió) | diferencia |
|---|---|---|---|---|---|---|
| empleador | poca historia | 685 | 5,236 | 0.036 | 0.012 | +0.024 |
| empleador | media | 581 | 5,404 | 0.012 | 0.004 | +0.008 |
| empleador | mucha | 392 | 5,604 | 0.006 | 0.002 | +0.004 |
| tamaño | poca historia | 589 | 5,332 | 0.041 | 0.012 | +0.029 |
| tamaño | media | 490 | 5,495 | 0.015 | 0.004 | +0.011 |
| tamaño | mucha | 381 | 5,615 | 0.007 | 0.002 | +0.005 |
| rama | poca historia | 479 | 5,442 | 0.039 | 0.013 | +0.026 |
| rama | media | 375 | 5,610 | 0.014 | 0.004 | +0.010 |
| rama | mucha | 228 | 5,768 | 0.006 | 0.002 | +0.004 |

Transición entre tipos, de 2014-12 (filas) a 2015-12 (columnas); proporción de cada fila. Permanencia (diagonal ponderada): 0.895.

| tipo | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.85 | 0.02 | 0.05 | 0.00 | 0.04 | 0.00 | 0.02 | 0.00 |
| 1 | 0.01 | 0.91 | 0.01 | 0.03 | 0.00 | 0.01 | 0.01 | 0.01 |
| 2 | 0.03 | 0.01 | 0.90 | 0.00 | 0.00 | 0.03 | 0.03 | 0.00 |
| 3 | 0.01 | 0.05 | 0.00 | 0.90 | 0.03 | 0.01 | 0.00 | 0.00 |
| 4 | 0.03 | 0.00 | 0.00 | 0.01 | 0.91 | 0.02 | 0.01 | 0.01 |
| 5 | 0.00 | 0.01 | 0.02 | 0.00 | 0.04 | 0.93 | 0.00 | 0.00 |
| 6 | 0.03 | 0.03 | 0.06 | 0.00 | 0.02 | 0.00 | 0.86 | 0.01 |
| 7 | 0.01 | 0.02 | 0.03 | 0.00 | 0.02 | 0.00 | 0.01 | 0.91 |

