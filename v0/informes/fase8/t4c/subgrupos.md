AUL por subgrupo (media entre semillas), etiqueta `y`.

| Modelo | mujeres (n=4,951, tasa 0.167) | varones (n=10,366, tasa 0.197) | 25-34 (n=5,794, tasa 0.235) | 35-44 (n=5,728, tasa 0.169) | 45-55 (n=3,795, tasa 0.142) |
|---|---|---|---|---|---|
| Transformer pre-entrenado (L2V) | 0.706 | 0.700 | 0.681 | 0.698 | 0.731 |
| Transformer sin pre-entrenar | 0.655 | 0.657 | 0.635 | 0.650 | 0.672 |
| GRU | 0.680 | 0.681 | 0.657 | 0.684 | 0.708 |
| FFNN | 0.655 | 0.655 | 0.638 | 0.650 | 0.657 |
| Regresión logística | 0.658 | 0.655 | 0.638 | 0.649 | 0.662 |
| Logística con estado actual | 0.635 | 0.629 | 0.613 | 0.625 | 0.624 |
| Tabla de vida (edad y sexo) | 0.582 | 0.560 | 0.535 | 0.544 | 0.529 |
