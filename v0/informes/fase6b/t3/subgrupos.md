AUL por subgrupo (media entre semillas), etiqueta `y`.

| Modelo | mujeres (n=4,950, tasa 0.102) | varones (n=10,366, tasa 0.114) | 25-34 (n=5,794, tasa 0.161) | 35-44 (n=5,728, tasa 0.090) | 45-55 (n=3,794, tasa 0.062) |
|---|---|---|---|---|---|
| Transformer pre-entrenado (L2V) | 0.747 | 0.714 | 0.683 | 0.718 | 0.750 |
| Transformer sin pre-entrenar | 0.674 | 0.620 | 0.615 | 0.570 | 0.593 |
| GRU | 0.720 | 0.693 | 0.662 | 0.689 | 0.713 |
| FFNN | 0.708 | 0.678 | 0.647 | 0.662 | 0.675 |
| Regresión logística | 0.704 | 0.675 | 0.644 | 0.656 | 0.678 |
| Logística con estado actual | 0.687 | 0.640 | 0.608 | 0.618 | 0.623 |
| Tabla de vida (edad y sexo) | 0.632 | 0.606 | 0.548 | 0.538 | 0.542 |
