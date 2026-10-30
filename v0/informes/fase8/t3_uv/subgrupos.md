AUL por subgrupo (media entre semillas), etiqueta `y`.

| Modelo | mujeres (n=4,950, tasa 0.102) | varones (n=10,366, tasa 0.114) | 25-34 (n=5,794, tasa 0.161) | 35-44 (n=5,728, tasa 0.090) | 45-55 (n=3,794, tasa 0.062) |
|---|---|---|---|---|---|
| Transformer pre-entrenado (L2V) | 0.670 | 0.656 | 0.628 | 0.624 | 0.671 |
| Transformer sin pre-entrenar | 0.617 | 0.576 | 0.535 | 0.487 | 0.521 |
| GRU | 0.603 | 0.594 | 0.545 | 0.513 | 0.564 |
| FFNN | 0.638 | 0.616 | 0.568 | 0.553 | 0.575 |
| Regresión logística | 0.634 | 0.618 | 0.565 | 0.549 | 0.587 |
| Logística con estado actual | 0.687 | 0.640 | 0.608 | 0.618 | 0.623 |
| Tabla de vida (edad y sexo) | 0.632 | 0.606 | 0.548 | 0.538 | 0.542 |
