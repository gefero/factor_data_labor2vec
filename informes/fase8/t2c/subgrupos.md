auc_macro por subgrupo (media entre semillas).

| Modelo | mujeres (n=4,921) | varones (n=10,329) | 25-34 (n=5,775) | 35-44 (n=5,704) | 45-55 (n=3,771) |
|---|---|---|---|---|---|
| Transformer pre-entrenado (L2V) | 0.733 | 0.745 | 0.727 | 0.745 | 0.747 |
| Transformer sin pre-entrenar | 0.541 | 0.568 | 0.545 | 0.543 | 0.567 |
| GRU | 0.693 | 0.712 | 0.691 | 0.706 | 0.714 |
| FFNN | 0.633 | 0.653 | 0.625 | 0.639 | 0.666 |
| Regresión logística | 0.653 | 0.667 | 0.654 | 0.652 | 0.675 |
| Logística con estado actual | 0.635 | 0.643 | 0.620 | 0.642 | 0.638 |
| Tabla de vida (edad y sexo) | 0.529 | 0.547 | 0.530 | 0.524 | 0.526 |
