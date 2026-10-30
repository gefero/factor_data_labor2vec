# Fase 9, etapa B: extras (generado por `analisis.etapa_b`)

## Controles del encoder

Mismas sondas (A1) con tres encoders: el pre-entrenado, el pre-entrenado **sin rama, provincia ni antigüedad de la empresa** (`pre_150ep_sinuv`, que ve documentos sin esos atributos) y uno con **pesos sin entrenar** (el mismo diseño, sin ningún aprendizaje). Lo primero separa el efecto de los atributos de «último valor»; lo segundo, cuánto aporta el pre-entrenamiento.

| encoder | letra | tamaño | salario | antigüedad | antigüedad (años) | dinámica: cambios_empresa_60 | dinámica: meses_desde_cambio_empresa | dinámica: meses_desde_cambio_tamano | dinámica: cambios_tamano_60 | dinámica: cambios_rama_60 | dinámica: huecos_60 | dinámica: hay_dif_36 | dinámica: dif_salario_36 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| pre-entrenado | 0.856 | 0.874 | 0.835 | 0.843 | 0.906 | 0.548 | 0.492 | 0.546 | 0.238 | 0.495 | 0.492 | 0.613 | 0.039 |
| sin «último valor» | 0.422 | 0.884 | 0.859 | 0.869 | 0.927 | 0.510 | 0.467 | 0.531 | 0.224 | 0.404 | 0.634 | 0.719 | 0.102 |
| pesos sin entrenar | 0.890 | 0.861 | 0.812 | 0.521 | 0.306 | 0.206 | 0.250 | 0.267 | 0.048 | 0.155 | 0.126 | 0.216 | 0.015 |
