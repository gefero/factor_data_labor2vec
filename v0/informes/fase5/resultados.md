# Resultados del análisis de conceptos (generado por `python -m labor2vec.analisis.conceptos`)

Tokens analizados: 509 (sin los 10 especiales). Nulos: 1000 permutaciones de etiquetas por medida; 5000 iteraciones en el test de permutación entre modelos. Modelos: `s0`, `s1`, `s2` = 150 épocas con las semillas 0, 1 y 2; `ep30` = 30 épocas (semilla 0).

## Orden de las variables ordinales

| Medida | s0 | s1 | s2 | ep30 |
|---|---|---|---|---|
| Salario (percentil 1-100): \|Spearman\| con la 1ª componente principal (p) | 0.905 (0.0010) | 0.920 (0.0010) | 0.930 (0.0010) | 0.799 (0.0010) |
| Salario: proporción de los 4 vecinos a ≤ 5 percentiles (nulo) | 1.000 (0.098) | 1.000 (0.098) | 1.000 (0.098) | 1.000 (0.098) |
| Salario: Spearman entre la distancia coseno de cada par y \|Δpercentil\| (nulo) | 0.841 (0.002) | 0.832 (0.001) | 0.841 (0.001) | 0.830 (0.001) |
| Año de nacimiento: \|Spearman\| con la 1ª componente (p) | 0.071 (0.5534) | 0.241 (0.0310) | 0.255 (0.0190) | 0.713 (0.0010) |
| Año de nacimiento: proporción de los 4 vecinos a ≤ 3 años (nulo) | 0.958 (0.077) | 0.951 (0.077) | 0.951 (0.077) | 0.766 (0.077) |
| Año de nacimiento: Spearman entre la distancia coseno de cada par y \|Δaño\| (nulo) | 0.184 (0.003) | 0.178 (0.003) | 0.226 (0.003) | 0.750 (-0.000) |
| Tamaño de empresa (4 niveles): Spearman distancia contra \|Δnivel\| | 0.83 | 0.83 | 0.83 | 0.83 |
| Antigüedad de la empresa (4 niveles): idem | 0.77 | 0.62 | 0.93 | 0.93 |

## Pureza de vecindades

| Medida | s0 | s1 | s2 | ep30 |
|---|---|---|---|---|
| Ramas: pureza de los 5 vecinos respecto de la letra (nulo; p) | 0.406 (0.219; 0.0010) | 0.391 (0.219; 0.0010) | 0.387 (0.218; 0.0010) | 0.406 (0.218; 0.0010) |
| Ramas: pureza respecto del r32 (nulo; p) | 0.143 (0.031; 0.0010) | 0.123 (0.030; 0.0010) | 0.138 (0.030; 0.0010) | 0.135 (0.030; 0.0010) |
| Provincias: pureza de los 3 vecinos respecto de la región (nulo; p) | 0.493 (0.151; 0.0010) | 0.467 (0.153; 0.0010) | 0.440 (0.151; 0.0010) | 0.427 (0.151; 0.0010) |

## Robustez entre modelos (Spearman entre las distancias coseno de los 509 tokens)

| Comparación | Spearman | Media del nulo | p (permutación) | p ajustado (BH) |
|---|---|---|---|---|
| s0 contra s1 | 0.725 | 0.000 | 0.0002 | 0.0003 |
| s0 contra s2 | 0.724 | 0.000 | 0.0002 | 0.0003 |
| s1 contra s2 | 0.731 | -0.000 | 0.0002 | 0.0003 |
| s0 contra ep30 | 0.696 | 0.000 | 0.0002 | 0.0003 |
| s0 contra columnas permutadas | -0.001 | -0.000 | 0.6917 | 0.6917 |
| s0 contra gaussiano | 0.002 | 0.000 | 0.2851 | 0.3422 |

Por categoría (sólo los tokens de cada una; 2.000 iteraciones):

| Categoría | s0 contra s1 | s0 contra s2 | s1 contra s2 |
|---|---|---|---|
| salario | 0.800 (p 0.0005) | 0.791 (p 0.0005) | 0.789 (p 0.0005) |
| nacimiento | 0.857 (p 0.0005) | 0.743 (p 0.0005) | 0.849 (p 0.0005) |
| rama | 0.372 (p 0.0005) | 0.398 (p 0.0005) | 0.399 (p 0.0005) |
| provincia | 0.409 (p 0.0005) | 0.308 (p 0.0005) | 0.389 (p 0.0005) |

## Hubness (grafo de 5 vecinos por coseno; grado medio = 5)

| Medida | s0 | s1 | s2 | ep30 |
|---|---|---|---|---|
| Asimetría de la distribución del grado de entrada | 1.33 | 1.18 | 0.94 | 0.39 |
| Grado máximo | 19 | 16 | 17 | 12 |
| Proporción de tokens con grado 0 | 0.4% | 0.6% | 0.2% | 1.0% |

Mediana del grado y proporción con grado 0 por categoría (modelo s0):

| Categoría | Tokens | Mediana | Máximo | Grado 0 |
|---|---|---|---|---|
| Sexo | 2 | 1 | 1 | 0% |
| Año de nacimiento | 77 | 5 | 8 | 0% |
| Salario (percentil) | 101 | 5 | 7 | 1% |
| Tamaño de empresa | 4 | 5 | 7 | 0% |
| Rama (r34) | 282 | 4 | 14 | 0% |
| Rama (r32) | 14 | 10 | 19 | 0% |
| Provincia | 25 | 5 | 8 | 0% |
| Antigüedad de la empresa | 4 | 4 | 7 | 0% |

## Vecinos de ejemplo (modelo s0)

| Token | Vecinos más cercanos (coseno), del más al menos cercano |
|---|---|
| RAMA_8000 (Enseñanza) | 1. RAMA_9100 (Servicios de asociaciones n.c.p.)<br>2. RAMA_8510 (Servicios relacionados con la salud humana)<br>3. RAMA_5521 (Servicios de expendio de comidas y bebidas en r…)<br>4. RAMA_7410 (Servicios jurídicos y de contabilidad)<br>5. RAMA_7493 (Servicios de limpieza de edificios) |
| RAMA_4520 (Construcción de edificios y sus partes y obras…) | 1. RAMA_7492 (Servicios de investigación y seguridad)<br>2. RAMA_7421 (Servicios de arquitectura e ingeniería y servic…)<br>3. RAMA_7010 (Servicios inmobiliarios realizados por cuenta p…)<br>4. RAMA_7499 (Servicios empresariales n.c.p.)<br>5. RAMA_7410 (Servicios jurídicos y de contabilidad) |
| RAMA_8510 (Servicios relacionados con la salud humana) | 1. RAMA_7492 (Servicios de investigación y seguridad)<br>2. RAMA_9100 (Servicios de asociaciones n.c.p.)<br>3. RAMA_8000 (Enseñanza)<br>4. RAMA_6022 (Servicio de transporte automotor de pasajeros)<br>5. RAMA_5239 (Venta al por menor en comercios especializados…) |
| RAMA_9100 (Servicios de asociaciones n.c.p.) | 1. RAMA_8000 (Enseñanza)<br>2. RAMA_8510 (Servicios relacionados con la salud humana)<br>3. RAMA_7010 (Servicios inmobiliarios realizados por cuenta p…)<br>4. RAMA_4520 (Construcción de edificios y sus partes y obras…)<br>5. RAMA_5231 (Venta al por menor de productos farmacéuticos) |
| RAMA_5211 (Venta al por menor en comercios no especializad…) | 1. RAMA_5521 (Servicios de expendio de comidas y bebidas en r…)<br>2. RAMA_7010 (Servicios inmobiliarios realizados por cuenta p…)<br>3. RAMA_7410 (Servicios jurídicos y de contabilidad)<br>4. RAMA_4520 (Construcción de edificios y sus partes y obras…)<br>5. RAMA_1810 (Fabricación de prendas de vestir) |
| RAMA_6022 (Servicio de transporte automotor de pasajeros) | 1. RAMA_8510 (Servicios relacionados con la salud humana)<br>2. RAMA_7010 (Servicios inmobiliarios realizados por cuenta p…)<br>3. RAMA_7492 (Servicios de investigación y seguridad)<br>4. RAMA_6021 (Servicio de transporte automotor de cargas)<br>5. RAMA_5521 (Servicios de expendio de comidas y bebidas en r…) |
| RAMA_7492 (Servicios de investigación y seguridad) | 1. RAMA_8510 (Servicios relacionados con la salud humana)<br>2. RAMA_4520 (Construcción de edificios y sus partes y obras…)<br>3. RAMA_7493 (Servicios de limpieza de edificios)<br>4. RAMA_5122 (Venta al por mayor de alimentos)<br>5. RAMA_6610 (Servicios de seguros) |
| RAMA_5521 (Servicios de expendio de comidas y bebidas en r…) | 1. RAMA_5512 (Servicios de alojamiento excepto en camping)<br>2. RAMA_9309 (Servicios n.c.p.)<br>3. RAMA_5211 (Venta al por menor en comercios no especializad…)<br>4. RAMA_5522 (Preparación y venta de comidas para llevar)<br>5. RAMA_7010 (Servicios inmobiliarios realizados por cuenta p…) |
| PROV_06 (Buenos Aires) | 1. PROV_02 (Ciudad Autónoma de Buenos Aires)<br>2. PROV_01 (Partidos de GBA)<br>3. PROV_14 (Córdoba)<br>4. PROV_82 (Santa Fe)<br>5. PROV_62 (Río Negro) |
| PROV_02 (Ciudad Autónoma de Buenos Aires) | 1. PROV_01 (Partidos de GBA)<br>2. PROV_06 (Buenos Aires)<br>3. PROV_14 (Córdoba)<br>4. PROV_50 (Mendoza)<br>5. PROV_82 (Santa Fe) |
| REM_000 (remuneración cero) | 1. REM_001 (percentil 1)<br>2. REM_002 (percentil 2)<br>3. REM_007 (percentil 7)<br>4. REM_003 (percentil 3)<br>5. REM_004 (percentil 4) |
| REM_050 (percentil 50) | 1. REM_051 (percentil 51)<br>2. REM_049 (percentil 49)<br>3. REM_048 (percentil 48)<br>4. REM_052 (percentil 52)<br>5. REM_045 (percentil 45) |
| REM_100 (percentil 100) | 1. REM_099 (percentil 99)<br>2. REM_098 (percentil 98)<br>3. REM_097 (percentil 97)<br>4. REM_096 (percentil 96)<br>5. REM_095 (percentil 95) |
| NAC_1980 (nacidos en 1980) | 1. NAC_1981 (nacidos en 1981)<br>2. NAC_1979 (nacidos en 1979)<br>3. NAC_1982 (nacidos en 1982)<br>4. NAC_1978 (nacidos en 1978)<br>5. NAC_1983 (nacidos en 1983) |
| TAM_1 (Más de 200 empleados) | 1. TAM_2 (Más de 50 y hasta 200 empleados)<br>2. TAM_3 (Más de 10 y hasta 50 empleados)<br>3. TAM_4 (Hasta 10)<br>4. ANT_1 (Anterior a 2001)<br>5. PROV_02 (Ciudad Autónoma de Buenos Aires) |
| SEXO_M (Mujer) | 1. SEXO_V (Varón)<br>2. NAC_1990 (nacidos en 1990)<br>3. NAC_1987 (nacidos en 1987)<br>4. NAC_1980 (nacidos en 1980)<br>5. NAC_1989 (nacidos en 1989) |

Vecinos de `REM_000` (remuneración cero) por modelo: s0: REM_001 (percentil 1), REM_002 (percentil 2), REM_007 (percentil 7); s1: REM_001 (percentil 1), REM_002 (percentil 2), REM_003 (percentil 3); s2: REM_001 (percentil 1), REM_005 (percentil 5), REM_004 (percentil 4); ep30: REM_003 (percentil 3), REM_001 (percentil 1), REM_094 (percentil 94).
