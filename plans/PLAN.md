# Plan: replicar life2vec sobre la MLER (Argentina)

Estado: borrador v0 (2026-10-15), resultado del análisis de las fuentes. Nada de modelado implementado todavía.

## 1. Resumen ejecutivo

- **Objetivo**: adaptar la metodología de life2vec (Savcisens et al., 2023) a la Muestra Longitudinal de Empleo Registrado (MLER, MTEySS): secuencias de eventos → transformer tipo BERT → pre-entrenamiento (MLM + SOP) → fine-tuning → espacio de conceptos y de personas.
- **Viabilidad**: alta. La MLER tiene 49,2 M de filas, 596.591 trabajadores, 1,84 M de relaciones laborales y 26 años (1996-2021). Es una sola fuente laboral, a escala mensual, sin salud, muerte, educación ni municipio. Eso cambia **qué** se puede predecir, no **cómo** se modela.
- **El cambio más importante respecto del paper**: no hay un resultado tipo "mortalidad". Las tareas de fine-tuning hay que definirlas desde cero con datos laborales (sección 5). Es la decisión de diseño que más condiciona el resto.
- **Cómputo**: el pre-entrenamiento del paper usa ~900 k secuencias muestreadas en total (30 épocas × 30.000), menos de dos pasadas sobre nuestro corpus. Entra en una sola GPU A30 del CCAD (Mendieta). El contenedor local **no tiene GPU y le quedan ~5 GB de disco**: sólo sirve para desarrollo y pruebas chicas.
- **Estrategia de código**: tomar de `life2vec-light` el esqueleto del modelo (embeddings, Performer, pre-entrenamiento), portar de `life2vec` (completo) las piezas de fine-tuning y métricas, y escribir un pipeline de datos propio con DuckDB/Polars. El pipeline de ambos repos está atado a dask y a los datos daneses.

## 2. Análisis de las fuentes

### 2.1 Paper (arXiv 2306.03009v1)

Pipeline:

1. **Datos**: registros diarios de Dinamarca (mercado laboral AMRUN + registro de pacientes LPR), 2008-2015, 25-65 años al 31-12-2015, ≥12 registros laborales en 2015. 2,3 M de personas para mortalidad (35-55 años).
2. **Lenguaje sintético**: cada registro es una "oración" de tokens de concepto. Las variables continuas se discretizan por cuantiles (ingreso en 100 bins, 10 bins para el período laboral). Hay una oración de background (sexo, origen, año y mes de nacimiento). Tokens especiales: `[CLS] [SEP] [PAD] [UNK] [MASK]`.
3. **Información posicional por evento**: edad (años cumplidos), posición absoluta (días desde el origen) y segmento (3 valores; desempata eventos del mismo día). Longitud máxima 2560 tokens; si se excede se descartan los eventos más viejos y se conserva el background.
4. **Modelo**: BERT-like con:
   - embeddings de token (centrados), Time2Vec para edad y posición absoluta (coseno) y embedding de segmento, combinados con pesos escalares entrenables inicializados en 0;
   - atención mixta: cabezas locales softmax (ventana 38) más cabezas Performer;
   - ReZero, ScaleNorm, Swish y weight tying.
5. **Pre-entrenamiento**: MLM (30 % de tokens; 80/10/10) más SOP (permutar o invertir eventos, clasificación sobre `[CLS]`). AdamW con OneCycle, 30 épocas de 30.000 secuencias, con augmentación.
6. **Fine-tuning**: mortalidad a 4 años (binaria, con *positive-unlabeled learning* porque hay emigrados y desaparecidos); extraversión (HEXACO, ordinal, con pérdida CDW + focal + label smoothing); emigración (SI).
7. **Baselines**: tablas de vida (edad+sexo), regresión logística, FFNN, GRU. Métricas: C-MCC corregido y AUL.
8. **Análisis**:
   - espacio de conceptos: PaCMAP, vecinos por coseno, test de aleatorización (3 modelos), hubness;
   - espacio de personas: DensMAP, saliency (SmoothGrad) y TCAV.

Divergencias entre paper y código que conviene saber (el código publicado es la referencia operativa):

- El paper describe 4 cabezas Performer + 4 softmax. El config de `life2vec` usa `n_heads: 10`, `n_local: 7`, `hidden_size: 280`, `hidden_ff: 2210`, 5 encoders, 436 random features, `lr 5e-3`, `weight_decay 0.01`, `local_window 38`, 8,4 M de parámetros.
- SOP: el paper cambia tokens entre eventos en el 10 % de los casos. `life2vec-light` clasifica 3 clases (orden normal, invertido en ~5 %, mezclado en ~5 %) con `label_smoothing=0.1`. Las pérdidas se ponderan 0,8 MLM / 0,2 SOP.
- Augmentación (config `mlm.yaml`): `p_sequence_resample 0.4`, `abspos_noise 0.1`, `hide_background 0.01`, `drop_tokens 0.01`.

### 2.2 Repos de referencia

| Repo | Qué aporta | Qué no sirve tal cual |
|---|---|---|
| `life2vec-light` | Backbone listo para datos propios: `embeddings.py` (Time2Vec, centrado, ReZero), `performer.py`, `transformer.py`, `models/pretrain.py` (MLM+SOP, OneCycle), tarea MLM, vocabulario, augmentación, datamodule. Ejemplos con datos sintéticos. Python 3.11, torch 2.3, PL 2.3.3. | Pipeline en dask/pandas, pensado para pocos datos. Sin fine-tuning, sin métricas ni losses del paper, sin logging. |
| `life2vec` (completo) | Hydra + PL. `cls_model.py` (fine-tuning binario), `hexaco_model.py`, `metrics.py` (C-MCC, AUL), losses (asimétrica PU, CDW, focal), baselines RNN y FFN, notebooks de embedding, visualización y TCAV. | Rutas redactadas. Dependencias extra (`focal_loss`, `coral_pytorch`, `imblearn`). Fuentes y poblaciones daneses (health, labour, education). |

**Decisión**: backbone y pre-entrenamiento desde `light`; fine-tuning, métricas y análisis portados desde el completo; datos y vocabulario escritos desde cero. Para que un resultado nuestro sea comparable con el paper, el modelo se copia (misma arquitectura y mismos hiperparámetros iniciales) y lo único que se cambia es la capa de datos.

### 2.3 Dataset MLER

Perfil propio sobre `data/MLER.csv` (2,96 GB). Convertirlo a parquet con DuckDB tarda ~30 s y da 218 MB.

**Estructura**: una fila por (trabajador, relación laboral, mes). Columnas:

| Variable | Qué es | Notas relevantes |
|---|---|---|
| `id_trabajador` | Id anonimizado (1-606.908; 596.591 presentes) | Es una clave aleatoria. Puede asignarse a train/val/test sin riesgo de fuga entre splits |
| `id_relacion` | Id de relación laboral del trabajador (1-152) | No es cronológico (sólo 32 % de las relaciones coincide con el orden de aparición). Mediana 2 relaciones por trabajador, p99 = 15 |
| `tiempo` | AAAAMM, 199601-202112 | 26 años, 312 meses |
| `rem_tot` | Remuneración bruta mensual (nominal) | Ver "calidad" abajo |
| `tramo_empleo` | Tamaño de la empresa en el año: 1 >200, 2 50-200, 3 10-49, 4 ≤9 | Varía por año, no dentro de relación-año. 5.206 filas con valor 0 (dato faltante) |
| `tr_anio_ini` | Antigüedad de la empresa en 4 tramos | Constante por relación |
| `provi` | Provincia (26 valores, códigos 0-94) | Constante por relación ("último dato") |
| `letra`, `r32`, `r34` | Rama CLANAE 1997: 15 letras / 57 dos dígitos / 302 cuatro dígitos | Jerarquía anidada. Constante por relación ("última rama") |
| `fnacim` | **Sólo año** de nacimiento (1911-2003) | Nulo en 862.376 filas (20.942 trabajadores, 3,5 %) |
| `sexo` | 0 sin dato, 1 mujer, 2 hombre | Consistente por trabajador. 9.882 trabajadores con 0 |
| `pondera` | Ponderador | Constante (33,33) en todas las filas: el diseño es muestreo simple al 3 %. No aporta |

**Hechos verificados**:

- `(id_trabajador, id_relacion, tiempo)` es único. No hay filas duplicadas.
- Todas las variables de empresa y `provi` son constantes dentro de cada relación (0 excepciones en 1,84 M).
- `sexo` y `fnacim` son constantes dentro de cada trabajador.
- Registros por trabajador: mediana 47, p90 221, p99 313, máx. 1.704 (la tabla es larga porque un trabajador puede tener varias relaciones en el mismo mes). Meses distintos por trabajador: mediana 46, máx. 312.
- Hay 1,34 M de (trabajador, mes) con más de una relación (2,8 % de los meses).
- Intermitencias: 569.567 saltos de más de un mes dentro de una relación (1,2 %); 331.765 son de más de 3 meses.
- Cobertura anual: ~136 k trabajadores activos en 1996, ~230 k en 2015-2018 y ~214 k en 2021. Hay caída en 2002 y en 2020 (COVID).
- Sexo: 210.418 mujeres y 376.291 hombres (consistente con el manual: 1 = mujer, 2 = hombre).

**Calidad y características con impacto directo en el diseño**:

1. **Remuneraciones nominales con inflación fuerte**. La mediana de `rem_tot` pasa de 557 (1996) a 73.491 (2021). Un binning global por cuantiles no sirve. Hay que **binear por percentil dentro de cada mes** (o año), lo que equivale a una deflactación implícita. No se dispone de un IPC confiable para 2007-2015, así que el percentil intra-mes es lo más defendible.
2. `rem_tot = 0`: 656.537 filas (1,3 %), que el manual interpreta como licencia. Informa algo, por lo que conviene un token propio (`INC_ZERO`) en vez de descartarlas (el paper trata el ingreso 0 como faltante).
3. Anonimización: ruido ±3 % en todos los salarios y **microagregación del 2 % superior por rama** (promedio de 3 vecinos). Los percentiles altos están levemente comprimidos. Hay 273 valores >10 M y 274.143 filas con 0 < salario < 30.
4. **Sólo año de nacimiento**: la edad tiene error de ±1 año y no existe el token de mes de nacimiento del background. Los mayores de 85 en algún momento tienen `fnacim` nulo (sesgo de selección en edades altas).
5. **Atributos "último valor"**: `provi`, rama y `tr_anio_ini` son los vigentes en el último dato de la relación, no en cada mes. Para predicción hacia adelante esto introduce una fuga **leve** (por ejemplo, una rama actualizada después del corte). Hay que documentarlo y, si conviene, medirlo con una ablación.
6. **Censura**: cualquier relación activa en 199601 o 202112 está censurada. Las tareas con horizonte futuro deben cortar antes de 202112 − horizonte.
7. **Población**: sólo empleo asalariado **registrado privado**. Que alguien "desaparezca" de la muestra mezcla informalidad, desempleo, empleo público, cuentapropismo, inactividad, jubilación, emigración y muerte. Esto afecta la interpretación de cualquier tarea de "salida".
8. **No hay identificador de empresa** (a propósito). No se pueden hacer análisis a nivel firma.
9. Falta el **descriptor de códigos** (`Anexo estadístico y descriptores.xlsx`, `..._ramas.xlsx`), citado por el manual y no incluido en `data/docs/`. Se necesita para etiquetar `provi`, `letra`, `r32` y `r34`, y es clave para validar el espacio de conceptos.

### 2.4 Mapeo paper → MLER

| Elemento del paper | Equivalente en MLER | Comentario |
|---|---|---|
| Eventos de salud (ICD-10) | **No existe** | Se pierde la mitad del vocabulario y la tarea de mortalidad |
| Registro laboral diario | Registro mensual (relación, mes) | Una oración por (relación, mes) |
| Ingreso (100 cuantiles) | `rem_tot`, 100 percentiles intra-mes + `INC_ZERO` | ver 2.3.1 |
| Tipo de empleo (ISCO) | **No existe** | No hay ocupación, sólo rama |
| Industria (DB07, 290) | `r34` (302 valores) | Jerarquía `letra` ⊂ `r32` ⊂ `r34` verificable |
| Tipo de empresa (ESA2010) | `tramo_empleo` (4) + `tr_anio_ini` (4) | |
| Municipio de residencia (97) | `provi` (26) | Constante por relación |
| Estado laboral, % días del mes, jornada, tramo impositivo | **No existen** | |
| Nacimiento (mes y año) | Sólo año | |
| Origen (nacional/extranjero) | **No existe** | |
| Posición absoluta (días) | Días desde 1996-01-01 (por ejemplo el día 15 del mes) | |
| Segmento | Para desempatar relaciones simultáneas | Aparece en el 2,8 % de los meses |
| 2008-2015 (8 años), 6 M personas | 1996-2021 (26 años), 0,6 M personas | Más profundidad temporal, menos personas y menos atributos por evento |

## 3. Diseño propuesto

### 3.1 Documentos (secuencias)

- Una oración por (relación, mes), ordenada por `tiempo` y luego por `id_relacion`.
- Oración base (variante A, fiel al paper): `[INC_p, TRAMO_n, IND_r34, PROV_x, ANT_k]` + `[SEP]`, 6 tokens por evento.
- Oración de background: `[CLS] SEX_x BYEAR_y [SEP]`. Sin mes de nacimiento.
- Largo esperado: mediana ≈ 280 tokens, p99 ≈ 1.900 tokens. Los pocos casos con >2560 se recortan por los eventos más viejos, como en el paper.
- **Variante B (ablación)**: los atributos estáticos de la relación sólo en el primer mes de cada relación, y después sólo ingreso + tramo. Acorta las secuencias a la mitad.
- Opcional: evento de "brecha" (meses sin registro) para que el modelo conozca los huecos. Se prueba como ablación.

### 3.2 Posiciones

- `abspos`: días desde 1996-01-01.
- `age`: `año(tiempo) − fnacim` (entero).
- `segment`: alternancia por índice dentro del mes cuando hay más de una relación.
- Ruido temporal de la augmentación: ±15 días (la distribución mensual es mucho más gruesa que el ±5 del paper).

### 3.3 Población y cortes

- **Corte principal** `T0 = 2015-12-31`, igual que el paper:
  - ventana de entrada 1996-2015 (o los últimos 8 años, para comparar con el paper);
  - ventana de predicción 2016-2019 (4 años; el 2020-21 queda como prueba fuera de distribución, ya que incluye el shock de COVID).
- Filtros análogos al paper: edad 25-65 al corte, ≥12 meses con registro en el último año, `sexo` y `fnacim` consistentes. En números: 230.308 trabajadores activos en 2015, 152.199 con ≥12 meses, 122.492 en la cohorte de 25-55 años.
- Pre-entrenamiento sobre **todos** los trabajadores (y todas las edades) con ventana de entrada completa. Un enfoque más rico es entrenar con cortes múltiples (cualquier mes ≥ 2001 como punto de corte) para multiplicar los ejemplos. Se evalúa después del primer modelo.
- Split 70/15/15 aleatorio por `id_trabajador`, igual que el paper. Más una evaluación **temporal** adicional (entrenar hasta T0, probar con cortes posteriores).

### 3.4 Modelo

Arquitectura idéntica a la del paper para empezar (hidden 280, 5 encoders, 8,4 M parámetros). El vocabulario es mucho más chico (~450-500 tokens contra 2.043), por lo que el modelo tendrá menos parámetros de embedding. Si el pre-entrenamiento no converge o sobreajusta, se reduce. `vocab_size` se calcula dinámicamente (en el config del paper está hardcodeado).

## 4. Tareas de fine-tuning (reemplazo de mortalidad y personalidad)

Ninguna variable de la MLER permite replicar mortalidad ni personalidad. Propuesta:

| # | Tarea | Definición | Análogo del paper | Tasa base aprox. |
|---|---|---|---|---|
| T1 | **Salida del empleo registrado** | y=1 si el trabajador no tiene ningún registro en 2019 (4 años después del corte). Variantes: sin registro en 2017 (1 año después) o sin registro de 2016 a 2019 | Mortalidad | ~16 % (2019); ~6,8 % (2017); 0,5 % (nunca vuelve) |
| T2 | **Posición salarial futura** (ordinal) | Quintil o decil del salario relativo en 2019, entre quienes siguen registrados | Personalidad (ordinal) | — |
| T3 | **Cambio de sector** | Cambio de `letra` entre dos años, entre quienes siguen registrados. Multiclase | Emigración (tarea adicional) | A medir |
| T4 | **Transición de tamaño de empresa** | Pasaje de `tramo_empleo` | — | A medir |

- **T1 es la tarea principal**. Tasa base calculada sobre la cohorte 25-55 años con ≥12 registros en 2015 (n = 122.492).
- A diferencia del paper, acá no hay etiquetas "no observables" en sentido estricto: no aparecer en la muestra es una etiqueta observada, aunque ambigua (sección 2.3.7). Se empieza con entropía cruzada y se evalúa si el enfoque PU suma algo. No se asume que aplica.
- **Baselines** (idénticos a los del paper): mayoría, azar, "tabla de vida" (regresión logística con edad y sexo), regresión logística con vector de conteos de tokens, FFNN y GRU con la misma entrada que el transformer.
- **Métricas**: AUC-ROC, AUL, PR-AUC, MCC con umbral óptimo en validación, calibración. QWK para T2. C-MCC sólo si se adopta PU.
- **Cuidado con la fuga** en T1: la rama, `provi` y `tr_anio_ini` son "último valor". Se agrega un control: repetir T1 sin esas variables.

## 5. Análisis de embeddings e interpretabilidad

Con la jerarquía de la MLER las validaciones son más fuertes que las del paper, porque hay estructura conocida:

- **Espacio de conceptos**:
  - ¿los 100 percentiles de ingreso salen ordenados en una curva (como en el paper)?
  - pureza de k-vecinos de los 302 `r34` respecto de su `letra` y su `r32`;
  - agrupamiento de provincias por región geográfica (pampeana, NOA, NEA, Cuyo, Patagonia);
  - orden de los años de nacimiento;
  - tokens `tramo_empleo` y `tr_anio_ini` ordenados.
  - Robustez con el test de aleatorización de 3 modelos del paper, y hubness.
- **Espacio de personas**: proyección (DensMAP/PaCMAP) sobre T1; TCAV con conceptos construibles desde los datos (sexo, sector, tamaño de firma, tercil de salario, antigüedad en el empleo registrado); saliency SmoothGrad.

## 6. Hoja de ruta

| Fase | Contenido | Entregable / criterio de aceptación |
|---|---|---|
| 0 | Entorno. Python 3.11 con `uv` (el repo light pinea 3.11.7 y torch 2.3). Torch CPU en el contenedor local, torch CUDA sólo en Mendieta. Estructura `src/` | `pyproject.toml`, test mínimo que importa el backbone |
| 1 | Datos: CSV → parquet particionado, reporte de calidad (reproducir 2.3), descriptores (descargar los xlsx del MTEySS) | `data/processed/mler.parquet`, notebook o script de QA |
| 2 | Tokenización y vocabulario (binning intra-mes, `INC_ZERO`, `r34`, etc.), construcción de documentos, populations y splits | Corpus serializado, estadísticas de largo, vocabulario con categorías |
| 3 | Smoke test en CPU con ~1 % de los trabajadores, usando el backbone de `light` | Una pasada de MLM+SOP sin errores, pérdida bajando |
| 4 | Pre-entrenamiento en Mendieta (patrón de jobs encadenados `short`/`multi` con reanudación; ver skill `ccad-unc`) | Checkpoint, curvas de pérdida y perplexidad MLM en validación |
| 5 | Evaluación del espacio de conceptos (sección 5) | Figuras y métricas de vecindad, tests de robustez |
| 6 | Fine-tuning T1 y baselines, luego T2-T4 | Tabla comparable con la A2 del paper |
| 7 | Interpretabilidad de personas (TCAV, saliency) | Notebook y figuras |
| 8 | Ablaciones (variante B, `r32` vs `r34`, con y sin atributos "último valor", cortes múltiples) e informe | Informe final |

Estimaciones de cómputo (no medidas): el pre-entrenamiento completo es ~900 k secuencias de ≤2560 tokens, con modelo de 8 M de parámetros. Debería caber en pocas horas de una A30; se confirma en la fase 3.

## 7. Riesgos y limitaciones

- **Sin salud, ocupación, educación ni jornada**: el lenguaje sintético es mucho más pobre, y el valor agregado del transformer sobre baselines de conteo puede ser menor al del paper. Hay que reportarlo como resultado, no esconderlo.
- **Mezcla de causas en T1**: "salir de la muestra" no es un evento bien definido (sección 2.3.7).
- **Fugas** por atributos "último valor" (2.3.5): se mitigan con ablación.
- **Datos anonimizados** (ruido ±3 %, microagregación del 2 % superior): efecto probable sobre los percentiles altos.
- **Tamaño de muestra**: 0,6 M de personas contra 3,3 M, y sólo ~122 k en la cohorte de T1. Los intervalos de confianza serán más anchos.
- **Disco local**: ~5 GB libres. Todo lo pesado va a `data/processed` en parquet comprimido o a `/scratch` del cluster. No instalar torch con CUDA localmente.
- **Cambio de versiones**: el entorno local trae Python 3.14. `performer-pytorch` y el código del paper están probados con 3.11, así que se usa 3.11 con `uv`.
- **Ética**: es un prototipo de investigación. El paper restringe explícitamente el uso individual automatizado y pide auditoría de equidad; vale lo mismo para estos modelos y para los datos (aun anonimizados, son microdatos de personas).

## 8. Decisiones tomadas (2026-10-15)

1. **Tarea principal**: T1, salida del empleo registrado.
2. **Alcance**: además de replicar, explorar variantes adaptadas a estos datos (la replicación fiel queda como línea de base).
3. **Cómputo**: CCAD, Mendieta. La SSH funciona desde el contenedor a través del agente (`grosati@mendieta`).
4. **Descriptores**: descargados del MTEySS en `data/docs/Anexo_estadistico_y_descriptores.xlsx`. Incluyen también cuadros de validación muestra contra población.
5. **Trabajadores sin `sexo` o sin `fnacim`**: se excluyen (20.949, 3,5 %).
6. **Idioma**: código y documentación en castellano; la traducción queda para después.

Correcciones a la sección 2.3, surgidas de los descriptores:
- Los faltantes de `provi`, `letra`, `r32`, `r34`, `tramo_empleo` y `sexo` vienen codificados como **0** ("Sin definir"), no como nulos.
- `r32 == r34 // 100` en todas las filas.
- `provi = 1` es "Partidos de GBA", separado de la provincia de Buenos Aires (`6`).

Python: 3.13 gestionado por `uv` en local y en el cluster (en lugar de 3.11), para reutilizar la receta ya probada en Mendieta.

Subplanes detallados de las fases 0 y 1: [PLAN_fases_0_1.md](PLAN_fases_0_1.md).
