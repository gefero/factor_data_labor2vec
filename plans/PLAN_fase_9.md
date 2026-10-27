# Subplan: fase 9 (espacio de personas: interpretarlo y usarlo)

Estado (2026-10-26): plan original aprobado. **Etapas A1 (sondas) y A2 (ejes) hechas.** **A3 (tipologías) rediseñada** como análisis de secuencias multicanal (MCSA) espejo del transformer: **pendiente de aprobación y de decisiones** (sección 5). **Etapa B (cluster) sin empezar.** Complementa [PLAN.md](PLAN.md) y sigue a [PLAN_fase_8.md](PLAN_fase_8.md). Este documento reemplaza al borrador `PLAN_fase_9_mcsa.md`, que quedó integrado acá.

## 1. Contexto y decisiones

La fase 7 dejó dos resultados débiles sobre el espacio de personas:

- **El TCAV es degenerado.** El gradiente del logit es casi el mismo para todas las personas (coseno 0,98 en T1), así que el puntaje queda en 0 o 1.
- **Los mapas no tenían con qué compararse.** Los vecinos se parecen en salario o rama, pero no sabemos si un conteo de tokens daría lo mismo.

Se ataca el espacio de dos maneras:

- **Interpretarlo:** qué información guarda el vector de cada persona y si es algo más que «qué tokens tuvo». En particular, si guarda la dinámica (cambios, huecos, tendencias), que es lo que justificaría un modelo de secuencias. (A1, A2, B1, B3)
- **Usarlo:** armar tipologías de trayectorias laborales y compararlas con el método clásico de la sociología. (A3, B2)

Decisiones del usuario:

- Las tipologías del modelo salen del encoder **pre-entrenado** (`pre_150ep`); los ajustados a T1 y T3 quedan como comparación.
- Se usa **todo el test** (18.337 personas), no una submuestra.
- **El OM de un solo canal (salario × tamaño, 10 años) se descartó** y sus resultados se borraron: ve mucho menos que el transformer y la comparación no es pareja. La referencia clásica es el **MCSA espejo del transformer**: recibe la misma información, con la misma resolución y la misma ventana.
- No se presenta como una competencia por ver «quién tiene la mejor tipología»: no hay una correcta con la que verificar. El MCSA sirve para probar si el transformer modela algo que un método aditivo no ve (sección 4, A3).
- El trabajo va en dos etapas: A es local con lo ya extraído; B usa jobs cortos en Mendieta.

## 2. Insumos que ya existen (reusar)

- **Vectores de persona:** `runs/fase7/{t1,t3,pre}/personas.npz` (val y test, `vec`, `m_ult`, `m_pen`, `id`, `y`, `logit`), de `analisis/extraer.py` (`Sonda`, `cargar_modelo`).
- **Conceptos por persona:** `data/processed/tareas/conceptos_personas.parquet`, con `personas.conceptos_de()` y `personas.alinear()`.
- **Las 20 variables de dinámica:** `tiempo_{split}_v239.npz`, con `tabulares.variables_de_tiempo()` y `tiempo.NOMBRES`.
- **Proporciones de tokens:** `tabulares.atributos`; cacheadas en `prop_tokens_{val,test}_v239.npz` (`espacio.proporciones`).
- **Etiquetas de T1 a T4:** `data/processed/tareas/t{1..4}.parquet`; continuidad en `continuidad.parquet`.
- **Herramientas:** `analisis/metricas.py`, `analisis/figuras.py`, `personas.figura_mapa` y `proyecciones`.
- **Opciones del documento:** `tokens/documento.py` (`Opciones.fin_ventana`; `control_temporal` es el patrón para las intervenciones).
- **Ya escrito en esta fase:** `analisis/espacio.py` (proporciones, sondas, ejes, y las funciones genéricas de A3: `arbol`, `cortar`, `silueta`, `submatriz`, `perfil_de_tipos`, `figura_index`, `validez_externa`) y `analisis/secuencias.py` (estados mensuales, `costos_trate`, kernel de OM con numba), con `tests/test_espacio.py` (10 pruebas). Del código del OM simple se **reusan** `costos_trate` y el kernel, que se generaliza a varios canales.
- Hay `numba` y `sklearn`; **no hay R** (ni TraMineR ni `seqHMM`).

## 3. Etapa A1: sondas lineales (hecha)

Qué información guarda cada espacio. Espacios: vector de `pre` (última y penúltima capa), vectores de `t1` y `t3`, proporciones de tokens y tokens + vector. Objetivos: conceptos de 2015 (sexo, edad, rama, tamaño, salario, antigüedad) y las 20 variables de dinámica. Método: ridge para las continuas (R²) y logística balanceada para las categóricas (exactitud balanceada), entrenadas en val y evaluadas en test, con IC por bootstrap. Resultados y metodología en [informes/fase9.md](../informes/fase9.md) (`informes/fase9/resultados_sondas.md`). Resumen:

- En lo estático el vector iguala a los tokens (excepto la antigüedad en años, 0,91 contra 0,51).
- En 17 de 19 variables de dinámica el vector gana por mucho (cambios de empresa en 60 meses 0,55 contra 0,22; huecos en 60 meses 0,49 contra 0,13).
- Concatenar tokens al vector casi no agrega. El ajuste a T3 refuerza los cambios; la evolución del salario no se lee linealmente.

**Pendientes:** regenerar las sondas (unos 30 minutos) para sacar el artefacto de «meses desde el último registro» (R² = 1 por variable constante; el código ya devuelve NaN); repetirlas con `pre_150ep_sinuv` (sin rama, provincia ni antigüedad) para separar el efecto de los atributos de «último valor».

## 4. Etapa A2: ejes (hecha) y A3: tipologías con MCSA (por aprobar)

### A2. Ejes del espacio (hecha)
PCA del vector `pre` y correlación o $\eta^2$ de cada componente con los conceptos y las variables de dinámica (`resultados_ejes.md`). PC2 es el salario ($r=+0{,}78$); PC3 y PC7, la rama; PC1 (22 %) casi no se explica con nuestras variables.

### A3. Tipologías de trayectorias con MCSA espejo del transformer

**Por qué.** El método está tomado de López-Pernas, Saqr, Helske y Murphy, *Multi-Channel Sequence Analysis in Educational Research* (`data/docs/978-3-031-54464-4_13.pdf`): canales sincronizados con el mismo calendario; **OM multicanal por costos aditivos** (`seqMD`: sustitución por canal con TRATE, costo total = suma ponderada, pesos iguales por defecto); Ward `ward.D2` y silueta (ASW) para elegir k; gráficos apilados por canal y por cluster. No se usa el alfabeto combinado (inviable con estos alfabetos).

**Correspondencia con el input del transformer.** Población: las 18.337 personas del test.

| elemento del documento del transformer | contraparte en el MCSA |
|---|---|
| `REM_000` a `REM_100` (101 niveles, percentil del mes) | canal **salario**, mismos 101 niveles |
| `TAM_1` a `TAM_4` | canal **tamaño**, 4 estados |
| `RAMA_xxxx` (282 a 4 dígitos + 14 a 2 dígitos para las raras) | canal **rama**, misma codificación del vocabulario (~296 estados) |
| `PROV_xx` (25) | canal **provincia**, 25 estados |
| `ANT_1` a `ANT_4` (tramo de año de inicio de la empresa) | canal **antigüedad de la empresa**, 4 estados |
| fecha de cada evento | implícita: **paso mensual**, 240 pasos de 1996-01 a 2015-12 (`fin_ventana = 239`) |
| edad | canal **edad**, costo ordinal |
| `SEXO` | canal **sexo**, constante, con peso de «un token» |
| `NAC_xxxx` | ya está en la edad |
| varias relaciones en el mismo mes | canal **relaciones simultáneas** (0, 1, 2, 3 o más); los otros canales llevan la relación principal (mayor remuneración) |
| meses sin evento | estado **sin registro** en todos los canales a la vez |
| truncado a 2.560 tokens | afecta a 60 de 18.337 documentos (0,3 %): se ignora y se declara |
| `segmento` | sin contraparte (auxiliar de formato) |

Datos que lo respaldan: el 3,06 % de los meses-persona tiene más de una relación (7.202 personas, el 39 %, tienen algún mes así; máximo 25).

**Costos.**
- Sustitución por canal con **TRATE**, estimada con **20.000 personas de train** (como el modelo fija sus cortes con train). Excepciones: la edad usa $2\,\lvert\Delta\text{edad}\rvert/\Delta_{\max}$ y el sexo, costo 2 entre sexos distintos.
- Costo combinado de sustituir dos instantes $=\sum_c w_c\,\text{costo}_c(a_c,b_c)$, calculado **al vuelo** en el DP con una tabla por canal.
- Pesos: 1 para REM, TAM, RAMA, PROV, ANT, edad y relaciones simultáneas (el modelo da un token a cada atributo). Sexo: $1/240$ por paso, de modo que una diferencia de sexo cuesta una sustitución.
- Inserción y borrado: la mitad del costo máximo de sustitución combinado (equivale al valor por defecto de TraMineR).

**Cálculo.** Con 240 pasos y 8 canales, el cálculo exacto es unas 9 horas. Propuesta: **DP con banda** (|i − j| ≤ 24 meses), de 1,5 a 2 horas, en segundo plano, con control: en 500 personas se compara exacto contra acotado (error máximo y correlación) y, si difieren, se ensancha la banda.

**Análisis.**
1. Ward con k = 4 a 12 y ASW; perfiles por tipo y gráficos apilados por canal (salario agregado en quintiles y rama en las 14 secciones, sólo para graficar).
2. Las tipologías del espacio del modelo (`pre`) y de las proporciones de tokens se arman **con el mismo procedimiento** (Ward sobre distancia coseno de los vectores centrados y normalizados, mismos cortes de k).
3. Acuerdo entre las tres (ARI y NMI).
4. Validez externa con T1 a T4: AUC por validación cruzada de una logística sobre el tipo, y NMI.
5. Covariables (sexo, edad) post hoc, con la advertencia del capítulo (ignora la heterogeneidad dentro de cada grupo).
6. **Lectura.** El OM multicanal tampoco modela interacciones entre canales (los costos se suman), que es justo lo que se sostiene que el transformer sí hace. Si el MCSA alcanza al modelo, no hay evidencia de interacciones que aporten; si el modelo sigue por encima, apoya que modela algo que un OM aditivo no ve. En ninguno de los dos casos se afirma que una tipología sea «la mejor».

**Variantes de sensibilidad** (después de la corrida base):
- (a) Con un canal extra de **situación** (aún no entró / hueco / licencia): separa entrada tardía de intermitencia.
- (b) Sin el canal de edad.
- (c) Rama, provincia y antigüedad de la empresa con peso 1/3 cada una: en cada cambio de empleador cambian a la vez y con pesos iguales pesan 3 veces más que un cambio de salario.
- (d) Costos de sustitución tomados de los **embeddings del transformer** (distancia coseno entre embeddings de token): aísla el aporte del contexto (la atención).
- (e) Comparación del paso trimestral contra el mensual en una submuestra de 2.000 personas, si se quiere abaratar.

**Lo que se aprendió con el OM simple (descartado)** y que se conserva como lección: centrar los vectores antes de normalizar casi no cambió las tipologías (la silueta de `pre`, 0,03 a 0,05, refleja falta real de estructura de grupos); y los tipos de OM de un canal se separaban sobre todo por **cuándo entró la persona**, no por la trayectoria. Eso motiva el canal de situación (variante a).

**Implementación.**
- `analisis/secuencias.py`: `canales_mensuales(ids)` → `(n, 240, 8)` uint16 desde el panel con duckdb, con la codificación de rama del vocabulario; `costos_por_canal(muestra_train)`; `distancias_om_mc(seqs, subs, pesos, indel, banda)` con numba paralelo, formato condensado float32 (670 MB).
- `analisis/espacio.py`: método `omc` en `arbol`, `silueta`, tipologías y validez externa (ya genéricos); la figura apilada por canal; el subcomando `omc`. Hay que sacar `om` de `METODOS` (usa el OM simple, ya borrado).
- Caché: `omc_test_v239.npy` y su árbol de Ward.

**Pruebas.**
- El kernel con **un solo canal y sin banda** da lo mismo que el OM simple (ya probado contra una referencia en Python).
- Coincide con un DP de referencia en Python puro con 2 y 3 canales, pesos y costos distintos.
- La banda ancha da lo mismo que la exacta.
- Simetría, diagonal 0, el costo total suma los costos por canal y el sexo aporta exactamente 2 cuando difiere.
- No se puede contrastar con `TraMineR::seqMD` (no hay R).

**MHMM (opcional, más adelante).** Mezcla de HMM con emisiones independientes por canal, EM en PyTorch sobre 18.337 × 240 × 8, BIC para elegir y covariables en la pertenencia. Sólo tiene sentido si el MCSA deja dudas.

## 5. Decisiones pendientes del usuario

1. ¿Banda de 24 meses con control exacto contra acotado (propuesto), o cálculo exacto (unas 9 horas)?
2. ¿TRATE estimado con 20.000 personas de train (propuesto) o con las del test?
3. ¿Qué variantes de sensibilidad se hacen, además de la base?
4. ¿La fase 9 se presenta como **prueba de la hipótesis** (modelo contra MCSA) o como **descripción** de lo que capta el transformer? El diseño no cambia, pero sí la redacción del informe.

## 6. Etapa B (Mendieta; se pide OK antes de encolar; todo `sbatch` con `--mail-type=BEGIN,END,FAIL`)

Cambios en `analisis/extraer.py`: `--sin-grad` (no calcular `g`, más rápido y con lote 32) e `--intervencion NOMBRE`, que se pasa a `Opciones`.

### B1. Estabilidad entre semillas
Extraer vectores de `pre_150ep_s1` y `_s2`, `t1_l2v_lr0001_s1` y `_s2`, y `t3_l2v_s1` y `_s2` (6 jobs cortos, `--n-saliency 0 --sin-grad`). Medir: CKA lineal entre semillas, superposición de los 10 vecinos más cercanos, ARI entre las tipologías del espacio del modelo rehechas con cada semilla, y las sondas de A1 en cada semilla.

### B2. Trayectorias en el espacio
`pre` con `--set datos.fin_ventana={179,203,227}` (ventanas que terminan en 2010, 2012 y 2014; la de 2015 ya está), 3 jobs. Medir: cuánto se mueve cada persona entre ventanas, si se mueven más quienes cambiaron de empresa, rama o tamaño en el medio, y la **matriz de transición entre tipos** (centroides de k-means aplicados a cada ventana). Es el análisis más sustantivo para lo laboral y el paper no lo tiene.

### B3. Intervenciones en la historia
`Opciones.intervencion`, aplicado en `construir_documento` sobre `ev` antes de tokenizar, como `control_temporal`: `sin_ultimo_anio`, `sin_cambios_recientes`, `salario_arriba`, `salario_abajo`, `atrasar_2_anios`. Muestra de 2.000 personas del test; jobs para `t1`, `t3` y `pre`. Medir el cambio del logit, el desplazamiento en el espacio `pre` y la fracción que cambia de tipo. Permite separar **tiempo** de **orden**: las sondas de A1 prueban información temporal, no dependencia del orden.

### Extras de bajo costo que se agregan a B
- Control con un **encoder sin pre-entrenar** (pesos al azar) para saber cuánto aporta el pre-entrenamiento a lo que muestran las sondas y los ejes.
- Sondas y tipologías con `pre_150ep_sinuv` (sección 3).

## 7. Archivos

- **Existentes de esta fase:** `src/labor2vec/analisis/espacio.py`, `src/labor2vec/analisis/secuencias.py`, `tests/test_espacio.py`, `informes/fase9.md`, `informes/fase9/`.
- **Nuevos por A3:** `canales_mensuales`, `costos_por_canal`, `distancias_om_mc` (`secuencias.py`); subcomando `omc` y figura apilada (`espacio.py`); pruebas del kernel multicanal.
- **Modificados por B:** `analisis/extraer.py` (`--sin-grad`, `--intervencion`), `tokens/documento.py` (`Opciones.intervencion`), `opciones_de()`.
- **Informe:** `informes/fase9.md` ya tiene la metodología, las sondas y los ejes; faltan A3, los límites de la rama y de `ANT`, y B. Al final se agrega una sección a `informe_final.md`.

## 8. Costos

- **A3:** el OM multicanal, de 1,5 a 2 horas en segundo plano (con OK previo); Ward y el resto, minutos. Memoria: 670 MB para las distancias, más una copia en float64 para Ward (unos 1,3 GB); el contenedor tiene unos 15 GB.
- **B:** unos 12 jobs en la partición `short`, de 3 a 15 minutos cada uno. Se pide OK antes de encolar.

## 9. Verificación

1. `.venv/bin/python -m pytest` y `ruff check` sobre lo nuevo.
2. **Coherencia:** con `--intervencion` vacío, la extracción reproduce `runs/fase7/t1/personas.npz`; el `fin_ventana=239` de B2 reproduce el `pre` existente.
3. **Revisión manual** de las figuras y de los perfiles de los tipos antes de escribir cada sección del informe.
