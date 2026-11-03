# Plan de v1: grilla mensual con huecos explícitos (SINREG)

Escrito el 2026-11-02. **Pendiente de aprobación**; no se implementó nada. Referencias: Cabezas y Montorsi, *Life Sequence Transformer* (LST, `data/docs/2506.01874v2.pdf`), y el informe final de v0 (`v0/informes/informe_final.md`).

## 1. Qué se quiere probar

En v0 los meses sin empleo registrado no producen token: sólo un salto entre fechas. Pero para la persona típica la mayor parte del calendario es hueco, y la salida del empleo registrado es justo lo que predice T1. En la fase 8b, una logística con variables de tiempo le ganó al transformer en T1 (AUL 0,634 contra 0,619).

**Hipótesis de v1:** si cada mes sin registro es un evento explícito con tipo, el transformer aprovecha mejor la historia de entradas y salidas.

Criterios de éxito, a evaluar con los mismos cortes y bootstrap que v0:
1. T1 (AUL) mejora respecto de v0 y alcanza o supera a la logística con tiempo (0,634).
2. T2, T3 y T4 no empeoran respecto de v0 (dentro del intervalo de confianza).
3. Si 1 no se cumple, v1 sirve igual como resultado negativo documentado: el hueco explícito no agrega nada que las variables de tiempo no den.

## 2. Decisiones ya tomadas (2026-11-02)

- Encoder, igual que v0 (MLM y SOP). El decoder generativo del LST queda fuera; la codificación se deja compatible con él.
- **Un evento por mes**, como v0, y no tramos con inicio y duración.
- Meses sin registro como eventos explícitos de tres tipos, definidos **sólo con la ventana de entrada** (nunca con datos posteriores al corte).
- Opción A: la secuencia empieza en el **primer mes observado** de la persona (como el LST) y la grilla termina siempre en el fin de la ventana (2015-12 en las tareas actuales).

## 3. Codificación

Para cada persona, después de recortar a la ventana (`mes_idx <= fin_ventana`) y de aplicar, si corresponde, una intervención de la fase 9:

- La grilla va del primer mes con registro hasta `fin_ventana`, mes por mes.
- Un mes con una o más relaciones: un evento por relación, igual que v0 (`REM TAM RAMA PROV EMPINI [SEP]`).
- Un mes sin ninguna relación: un evento `SINREG_x [SEP]`, con
  - `SINREG_INT` (intermitencia): hay un registro posterior dentro de la ventana;
  - `SINREG_SAL` (salida): no hay ningún registro posterior hasta el fin de la ventana.
- El tercer tipo del planteo (antes de la primera entrada) **no es token**: antes del primer evento no hay nada, y el modelo lo lee por edad y año del primer evento. Se lo llama «previo» en los informes y es el **primer registro observado**, no necesariamente la primera entrada al mercado (la MLER empieza en 1996).
- Edad y posición absoluta de un `SINREG` son las del mes que representa, igual que en un evento de empleo.
- Vocabulario: 519 → 521 tokens (categoría `sinreg`).

Ejemplo (una persona que entra en 2010-03, trabaja 2 meses, tiene 1 mes de hueco, trabaja 1 mes y sale):

```
[CLS] SEXO NAC [SEP]
REM TAM RAMA PROV EMPINI [SEP]   (2010-03)
REM TAM RAMA PROV EMPINI [SEP]   (2010-04)
SINREG_INT [SEP]              (2010-05)
REM TAM RAMA PROV EMPINI [SEP]   (2010-06)
SINREG_SAL [SEP] ...          (2010-07 hasta 2015-12)
```

### Variables de un evento de empleo (aclaradas el 2026-11-03)

| Token | Variable de la MLER | Qué es |
|---|---|---|
| `REM_000` a `REM_100` | `rem_tot` | Percentil del salario del mes (0 = remuneración nula) |
| `TAM_k` | `tramo_empleo` | Tamaño de la empresa (4 tramos); cambia dentro de la relación en el 11 % de los casos |
| `RAMA_<r34>` / `RAMA2_<r32>` | `r34` y `r32` | **Una sola variable de rama:** el r34 (4 dígitos, 301 códigos) para las 282 ramas con al menos 50 trabajadores en train y el r32 (2 dígitos) para las 19 raras. Los niveles r34 ⊂ r32 ⊂ letra están anidados (se verificó en el panel). La `letra` no entra al transformer. Decisión: se mantiene así por comparabilidad con v0 |
| `PROV_xx` | `provi` | Provincia |
| `EMPINI_1` a `EMPINI_4` | `tr_anio_ini` | **Inicio de actividades de la empresa** (padrón de AFIP): antes de 2001, 2001-05, 2006-10, después de 2010. Es la edad de la EMPRESA, no la antigüedad del trabajador. En v0 se llamaba «antigüedad» (`ANT_k`); en v1 se renombró |

Rama, provincia e inicio de la empresa son atributos de «último valor» (el manual: «última rama disponible»; el empleador puede actualizarlos) y son constantes dentro de cada relación (0 % de relaciones con más de un valor). La **antigüedad del trabajador en la relación no es una variable del transformer** ni de v0: sólo la usaban las líneas de base (`tareas/tiempo.py`, desde el primer mes de la relación, con censura a la izquierda en 1996-01) y el mapa de personas. Probarla como token es la fase 8.

Consecuencias que hay que tener presentes:
- La cohorte de T1 exige 12 meses registrados en 2015, así que sus secuencias **no terminan en `SINREG_SAL`**. El token aparece sobre todo en el pre-entrenamiento de quienes salieron antes del corte. Para T1 el valor de v1 está en los huecos intermitentes de la historia previa.
- Un hueco entre registros que se cierra después de 2015 se vería como `SAL` en una ventana a 2015-12 y como `INT` en una a 2021-12: el tipo es una propiedad de la ventana, no de la persona.
- La definición estricta de «salida definitiva» (sin ningún registro entre 2016 y 2021) se usa sólo en el análisis, nunca como entrada.
- Quien ya estaba registrado en enero de 1996 tiene el primer mes censurado a la izquierda; se anota, no se marca.

## 4. Qué cambia en el código

El corpus de eventos de v0 **no se regenera**: los huecos se derivan de `mes_idx` al armar el documento, porque dependen de la ventana.

| Módulo (copia en `v1/src/labor2vec/`) | Cambio |
|---|---|
| `tokens/vocabulario.py` | Dos tokens `SINREG_INT` y `SINREG_SAL` y su tabla de búsqueda |
| `tokens/documento.py` | `Opciones.grilla` y `Opciones.sinreg` (`"dos"` \| `"uno"` para el control); `Eventos` gana un campo `tipo` (0 empleo, 1 INT, 2 SAL); la grilla se arma después del recorte de la ventana y de las intervenciones; `bloques_de_eventos` emite `SINREG [SEP]` |
| `tokens/documento.py` (augmentación) | El remuestreo, el ruido temporal y el `drop_tokens` actúan sobre la grilla ya armada; ver decisiones menores |
| `tokens/datos.py`, `tareas/datos.py` | Pasar la opción desde el YAML |
| `analisis/*`, `tareas/*` | Sin cambios de fondo; las categorías de tokens se leen del vocabulario |
| Tests | Grilla (sin huecos al inicio, huecos intermedios, cola hasta el fin de la ventana, persona sin huecos, hueco cuyo tipo cambia con la ventana), vocabulario, largo del documento, SOP y MLM con tokens `SINREG` |

## 5. Estructura de carpetas y entorno

El disco del contenedor está al 93 % (8,6 GB libres), así que v1 **no duplica datos ni el entorno**:

- `v1/` lleva sólo código: `src/`, `tests/`, `configs/`, `scripts/`, `informes/`, `pyproject.toml`, `README.md` (copiados de v0, sin `runs/`, `data/processed/` ni informes viejos). Este plan vive en `plans/` de la raíz.
- `v1/data/processed` es un enlace simbólico a `../../v0/data/processed` (el corpus y el panel son los mismos); lo nuevo de v1 va a `v1/data/` y `v1/runs/`.
- Entorno: se reutiliza `v0/.venv` con `PYTHONPATH=v1/src` (va antes que el `.pth` editable de v0).
- Mendieta: espejo `~/labor2vec/v1/` con el mismo enlace a `v0/data`, y `PYTHONPATH` en `entorno_scratch.sh` como ya estaba previsto.

## 6. Fases

| Fase | Contenido | Criterio de aceptación | Pide OK |
|---|---|---|---|
| 0 | Crear `v1/`, enlace a los datos, correr los tests heredados de v0 | Los 166 tests de v0 pasan desde `v1/` | Correr tests |
| 1 | Codificación con grilla: código, tests y vocabulario nuevo | Tests en verde; informe de largos (mediana, p90, p99, truncados a 2.560) y de proporción de meses `SINREG` por tipo | — |
| 2 | Humo en CPU y piloto en Mendieta de una pasada por train | Pérdida baja; pérdida de SOP contra la referencia 1,636; reparto de la pérdida MLM entre `SINREG` y el resto | Encolar |
| 3 | Pre-entrenamiento de 150 épocas, semilla 0 (v0 tardó ≈ 5 h; con ~1,4 veces más tokens de media (medido en la fase 1) se estima 7-10 h, se mide en la fase 2) | Convergencia y perplejidad por categoría | Encolar |
| 4 | Ajuste fino de T1-T4 (y T2c/T4c) con 3 semillas, GRU sobre los documentos v1, comparación con v0 y con las líneas de base de v0 | Tabla de v1 contra v0 y contra la logística con tiempo, con IC por bootstrap | Encolar |
| 5 | Control con un solo token `SINREG` (`sinreg: uno`): ¿importa distinguir intermitencia de salida? | Misma tabla de T1-T4 | Encolar |
| 6 | Análisis: espacio de conceptos (¿se separan `INT` y `SAL`?), saliency por tipo de token, mapa de personas | Informe corto | Cómputo local |
| 7 | Opcional: embeddings de tiempo acotados al estilo del LST como control de extrapolación a 2019-2021 | Misma tabla | Encolar |
| 8 | Opcional: variante con un token de **antigüedad del trabajador en la relación** por mes de empleo (tramos y una marca «desde antes de 1996» por la censura); la antigüedad es información del pasado, sin fuga | Misma tabla, contra v1 principal | Encolar |

Se cambia **una variable por vez** (por eso la fase 7 va aparte): v1 principal = codificación nueva con el resto igual a v0. Se usan los mismos hiperparámetros, vocabulario (+2), ventana y cohortes.

## 7. Decisiones menores, con default

Se pueden objetar; si no, se siguen.

| Tema | Default | Por qué |
|---|---|---|
| Bloque de un hueco | `SINREG_x [SEP]` (2 tokens) | Mantiene la regla «un evento = bloque + `[SEP]`» para segmentos y SOP |
| MLM sobre `SINREG` | Enmascarables como cualquier token; la pérdida se reporta aparte | Son el 27,7 % de los tokens de eventos (medido) y casi triviales por contexto; si diluyen la pérdida, se baja su probabilidad (se ve en la fase 2) |
| SOP | Permuta sólo entre bloques del mismo largo (decisión del 2026-11-02) | Con la grilla los bloques miden 6 o 2 tokens; una permutación libre movía los `[SEP]` en el 86 % de los documentos y el SOP se resolvía por ese atajo (AUC 0,875 contra 0,56 del control). Con un solo largo es idéntico a v0 |
| Remuestreo (p = 0,4) | Sobre la grilla completa, empleo y `SINREG` por igual | Quitar un mes deja un salto de fecha sin token, que es un faltante de datos; es una regularización y puede ser control |
| Embeddings de tiempo | Los de v0 (Time2Vec con tendencia lineal) | El LST encontró que la tendencia lineal no acotada extrapola mal; se prueba aparte (fase 7) |
| Pluriempleo | Un evento por relación, mezclado como v0 | Sin cambios |
| Edad | Entero, como v0 | Sin cambios |

## 8. Riesgos y límites

- **Largo:** la persona mediana pasa de 264 a 508 tokens de eventos (1,9 veces), la media de 426 a 589 (1,4 veces) y la cola casi no cambia (p99 1.452 contra 1.462); medido en la fase 1 (`v1/informes/fase1.md`). Cabe en 2.560 y el costo sigue a la media. Si no entra en `multi`, se encadenan jobs.
- **Hueco no interpretable:** «sin registro» mezcla desempleo, informalidad, sector público, inactividad, autónomos, migración y fallecimiento. El modelo no puede distinguirlos y los informes no deben interpretarlos.
- **Fuga residual de v0** (rama, provincia y antigüedad del último registro): sigue igual; se mantiene la variante `sin_ultimo_valor` como control en T3 y T4.
- **Comparabilidad:** la perplejidad MLM de v1 no es comparable con la de v0 (otro conjunto de objetivos). Para comparar se usa la de los tokens que existen en ambos.
- **Un solo pre-entrenamiento** al comienzo; si la diferencia con v0 es chica, hacen falta semillas extra antes de afirmar nada.

## 9. Fuera de alcance

Decoder generativo y contrafactuales; tramos con inicio y duración; cambios de arquitectura; OM y MCSA multicanal (ver memoria `mcsa-retomar`); manuscrito (v0 sigue siendo el manuscrito vigente hasta tener resultados de v1).
