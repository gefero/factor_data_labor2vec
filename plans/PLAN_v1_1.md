# Plan de v1.1: v1 con el pre-entrenamiento corregido

Escrito el 2026-11-04. **Pendiente de aprobación**; no se implementó nada. Motivo y evidencia: `v1/informes/fase3_evaluacion.md` (secciones 2, 4.B y 6).

## 1. Qué se quiere probar

v1 (grilla mensual con meses sin registro explícitos) predice el empleo peor que v0 en el MLM: +0,11 nats sobre los mismos casos, diez veces el ruido de semilla, parejo para todas las personas. Además, su mejora en el SOP era un atajo. La codificación de los documentos no se discute: lo que se corrige es **cómo se entrena** sobre ella.

**Hipótesis de v1.1:** el déficit viene de la dilución del gradiente del empleo. Los `SINREG` son casi un cuarto de los objetivos del MLM y son casi gratis, mientras que el SOP conserva su peso fijo y ofrecía un atajo fácil. Corrigiendo eso, el modelo debería predecir el empleo como v0, sin perder la información de los huecos.

**Criterios, fijados antes de entrenar** (con `diagnostico_pre.py`, los mismos 3.696 trabajadores y casos de la sección 6 de la evaluación):
1. **Dilución confirmada** si la diferencia de NLL de empleo entre v1.1 y v0 (semilla 0) queda dentro de la banda de semilla de v0 (≤ 0,02 nats) o cerca (≤ 0,03). En ese caso v1.1 pasa a ser el modelo de la fase 4.
2. **Mejora parcial** si queda entre v0 y v1 (entre 0,03 y 0,10). La dilución explica una parte; se informa y se decide si pasar igual a la fase 4.
3. **Dilución descartada** si queda en ~0,11, como v1. La causa es otra (por ejemplo, capacidad del modelo).
4. **SOP**: por construcción no debe haber casos con atajo ni alteraciones invisibles (se verifica). Se informa la AUC pareada contra las tres semillas de v0, sin un umbral de éxito: es una tarea auxiliar.

## 2. Estructura: carpeta `v1_1/`, copia de `v1/`

Se sigue el mismo esquema que v0 → v1: `v1/` queda congelada tal como la describen sus informes, y `v1_1/` es una copia con los cambios.

- Se copian `src/`, `tests/`, `scripts/`, `configs/`, `pyproject.toml` y `README.md`. Los informes no se copian: los de v1 quedan en `v1/informes/` y los nuevos van a `v1_1/informes/`.
- `data/processed` es un enlace a `../../v0/data/processed`, como en v1. `runs/` y `logs/` arrancan vacíos.
- Mismo entorno de Python (el de v0) y el mismo tar de entorno en Mendieta. El destino en el cluster es `~/labor2vec/v1_1`.
- Los cambios entran como **opciones** con valor por defecto igual al de v1, y la configuración de v1.1 las activa. Así, desde `v1_1/` se puede reproducir v1 exactamente, como control (igual que `grilla: false` reproduce v0).

**Antes de copiar:** conviene commitear lo pendiente de v1 (`diagnostico_pre.py` y los informes de la evaluación), para que `v1_1/` parta de un punto registrado.

## 3. Cambios de código

### 3.1 SOP sin atajo — `Opciones.sop_sinreg_fijo` (v1: `false`; v1.1: `true`)

- En `_orden_sop` (`tokens/documento.py`), los bloques `SINREG` no se mueven nunca. Se permutan sólo los bloques de empleo entre sí (y entre los del mismo largo, como ahora, para la variante de eventos compactos).
- `_hay_orden_que_perturbar` exige al menos dos bloques de empleo **de contenido distinto**.
- Si la alteración deja los tokens idénticos (por ejemplo, la inversión de una secuencia simétrica), el mezclado se vuelve a sortear y el invertido queda como normal.
- Hace falta saber qué bloque es `SINREG`: `bloques_de_eventos` ya conoce el tipo de cada evento, y se lo pasa a `_orden_sop`.

### 3.2 `SINREG` sólo como entrada — `Opciones.mlm_sinreg` (v1: `"objetivo"`; v1.1: `"entrada"`)

- `enmascarar` recibe los ids que no se pueden tapar, además de los especiales; con `"entrada"`, `SINREG_INT` y `SINREG_SAL` quedan excluidos.
- **Reemplazos al azar del 10 %:** con `"entrada"` no sortean tokens `SINREG`. Si no, un `SINREG` aparecería donde hay empleo, cosa que nunca pasa sin el arreglo.
- **Línea de base unigrama** (`tokens/linea_base.py`): reparte la probabilidad sólo entre los tokens que pueden ser objetivo. Si no, la perplejidad unigrama queda mal calculada.

### 3.3 Cantidad a tapar como en v0 — `Opciones.mask_base` (v1: `"largo"`; v1.1: `"sin_sinreg"`)

- Con `"sin_sinreg"` se tapa `floor(0,3 × (largo − 2 × cantidad de SINREG))`. Ese número es exactamente el largo del documento de la misma persona en v0, así que cada persona tiene tantos objetivos como en v0.
- Junto con 3.2 se obtiene que, en validación y en documentos sin alterar, v0 y v1.1 tapen **los mismos casos** de cada persona: los candidatos son los mismos, en el mismo orden, con el mismo azar. Lo verifica un test.

### 3.4 Validación desagregada (sólo medición)

- En `modelo/pretrain.py`, además de la pérdida total, se registran la NLL y el acierto del MLM por grupo de objetivos: empleo, fondo (sexo y nacimiento) y `SINREG` (este último queda vacío en v1.1). El grupo de cada token sale de un buffer armado con las categorías del vocabulario.
- **Aviso:** la AUC del SOP de la validación va a dar menos que el 0,913 de v1. No es un empeoramiento: era el atajo.

### 3.5 Tests

- Con las opciones en sus valores por defecto, los documentos, las máscaras y el SOP son idénticos a los de v1 (test de regresión sobre una muestra), y con `grilla: false`, a los de v0.
- SOP de v1.1: los `SINREG` no cambian de lugar, ningún alterado viola el orden INT/SAL y ningún alterado queda idéntico al normal.
- MLM de v1.1: ningún objetivo ni reemplazo es `SINREG`; la cantidad de objetivos es la de v0, y los casos tapados coinciden con los de v0 en documentos sin alterar.
- Línea de base unigrama sin masa en los `SINREG`.

## 4. Pasos

| Paso | Qué | Dónde | Costo | Necesita OK |
|---|---|---|---|---|
| 0 | Commit de lo pendiente de v1 | local | — | sí (pedido de commit) |
| 1 | Copiar `v1/` a `v1_1/`, enlace de datos, README, config de pre-entrenamiento con las tres opciones | local | minutos | este plan |
| 2 | Cambios 3.1-3.4 y tests 3.5; correr toda la suite | local | ~1 h de trabajo; tests ~2 min | este plan |
| 3 | Humo en CPU (15 épocas, muestra del 1 %, como en v1): que entrene, que el SOP no tenga atajo y que las métricas desagregadas salgan | local | ~10-15 min de CPU | sí, antes de correr |
| 4 | Sincronizar `~/labor2vec/v1_1` en Mendieta y piloto corto (una pasada, partición `short`) | Mendieta | ~30 min de GPU | sí, antes de encolar |
| 5 | Pre-entrenamiento `pre_150ep` de v1.1, semilla 0, mismo calendario que v0 y v1 (56.250 pasos), 6 tramos `short` encadenados más uno de seguro, mails BEGIN, END, FAIL | Mendieta | ~5,5 h de GPU | sí, antes de encolar |
| 6 | `diagnostico_pre.py` con v1.1 contra v0 (semillas 0, 1 y 2) y contra v1; informe `v1_1/informes/fase3.md` con los criterios de la sección 1 | local | ~1 h de CPU | sí, antes de correr |
| 7 | Fase 4 (ajuste fino de T1 a T4 con 3 semillas y GRU) con el modelo que corresponda según el resultado | Mendieta | a planificar | plan aparte |

El piloto (paso 4) se puede saltear: los documentos son los mismos que en v1 y el costo por paso no cambia. Lo propongo igual porque es barato y confirma en GPU que no se rompió nada antes de gastar 5,5 h.

## 5. Qué queda fuera

- **Indicador de campo dentro del evento** (la sección 6, D5, de la evaluación): afecta a v0 y v1 por igual y cambia la comparación con life2vec. Se decide aparte.
- **Una corrida que aísle la dilución** (sólo el arreglo 3.1, sin 3.2 ni 3.3): no se hace. Si el resultado de v1.1 es ambiguo, se puede agregar después.
- Cambios de arquitectura, tamaño del modelo, ventana local o calendario.

## 6. Decisiones para el usuario

1. ¿Carpeta `v1_1/`, copia de v1 (recomendado, sección 2), o las mismas opciones dentro de `v1/` con configuraciones y corridas «v1.1»?
2. ¿Commit previo de lo pendiente de v1 (paso 0)?
3. ¿Piloto en GPU (paso 4) o directo al pre-entrenamiento?
