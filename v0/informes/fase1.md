# Fase 1: ingesta, validación y base de población

Fecha: 2026-10-15. Todo se regenera con un comando: `python -m labor2vec.datos --forzar-ingesta`. Tarda ~9 min en el contenedor local y la memoria disponible nunca bajó de 9 GB.

## Productos (`data/processed/`, no versionados)

| Archivo | Contenido |
|---|---|
| `panel_base.parquet` | 49.193.201 filas (trabajador × relación × mes), tipadas, ordenadas, códigos 0 → nulo, columnas de tiempo derivadas (`anio`, `mes`, `mes_idx`, `fecha`, `dias_desde_1996`) |
| `panel.parquet` | `panel_base` + `split` + `pct_rem_mes` (0 = remuneración 0; 1-100 = percentil del mes) |
| `ingesta.json` | Tamaño y SHA-256 del CSV de origen |
| `descriptores/*.parquet` | Códigos del MTEySS: sexo, provincia (con región INDEC), letra, r32, r34, tramo, antigüedad |
| `trabajadores.parquet` | 596.591 trabajadores, con motivo de exclusión |
| `relaciones.parquet` | 1.844.073 relaciones: atributos fijos, duración, brechas, censura |
| `splits.parquet` | 575.642 elegibles: train 402.949 / val 86.347 / test 86.346 (semilla 20261015) |
| `cortes_salario_mensual.parquet` | 99 cortes de percentil por mes (312 meses), calculados sólo con train |

## Validación contra los cuadros oficiales

Detalle en [validacion_mler.md](validacion_mler.md).

- **Empleo**: coincidencia exacta en C1.1 (mensual, 312 meses), C1.2 (por sexo), C1.3 (por letra, 4.368 celdas), C1.5 (por provincia) y C1.8 (por rama a 4 dígitos, 1.199 celdas). En C1.6 hay una sola diferencia: 2 casos "sin dato" de tamaño en diciembre de 2005.
- **Definición confirmada**: el MTEySS cuenta como "empleo" las filas (relación × mes), incluidas las remuneraciones 0.
- **Percentiles salariales (C2.2)**: coincidencia exacta (< 0,01 %) con cuantil discreto sobre todas las filas.
- **Remuneración media (C2.1)**: la nuestra queda **siempre** entre 0,1 % y 1,5 % por debajo de la oficial (media anual entre −0,25 % y −0,75 %). La hipótesis es que el cuadro oficial se calculó antes de la anonimización de los salarios altos (microagregación por encima del P98), que afecta a la media pero no a los percentiles. No impacta en el proyecto, que usa percentiles.

## Hallazgos de calidad

Detalle en [qa_mler.md](qa_mler.md).

- **Edades**: todas entre 18 y 85 años. El MTEySS ya eliminó los años de nacimiento fuera de ese rango.
- **`fnacim` faltante**: pasa del 0,8 % de las filas en 1996 al 2,1 % en 2021, porque a los que alguna vez tuvieron más de 85 años se les borra el dato. Estos trabajadores quedan excluidos (decisión tomada). Sesgo a tener en cuenta: se pierden historias largas de trabajadores mayores.
- **Remuneraciones 0**: crecen fuerte desde 2011 (de ~12 mil a ~55 mil filas por año). Se mantienen como token propio, pero conviene revisar el patrón en la fase 2.
- **Ramas**: 301 códigos r34 con trabajadores de train, 19 de ellos con menos de 50 trabajadores. Son candidatos a `[UNK]` o a subir a r32.
- **Splits balanceados**: proporción de mujeres, año de nacimiento, meses e historias similares en los tres.
- **Largo de las secuencias (variante A, 6 tokens por evento)**: mediana 298, p90 1.348, p99 1.882. Sólo el 0,24 % supera 2560 tokens, pero el 36,8 % supera 512. Conviene relleno dinámico o agrupar por largo en la fase 2, para no pagar 2560 tokens en cada secuencia.

## Incidente de memoria (2026-10-15)

La primera versión del paso de salario pegaba la lista de 99 cortes a cada una de las 49 M filas, y DuckDB estaba configurado con la mitad de la RAM y 8 hilos. Al correr el pipeline completo por segunda vez, la máquina se quedó sin memoria y se tildó.

Correcciones:
- DuckDB usa por defecto un cuarto de la RAM y 4 hilos (`utiles/duck.py`);
- el percentil se calcula con un ASOF JOIN contra una tabla larga de cortes.

Verificado: el panel resultante es idéntico al de la versión anterior (misma huella sobre todas las columnas clave), y los splits también.

## Criterio de aceptación

1. Productos generados con un solo comando: ✔.
2. Validación contra los cuadros oficiales: ✔ (las únicas diferencias están documentadas y explicadas).
3. Reporte de calidad: revisado y aprobado (2026-10-16): ✔.
4. `pytest`: ✔ (13 pruebas: 4 del modelo y 9 de invariantes de datos).
5. Sincronización de `data/processed` a Mendieta: pendiente (~380 MB, sin `panel_base`).
