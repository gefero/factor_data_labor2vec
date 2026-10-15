# labor2vec

Adaptación de **life2vec** ([Savcisens et al., 2023](https://arxiv.org/abs/2306.03009)) a las trayectorias laborales de la **Muestra Longitudinal de Empleo Registrado (MLER)** del Ministerio de Trabajo, Empleo y Seguridad Social de la Argentina.

life2vec representa la vida de cada persona como una secuencia de eventos escrita en un "lenguaje sintético", y entrena con ella un transformer tipo BERT. Primero lo pre-entrena para aprender la estructura de las secuencias, y después lo ajusta para predecir resultados individuales. Este proyecto aplica esa metodología a 26 años (1996-2021) de historias de empleo asalariado registrado privado: ~600 mil trabajadores y 1,8 millones de relaciones laborales, a resolución mensual. Además de replicar el método, explora variantes adaptadas a estos datos.

## Objetivos

1. **Lenguaje sintético de trayectorias laborales**: cada mes de cada relación laboral es una "oración" con:
   - remuneración relativa (percentil dentro del mes, para neutralizar la inflación);
   - tamaño y antigüedad de la empresa;
   - rama de actividad (CLANAE);
   - provincia.

   Cada persona suma además una oración de fondo con sexo y año de nacimiento.
2. **Pre-entrenamiento** con *masked language modeling* y predicción del orden de la secuencia, como en el paper.
3. **Fine-tuning** sobre una tarea principal (T1): predecir la **salida del empleo registrado** en los 4 años siguientes a un corte (2015-12). Ocupa el lugar de la predicción de mortalidad del paper. Se compara contra baselines clásicos (regresión logística, redes simples, GRU).
4. **Análisis de los espacios de embeddings**: ¿el modelo ordena los percentiles salariales?, ¿agrupa las ramas según la jerarquía CLANAE y las provincias según su región?; representación de personas y TCAV.

El plan completo y las decisiones tomadas están en [plans/PLAN.md](plans/PLAN.md). El detalle de las fases en curso, en [plans/PLAN_fases_0_1.md](plans/PLAN_fases_0_1.md).

## Datos

| Archivo | Contenido |
|---|---|
| `data/MLER.csv` | Microdatos de la MLER: una fila por trabajador, relación laboral y mes. **No se versiona.** |
| `data/docs/manual_de_muestra_longitudinal_0.pdf` | Manual de la muestra (MTEySS, 2022) |
| `data/docs/Anexo_estadistico_y_descriptores.xlsx` | Códigos de las variables y cuadros de validación muestra contra población |
| `data/docs/2306.03009v1.pdf` | Paper de referencia |
| `data/processed/` | Datos procesados (parquet), generados por los scripts de `labor2vec.datos` |

Fuente de los datos: <http://www.trabajo.gob.ar/estadisticas/oede/mler.asp>. Aunque los datos están anonimizados, son microdatos de personas: no se suben al repositorio ni se comparten fuera del proyecto.

## Estructura

```
src/labor2vec/
  datos/      ingesta, descriptores, validación, calidad, población y splits
  tokens/     vocabulario y construcción de secuencias
  modelo/     backbone de life2vec (copiado de life2vec-light, licencia MIT)
  tareas/     pre-entrenamiento (MLM + SOP) y fine-tuning (T1)
  utiles/     configuración, semillas, rutas
configs/      configuraciones de experimentos (YAML)
scripts/cluster/  entorno y jobs de Slurm para Mendieta (CCAD-UNC)
tests/        pruebas
informes/     reportes de calidad y resultados
plans/        planes de trabajo
```

## Entorno

Se requiere [uv](https://docs.astral.sh/uv/). El proyecto usa Python 3.13 gestionado por uv.

```bash
# local, sin GPU (desarrollo y pruebas)
uv sync --extra cpu --group dev
source .venv/bin/activate      # o anteponer `uv run` a cada comando
pytest

# Mendieta (GPU A30): ver scripts/cluster/
uv sync --extra cu126
```

El entrenamiento corre en el cluster **Mendieta** del CCAD-UNC (GPU NVIDIA A30) mediante Slurm. El entorno local sólo se usa para desarrollo, pruebas chicas y análisis.

## Estado

- [x] Análisis de fuentes y plan general
- [x] Fase 0: entorno, estructura y prueba en GPU ([informes/fase0.md](informes/fase0.md))
- [ ] Fase 1: ingesta, validación y base de población
- [ ] Fase 2: tokenización y corpus
- [ ] Fases 3-4: prueba chica y pre-entrenamiento
- [ ] Fases 5-8: espacio de conceptos, fine-tuning, interpretabilidad y ablaciones

## Licencia

El código del proyecto se distribuye con licencia [MIT](LICENSE). Los archivos de `src/labor2vec/modelo/` copiados de life2vec-light conservan su propia licencia MIT (© 2024 Germans Savcisens; ver `src/labor2vec/modelo/LICENSE_life2vec-light`). La licencia no cubre los datos de la MLER ni la documentación de `data/docs/`, que pertenecen al MTEySS y a sus autores.

## Referencias y créditos

- Savcisens, G. et al. (2024). *Using sequences of life-events to predict human lives*. Nature Computational Science, 4, 43-56. <https://doi.org/10.1038/s43588-023-00573-5>
- Código de referencia: [carlomarxdk/life2vec-light](https://github.com/carlomarxdk/life2vec-light) y [SocialComplexityLab/life2vec](https://github.com/SocialComplexityLab/life2vec) (licencia MIT, © Germans Savcisens). Los archivos copiados o adaptados de esos repositorios lo indican en su encabezado.
- MTEySS, Observatorio de Empleo y Dinámica Empresarial: *Manual de la Muestra Longitudinal de Empleo Registrado* (2022).
- Cómputo: UNC Supercómputo (CCAD), Universidad Nacional de Córdoba.
