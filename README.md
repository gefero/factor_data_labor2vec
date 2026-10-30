# labor2vec

Adaptación de life2vec (Savcisens et al., 2023) a la Muestra Longitudinal de Empleo Registrado (MLER).

## Estructura

- `v0/`: primera versión completa del proyecto (fases 0 a 9 y manuscrito preliminar). Es autocontenida: código,
  pruebas, configuraciones, planes, informes, corridas (`v0/runs/`) y derivados de los datos (`v0/data/processed/`).
  Se instala y se corre desde `v0/`. Ver `v0/README.md`.
- `data/`: datos crudos compartidos (`MLER.csv`, `mler_csv.rar`, que no se versionan) y documentación (`data/docs/`).

## Por qué v1

En v0 los meses sin empleo registrado no generan ningún evento ni token: la salida del mercado registrado sólo
deja un hueco entre fechas. La versión siguiente parte de repensar esa codificación.
