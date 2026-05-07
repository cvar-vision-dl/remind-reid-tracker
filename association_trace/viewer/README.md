# Association Trace Viewer

Visor offline para `association_trace` sin dependencias externas.

## Uso

```bash
python3 APP2/Src/association_trace/viewer/serve_trace_viewer.py
```

Por defecto sirve:

- la UI en `http://127.0.0.1:8765/`
- las runs desde `APP2/outputs/association_trace`

Opciones utiles:

```bash
python3 APP2/Src/association_trace/viewer/serve_trace_viewer.py   --host 127.0.0.1   --port 8765   --runs-dir APP2/outputs/association_trace
```

## Qué hace hoy

- lista runs disponibles
- carga `manifest.json`, `pipeline_schema.json` y traces `frame + class`
- renderiza el grafo top-down del schema `v1`
- permite foco por deteccion
- muestra checks, values y filas del nodo seleccionado
- soporta pan y zoom sobre el grafo
