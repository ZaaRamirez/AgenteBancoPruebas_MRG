# Conjuntos etiquetados para el benchmark VLM

Cada conjunto es una carpeta `datos/vlm/<nombre>/` con `imagenes/` y `etiquetas.jsonl`
(una linea por imagen con su respuesta correcta). Se construyen con:

    python3 -m benchmark vlm-datos desde-lerobot --fuente lerobot:arrg-unam/omx_pick_hidroferol --salida omx_hidroferol --destino "brown box"
    python3 -m benchmark vlm-datos capturar --robot omx --tipo inventario --salida inventario_estante
    python3 -m benchmark vlm-datos desde-carpeta --carpeta fotos/ --etiquetas fotos/etiquetas.csv --salida fotos_estacion
    python3 -m benchmark vlm-datos revisar <nombre>        # mosaico para comprobar las etiquetas

Formato y detalles: BENCHMARK.md, seccion 3.1.
