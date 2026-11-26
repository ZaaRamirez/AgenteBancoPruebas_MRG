"""PLANTILLA para un indicador nuevo. Copiar este archivo sin el guion bajo
(p. ej. indicadores/mios.py) y editar. Los archivos que empiezan con _ se ignoran.

Ejemplo: exactitud del inventario del VLM contra una referencia anotada a mano
(la imagen que vio el VLM se guarda si registro.guardar_imagenes: true).
"""
from evaluar import fila

REFERENCIA = {"Hidroferol": 3, "Paracetamol": 2}     # conteo real en la escena


def exactitud_inventario(c):
    out = []
    for e in c.eventos("vlm"):
        if e["proposito"] != "observacion_inicial" or not isinstance(e.get("datos"), dict):
            continue
        vistos = {str(i.get("name", "")).lower(): i.get("count") for i in e["datos"].get("items", [])}
        aciertos = [vistos.get(k.lower()) == v for k, v in REFERENCIA.items()]
        out.append(fila("percepcion", e["modelo"], "exactitud_inventario", aciertos, "fraccion"))
    return out


INDICADORES = [exactitud_inventario]
