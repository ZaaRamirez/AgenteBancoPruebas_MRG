"""Fuentes de tareas (capa de interaccion).
    archivo  -> ensayos en config/ensayos/*.yaml (repetibles, para evaluar)
    terminal -> se escriben las tareas a mano
    ros2     -> topico /agente/tarea (std_msgs/String con JSON); estados en /agente/estado
"""
import importlib

TIPOS = {"archivo": "agente.fuentes.archivo:FuenteArchivo",
         "terminal": "agente.fuentes.terminal:FuenteTerminal",
         "ros2": "agente.fuentes.ros2:FuenteROS2"}


def crear_fuente(tipo: str, **kw):
    clase = TIPOS.get(tipo, tipo)
    mod, cls = clase.split(":")
    return getattr(importlib.import_module(mod), cls)(**kw)
