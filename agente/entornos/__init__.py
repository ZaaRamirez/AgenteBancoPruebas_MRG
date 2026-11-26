"""Entornos: conexion del agente con la plataforma.
    simulado -> en memoria, sin ROS (pruebas de arquitectura)
    ros2     -> mensajes ROS 2 estandar (gemelo MuJoCo o robot fisico)
    paquete.modulo:Clase -> cualquier Entorno propio
"""
import importlib

TIPOS = {"simulado": "agente.entornos.simulado:EntornoSimulado",
         "ros2": "agente.entornos.ros2:EntornoROS2"}


def crear_entorno(tipo: str = "simulado", **kw):
    clase = TIPOS.get(tipo, tipo)
    if ":" not in clase:
        raise ValueError(f"Entorno '{tipo}' desconocido: {sorted(TIPOS)} o 'paquete.modulo:Clase'")
    mod, cls = clase.split(":")
    return getattr(importlib.import_module(mod), cls)(**kw)
