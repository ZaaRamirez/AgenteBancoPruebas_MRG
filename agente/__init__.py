"""Agente fundacional modular (LLM + VLM + VLA) independiente de la tarea y del modelo.

El nucleo (este paquete) no depende de ROS. Solo agente/entornos/ros2.py y
agente/fuentes/ros2.py usan rclpy, para conectarse al robot (simulado o fisico)
y, si se quiere, recibir tareas por un topico.
"""
from .agente import Agente  # noqa: F401
from .tipos import Observacion, Tarea  # noqa: F401
