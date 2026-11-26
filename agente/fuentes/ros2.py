"""Tareas por ROS 2: la capa de interaccion como nodo.

  Entrada  /agente/tarea   std_msgs/String  JSON de la tarea, texto libre o {"cambiar": {...}}
  Salida   /agente/estado  std_msgs/String  JSON con cada estado (aceptada, subtarea, completada, ...)

Ejemplo desde otra terminal:
  ros2 topic pub --once /agente/tarea std_msgs/String \
    "{data: '{\"perfil\": \"empaquetado_medicamentos\", \"parametros\": {\"items\": {\"Hidroferol\": 1}, \"destination\": \"brown box\"}}'}"
"""
from __future__ import annotations

import json
import queue

from ..entornos import ros_comun
from ..interfaces import FuenteTareas
from .comun import a_tarea


class FuenteROS2(FuenteTareas):
    def __init__(self, topico_tarea: str = "/agente/tarea", topico_estado: str = "/agente/estado",
                 perfil: str = "manipulacion_general", **_):
        from std_msgs.msg import String
        self._String = String
        self.cola: "queue.Queue" = queue.Queue()
        self.perfil = perfil
        self.nodo = ros_comun.crear_nodo("agente_interaccion")
        self.nodo.create_subscription(String, topico_tarea, lambda m: self.cola.put(m.data), 10)
        self.pub = self.nodo.create_publisher(String, topico_estado, 10)
        self.nodo.get_logger().info(f"Esperando tareas en {topico_tarea}; estados en {topico_estado}")

    def __iter__(self):
        import rclpy
        while rclpy.ok():
            try:
                dato = self.cola.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                yield a_tarea(dato, self.perfil)
            except Exception as exc:
                self.reportar({"estado": "rechazada", "detalle": f"tarea no valida: {exc}"})

    def reportar(self, estado):
        self.pub.publish(self._String(data=json.dumps(estado, ensure_ascii=False, default=str)))
