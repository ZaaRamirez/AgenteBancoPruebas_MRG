"""Proceso ROS 2 compartido: una sola inicializacion de rclpy y un ejecutor en un
hilo de fondo para todos los nodos de conexion (robot, fuente de tareas)."""
from __future__ import annotations

import threading

_estado = {"ejecutor": None, "hilo": None, "nodos": []}
_lock = threading.Lock()


def crear_nodo(nombre: str, usar_tiempo_sim: bool = False):
    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from rclpy.parameter import Parameter
    with _lock:
        if not rclpy.ok():
            # Sin manejadores de senales de ROS: Ctrl+C y SIGTERM detienen el agente como
            # cualquier programa de Python (el agente no es un nodo, solo los usa).
            try:
                from rclpy.signals import SignalHandlerOptions
                rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
            except (ImportError, TypeError):     # versiones viejas de rclpy
                rclpy.init()
        if _estado["ejecutor"] is None:
            _estado["ejecutor"] = MultiThreadedExecutor(num_threads=4)
            _estado["hilo"] = threading.Thread(target=_girar, daemon=True)
            _estado["hilo"].start()
        nodo = rclpy.create_node(nombre, parameter_overrides=[
            Parameter("use_sim_time", Parameter.Type.BOOL, bool(usar_tiempo_sim))])
        _estado["ejecutor"].add_node(nodo)
        _estado["nodos"].append(nodo)
        return nodo


def _girar():
    try:
        _estado["ejecutor"].spin()
    except Exception:
        pass


def apagar():
    import rclpy
    with _lock:
        for n in _estado["nodos"]:
            try:
                _estado["ejecutor"].remove_node(n)
                n.destroy_node()
            except Exception:
                pass
        _estado["nodos"].clear()
        if _estado["ejecutor"] is not None:
            _estado["ejecutor"].shutdown()
            _estado["ejecutor"] = None
        if rclpy.ok():
            rclpy.shutdown()
