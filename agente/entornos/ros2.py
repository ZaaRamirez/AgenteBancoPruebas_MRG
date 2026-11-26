"""Conexion del agente con la plataforma a traves de ROS 2.

Es el UNICO nodo ROS que necesita el agente para actuar. Usa solo mensajes
estandar, de modo que funciona igual con el gemelo en MuJoCo y con el robot fisico
si ambos publican/aceptan:
    sensor_msgs/Image                  camaras            (entrada)
    sensor_msgs/JointState             estado articular   (entrada)
    trajectory_msgs/JointTrajectory    brazos             (salida, por topico)
    control_msgs/FollowJointTrajectory brazos             (salida, por accion; opcional)
    control_msgs/GripperCommand        pinzas             (salida, por accion)
Los topicos y nombres de articulacion salen de config/robots/<robot>.yaml.
"""
from __future__ import annotations

import threading
import time

import numpy as np

from ..interfaces import Entorno
from ..tipos import Observacion
from ..util.imagen import ros_a_rgb
from . import ros_comun
from .robot import cargar_robot


class EntornoROS2(Entorno):
    def __init__(self, robot: str = "omx", usar_tiempo_sim: bool = False, espera_datos_s: float = 15.0,
                 nombre_nodo: str = "agente_conexion_robot", **_):
        self.robot = cargar_robot(robot)
        self.especificacion = self.robot.especificacion()
        self.usar_tiempo_sim = bool(usar_tiempo_sim)
        self.espera_datos_s = float(espera_datos_s)
        self.nombre_nodo = nombre_nodo
        self._lock = threading.Lock()
        self._imagenes, self._posiciones, self._ultimo_js = {}, {}, 0.0
        self._ultima_pinza = {}
        self._t = time.time()
        self.nodo = None

    # ------------------------------------------------------------------ conexion
    def conectar(self):
        from rclpy.action import ActionClient
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import Image, JointState
        from trajectory_msgs.msg import JointTrajectory
        try:
            from control_msgs.action import FollowJointTrajectory, GripperCommand
        except ImportError:
            FollowJointTrajectory = GripperCommand = None
        self._GripperCommand, self._FJT = GripperCommand, FollowJointTrajectory
        n = self.nodo = ros_comun.crear_nodo(self.nombre_nodo, self.usar_tiempo_sim)
        for cam, topico in self.robot.camaras.items():
            n.create_subscription(Image, topico, lambda m, c=cam: self._cb_imagen(c, m), qos_profile_sensor_data)
        n.create_subscription(JointState, self.robot.estado_articular, self._cb_js, 20)
        self._pub, self._cli_brazo, self._cli_pinza = {}, {}, {}
        for b in self.robot.brazos:
            if b.trayectoria:
                self._pub[b.nombre] = n.create_publisher(JointTrajectory, b.trayectoria, 10)
            if b.modo_comando == "accion" and FollowJointTrajectory and b.accion_trayectoria:
                self._cli_brazo[b.nombre] = ActionClient(n, FollowJointTrajectory, b.accion_trayectoria)
            if b.pinza and GripperCommand:
                self._cli_pinza[b.nombre] = ActionClient(n, GripperCommand, b.pinza.accion)
            elif b.pinza:
                n.get_logger().warn("control_msgs no instalado: la pinza no se comandara")
        self._esperar_datos()

    def _esperar_datos(self):
        fin = time.time() + self.espera_datos_s
        while time.time() < fin:
            with self._lock:
                listo = self._posiciones and all(c in self._imagenes for c in self.robot.camaras)
            if listo:
                return
            time.sleep(0.1)
        with self._lock:
            faltan = [self.robot.camaras[c] for c in self.robot.camaras if c not in self._imagenes]
            if not self._posiciones:
                faltan.append(self.robot.estado_articular)
        raise TimeoutError(f"Sin datos del robot tras {self.espera_datos_s:.0f} s en: {faltan}. "
                           "¿Esta corriendo la simulacion o el driver? (ros2 topic list)")

    def cerrar(self):
        ros_comun.apagar()

    # ------------------------------------------------------------------ entradas
    def _cb_imagen(self, cam, msg):
        try:
            img = ros_a_rgb(msg)
        except ValueError as exc:
            self.nodo.get_logger().warn(str(exc), throttle_duration_sec=5.0)
            return
        with self._lock:
            self._imagenes[cam] = img

    def _cb_js(self, msg):
        with self._lock:
            self._posiciones.update(dict(zip(msg.name, msg.position)))
            self._ultimo_js = time.time()

    def observar(self) -> Observacion:
        with self._lock:
            imgs = dict(self._imagenes)
            pos = dict(self._posiciones)
            edad = time.time() - self._ultimo_js
        return Observacion(imagenes=imgs, estado=self.robot.vector_estado(pos),
                           info={"entorno": "ros2", "edad_estado_s": edad, "posiciones": pos})

    # ------------------------------------------------------------------ salidas
    def actuar(self, accion):
        from builtin_interfaces.msg import Duration, Time
        from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
        a = np.asarray(accion, dtype=np.float32)
        for b in self.robot.brazos:
            i0, i1 = b.indices_accion
            if i1 <= len(a):
                tr = JointTrajectory()
                tr.joint_names = list(b.articulaciones)
                pt = JointTrajectoryPoint()
                pt.positions = b.a_rad(a[i0:i1])
                pt.velocities = [0.0] * len(pt.positions)
                pt.accelerations = [0.0] * len(pt.positions)
                pt.time_from_start = Duration(sec=0, nanosec=b.duracion_punto_ms * 1_000_000)
                tr.points = [pt]
                cli = self._cli_brazo.get(b.nombre)
                if cli is not None and cli.server_is_ready():
                    tr.header.stamp = self.nodo.get_clock().now().to_msg()
                    meta = self._FJT.Goal()
                    meta.trajectory = tr
                    meta.goal_time_tolerance = Duration(sec=1, nanosec=0)
                    cli.send_goal_async(meta)
                elif b.nombre in self._pub:
                    tr.header.stamp = Time(sec=0, nanosec=0)    # modo reactivo del controlador
                    self._pub[b.nombre].publish(tr)
            cli = self._cli_pinza.get(b.nombre)
            if cli is not None and b.pinza.indice_accion < len(a):
                cmd = b.pinza.a_comando(float(a[b.pinza.indice_accion]))
                if abs(cmd - self._ultima_pinza.get(b.nombre, 1e9)) >= 0.05 and cli.server_is_ready():
                    meta = self._GripperCommand.Goal()
                    meta.command.position = float(cmd)
                    meta.command.max_effort = float(b.pinza.esfuerzo_max)
                    cli.send_goal_async(meta)
                    self._ultima_pinza[b.nombre] = cmd

    def esperar_ciclo(self):
        dt = 1.0 / self.especificacion.hz - (time.time() - self._t)
        if dt > 0:
            time.sleep(dt)
        self._t = time.time()
