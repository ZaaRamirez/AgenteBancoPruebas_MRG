#!/usr/bin/env python3
"""Robot ficticio (nodo ROS 2) para probar la conexion del agente sin MuJoCo ni hardware.

Publica las camaras y /joint_states de config/robots/<robot>.yaml y mueve sus
articulaciones hacia los JointTrajectory / GripperCommand que recibe: es decir,
se comporta como el gemelo o el driver fisico desde el punto de vista de los mensajes.

  python3 herramientas/robot_ficticio.py omx
"""
import os
import sys

import numpy as np
import rclpy
from rclpy.action import ActionServer
from rclpy.node import Node
from sensor_msgs.msg import Image, JointState
from trajectory_msgs.msg import JointTrajectory

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from agente.entornos.robot import cargar_robot  # noqa: E402

try:
    from control_msgs.action import GripperCommand
except ImportError:
    GripperCommand = None


class RobotFicticio(Node):
    def __init__(self, robot):
        super().__init__("robot_ficticio")
        self.r = cargar_robot(robot)
        self.pos, self.obj = {}, {}
        for b in self.r.brazos:
            for j in b.articulaciones:
                self.pos[j] = self.obj[j] = 0.0
            if b.trayectoria:
                self.create_subscription(JointTrajectory, b.trayectoria, self.cb_tray, 10)
            if b.pinza:
                self.pos[b.pinza.articulacion] = self.obj[b.pinza.articulacion] = 0.0
                if GripperCommand:
                    ActionServer(self, GripperCommand, b.pinza.accion,
                                 lambda g, j=b.pinza.articulacion: self.cb_pinza(g, j))
        self.pub_js = self.create_publisher(JointState, self.r.estado_articular, 10)
        self.pub_img = {c: self.create_publisher(Image, t, 10) for c, t in self.r.camaras.items()}
        self.create_timer(1 / 30, self.paso)
        self.get_logger().info(f"Robot ficticio '{self.r.nombre}' publicando {list(self.r.camaras.values())} "
                               f"y {self.r.estado_articular}")

    def cb_tray(self, m):
        if m.points:
            for n, p in zip(m.joint_names, m.points[-1].positions):
                self.obj[n] = p

    def cb_pinza(self, g, j):
        self.obj[j] = g.request.command.position
        g.succeed()
        res = GripperCommand.Result()
        res.position, res.reached_goal = self.obj[j], True
        return res

    def paso(self):
        for j in self.pos:
            self.pos[j] += 0.25 * (self.obj[j] - self.pos[j])
        js = JointState()
        js.header.stamp = self.get_clock().now().to_msg()
        js.name, js.position = list(self.pos), [float(v) for v in self.pos.values()]
        self.pub_js.publish(js)
        img = np.full((120, 160, 3), 200, np.uint8)
        x = int(80 + 60 * np.tanh(sum(self.pos.values())))
        img[50:70, max(0, x - 10):x + 10] = (200, 30, 30)
        for pub in self.pub_img.values():
            m = Image()
            m.header.stamp = js.header.stamp
            m.height, m.width, m.encoding, m.step = 120, 160, "rgb8", 160 * 3
            m.data = img.tobytes()
            pub.publish(m)


def main():
    rclpy.init()
    n = RobotFicticio(sys.argv[1] if len(sys.argv) > 1 else "omx")
    try:
        rclpy.spin(n)
    except KeyboardInterrupt:
        pass
    finally:
        n.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
