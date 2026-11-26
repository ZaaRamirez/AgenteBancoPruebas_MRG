"""Entorno simulado en memoria: prueba la arquitectura sin ROS, MuJoCo ni robot.

El estado sigue a la accion (primer orden) y la camara es una imagen sintetica o
un archivo fijo (p. ej. un cuadro del conjunto de datos, para probar el VLM).
"""
from __future__ import annotations

import time

import numpy as np

from ..interfaces import Entorno
from ..tipos import Observacion
from ..util.imagen import leer_rgb
from .robot import cargar_robot


class EntornoSimulado(Entorno):
    def __init__(self, robot: str = "omx", tiempo_real: bool = False, imagen: str = "", ganancia: float = 0.5, **_):
        self.robot = cargar_robot(robot)
        self.especificacion = self.robot.especificacion()
        self.tiempo_real = bool(tiempo_real)
        self.ganancia = float(ganancia)
        self.estado = np.zeros(self.especificacion.dim, dtype=np.float32)
        self.imagen_fija = leer_rgb(imagen) if imagen else None
        self.camaras = list(self.robot.camaras) or ["front"]
        self._t = time.time()

    def observar(self) -> Observacion:
        if self.imagen_fija is not None:
            img = self.imagen_fija
        else:
            img = np.full((120, 160, 3), 210, np.uint8)
            x = int(80 + 60 * np.tanh(float(self.estado.sum()) / 100.0))
            img[50:70, max(0, x - 10):x + 10] = (200, 40, 40)
        return Observacion(imagenes={c: img for c in self.camaras}, estado=self.estado.copy(),
                           info={"entorno": "simulado"})

    def actuar(self, accion):
        a = np.asarray(accion, dtype=np.float32)[: len(self.estado)]
        self.estado[: len(a)] += self.ganancia * (a - self.estado[: len(a)])

    def esperar_ciclo(self):
        if self.tiempo_real:
            dt = 1.0 / self.especificacion.hz - (time.time() - self._t)
            if dt > 0:
                time.sleep(dt)
            self._t = time.time()
