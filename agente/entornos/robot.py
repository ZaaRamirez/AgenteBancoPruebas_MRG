"""Descripcion de la plataforma (config/robots/<robot>.yaml).

Define que camaras hay, que articulaciones forman el vector de accion del VLA,
sus unidades, la pinza y los topicos ROS 2. El mismo archivo sirve para el gemelo
en MuJoCo y para el robot fisico mientras ambos usen los mismos mensajes.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..tipos import EspecificacionAccion
from ..util.config import cargar_yaml


@dataclass
class Pinza:
    articulacion: str
    indice_accion: int
    rango_accion: Sequence[float] = (0.0, 80.0)     # unidades del conjunto de datos
    rango_comando: Sequence[float] = (0.0, 2.0)     # unidades del controlador
    accion: str = "/gripper_controller/gripper_cmd"
    esfuerzo_max: float = 100.0

    def a_comando(self, a: float) -> float:
        lo, hi = self.rango_accion
        clo, chi = self.rango_comando
        return clo + (min(max(a, lo), hi) - lo) / (hi - lo) * (chi - clo)

    def a_unidades_accion(self, pos: float) -> float:
        lo, hi = self.rango_accion
        clo, chi = self.rango_comando
        return lo + (min(max(pos, clo), chi) - clo) / (chi - clo) * (hi - lo)


@dataclass
class Brazo:
    nombre: str
    articulaciones: List[str]
    indices_accion: Sequence[int]                   # [inicio, fin) en el vector de accion
    trayectoria: str = ""
    unidades: str = "grados"                        # grados | radianes
    duracion_punto_ms: int = 150
    modo_comando: str = "topico"                    # topico | accion
    accion_trayectoria: str = ""
    pinza: Optional[Pinza] = None

    def a_rad(self, valores) -> List[float]:
        return [math.radians(float(v)) if self.unidades == "grados" else float(v) for v in valores]

    def desde_rad(self, valores) -> List[float]:
        return [math.degrees(float(v)) if self.unidades == "grados" else float(v) for v in valores]


@dataclass
class Robot:
    nombre: str
    brazos: List[Brazo]
    camaras: Dict[str, str] = field(default_factory=dict)      # nombre -> topico
    estado_articular: str = "/joint_states"
    hz: float = 10.0

    @property
    def dim(self) -> int:
        d = 0
        for b in self.brazos:
            d = max(d, b.indices_accion[1], (b.pinza.indice_accion + 1) if b.pinza else 0)
        return d

    def especificacion(self) -> EspecificacionAccion:
        idx, nombres = [], [""] * self.dim
        for b in self.brazos:
            i0, i1 = b.indices_accion
            idx.extend(range(i0, i1))
            for k, j in zip(range(i0, i1), b.articulaciones):
                nombres[k] = j
            if b.pinza:
                nombres[b.pinza.indice_accion] = b.pinza.articulacion
        return EspecificacionAccion(dim=self.dim, indices_articulaciones=idx, hz=self.hz, nombres=nombres)

    def vector_estado(self, posiciones: Dict[str, float]) -> np.ndarray:
        """Posiciones de /joint_states (rad) -> vector en unidades del VLA."""
        s = np.zeros(self.dim, dtype=np.float32)
        for b in self.brazos:
            i0, i1 = b.indices_accion
            s[i0:i1] = b.desde_rad([posiciones.get(j, 0.0) for j in b.articulaciones])
            if b.pinza and b.pinza.articulacion in posiciones:
                s[b.pinza.indice_accion] = b.pinza.a_unidades_accion(posiciones[b.pinza.articulacion])
        return s


def cargar_robot(nombre: str) -> Robot:
    d = cargar_yaml(nombre, "robots")
    brazos = []
    for b in d["brazos"]:
        p = b.get("pinza")
        brazos.append(Brazo(
            nombre=b["nombre"], articulaciones=list(b["articulaciones"]), indices_accion=list(b["indices_accion"]),
            trayectoria=b.get("trayectoria", ""), unidades=b.get("unidades", "grados"),
            duracion_punto_ms=int(b.get("duracion_punto_ms", 150)), modo_comando=b.get("modo_comando", "topico"),
            accion_trayectoria=b.get("accion_trayectoria", ""), pinza=Pinza(**p) if p else None))
    return Robot(nombre=d.get("nombre", nombre), brazos=brazos, camaras=dict(d.get("camaras", {})),
                 estado_articular=d.get("estado_articular", "/joint_states"), hz=float(d.get("hz", 10.0)))
