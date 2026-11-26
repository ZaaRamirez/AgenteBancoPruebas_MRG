"""Tipos de datos que circulan entre las capas del agente.

Son los mismos sin importar la tarea, el modelo o si el robot es simulado o fisico.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class Tarea:
    """Lo que recibe la capa de interaccion. ``perfil`` indica que perfil de tarea
    (config/tareas/) interpreta los ``parametros``; ``instruccion`` es opcional."""
    id: str
    perfil: str = "manipulacion_general"
    parametros: Dict[str, Any] = field(default_factory=dict)
    instruccion: str = ""
    solicitante: str = ""
    referencia: Dict[str, Any] = field(default_factory=dict)   # opcional: plan esperado (evaluacion)

    def como_dict(self):
        return asdict(self)


@dataclass
class Observacion:
    """Lo que entrega el entorno: imagenes por camara y estado articular en las
    unidades del vector de accion del VLA."""
    imagenes: Dict[str, np.ndarray] = field(default_factory=dict)
    estado: Optional[np.ndarray] = None
    tiempo: float = field(default_factory=time.time)
    info: Dict[str, Any] = field(default_factory=dict)

    def imagen(self, nombre: str = "") -> Optional[np.ndarray]:
        if nombre and nombre in self.imagenes:
            return self.imagenes[nombre]
        return next(iter(self.imagenes.values()), None)


@dataclass
class EspecificacionAccion:
    """Forma del vector de accion de la plataforma (lo define config/robots/*.yaml)."""
    dim: int
    indices_articulaciones: List[int]
    hz: float = 10.0
    nombres: List[str] = field(default_factory=list)


@dataclass
class Respuesta:
    """Respuesta de un modelo de lenguaje o vision ya procesada por su capa."""
    texto: str = ""
    datos: Any = None
    latencia_s: float = 0.0
    formato_ok: bool = False
    error: str = ""


@dataclass
class Decision:
    """Accion propuesta por la capa de razonamiento y su validacion."""
    accion: str = ""
    argumentos: Dict[str, Any] = field(default_factory=dict)
    pensamiento: str = ""
    texto: str = ""
    latencia_s: float = 0.0
    formato_ok: bool = False
    valida: bool = False
    motivo: str = ""


@dataclass
class ResultadoEjecucion:
    terminado: bool
    motivo: str
    pasos: int
    duracion_s: float
    latencia_media_s: float = 0.0


@dataclass
class ResultadoTarea:
    tarea_id: str
    estado: str            # completada | fallida | no_realizable | abortada
    exito: bool
    detalle: str = ""
    duracion_s: float = 0.0
    subtareas: int = 0
    reintentos: int = 0
    progreso: float = -1.0
    llamadas_llm: int = 0
    subtareas_exitosas: int = 0
    unidades_requeridas: int = -1       # segun el seguimiento (-1 = sin conteo)
