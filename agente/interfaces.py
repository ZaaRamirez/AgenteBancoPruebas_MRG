"""Contratos de la arquitectura (Seccion III del articulo).

Cada capa del agente depende SOLO de estas interfaces. Para evaluar un modelo nuevo
se escribe una clase que cumpla el contrato de su capa y se registra en
config/modelos.yaml; para conectar otro robot se escribe un Entorno. El ciclo del
agente (agente/agente.py) no cambia.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Optional, Union

import numpy as np

from .tipos import EspecificacionAccion, Observacion, Tarea


# ----------------------------------------------------------------- modelos por capa
class ModeloLenguaje(ABC):
    """Capa de RAZONAMIENTO (LLM): texto -> texto."""
    nombre: str = ""

    @abstractmethod
    def generar(self, prompt: str, esquema: Any = None, contexto: Optional[Dict] = None) -> str:
        """``esquema``: None, "json" o un JSON Schema para salida estructurada.
        ``contexto``: datos auxiliares (solo los usan los modelos de prueba)."""


class ModeloVisionLenguaje(ABC):
    """Capa de PERCEPCION (VLM): imagenes + pregunta -> texto."""
    nombre: str = ""

    @abstractmethod
    def consultar(self, prompt: str, imagenes: List[np.ndarray], esquema: Any = None,
                  contexto: Optional[Dict] = None) -> str:
        ...


class PoliticaVLA(ABC):
    """Capa de EJECUCION (VLA): observacion + instruccion -> accion (vector)."""
    nombre: str = ""

    def reiniciar(self) -> None:
        """Se llama al inicio de cada subtarea."""

    @abstractmethod
    def predecir(self, obs: Observacion, instruccion: str) -> np.ndarray:
        ...


# ----------------------------------------------------------------- conexion con el mundo
class Entorno(ABC):
    """Conexion del agente con la plataforma. Es lo UNICO que cambia entre el
    simulador y el robot fisico; con ROS 2 ambos usan los mismos mensajes."""
    especificacion: EspecificacionAccion

    def conectar(self) -> None:
        ...

    def cerrar(self) -> None:
        ...

    @abstractmethod
    def observar(self) -> Observacion:
        ...

    @abstractmethod
    def actuar(self, accion: np.ndarray) -> None:
        ...

    def esperar_ciclo(self) -> None:
        """Mantiene la frecuencia de control (no hace nada en simulacion rapida)."""


class FuenteTareas(ABC):
    """De donde llegan las tareas (archivo, terminal, topico ROS 2, ...).
    Puede entregar tambien comandos de cambio de modelo: {"cambiar": {"llm": "deepseek-r1"}}."""

    @abstractmethod
    def __iter__(self) -> Iterator[Union[Tarea, Dict[str, Any]]]:
        ...

    def reportar(self, estado: Dict[str, Any]) -> None:
        """Recibe los estados de la tarea (aceptada, subtarea, completada, ...)."""

    def cerrar(self) -> None:
        ...
