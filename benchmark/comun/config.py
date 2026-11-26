"""Carga de configuraciones del benchmark (config/benchmark/*.yaml) y de modelos con
parametros comunes."""
from __future__ import annotations

import copy
import os
from typing import Any, Dict, List

from agente.modelos import catalogo
from agente.util.config import CONFIG, aplicar, cargar_rutas, cargar_yaml, expandir


def cargar(nombre: str, asignaciones: List[str] = None) -> Dict[str, Any]:
    cargar_rutas()
    cfg = cargar_yaml(nombre, "benchmark")
    return expandir(aplicar(cfg, asignaciones or []))


def vocabulario(nombre: str = "objetos") -> Dict[str, Any]:
    return cargar_yaml(nombre, "benchmark")


def especificacion(capa: str, nombre: str, sobrescribir: Dict[str, Any] = None) -> Dict[str, Any]:
    """Modelo del catalogo + parametros comunes como dict en linea (para el Agente)."""
    clase, kwargs = catalogo.resolver(capa, nombre)
    kwargs = dict(kwargs)
    kwargs.update({k: v for k, v in (sobrescribir or {}).items() if v is not None})
    return {"backend": clase, "etiqueta": nombre, **kwargs}


def modelo(capa: str, nombre: str, sobrescribir: Dict[str, Any] = None):
    return catalogo.cargar(capa, nombre, usar_cache=False, sobrescribir=sobrescribir)


def lista(x) -> List[str]:
    if isinstance(x, str):
        return [m.strip() for m in x.split(",") if m.strip()]
    return list(x or [])


def nombre_archivo(x: str) -> str:
    return str(x).replace(":", "_").replace("/", "_").replace(" ", "_")
