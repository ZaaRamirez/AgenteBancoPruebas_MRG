"""Catalogo de modelos (config/modelos.yaml) y carga perezosa con cache.

Cada capa elige su modelo por un nombre corto (``qwen3.8``, ``gemma4``,
``molmoact2_omx``). Tambien se acepta ``backend:modelo`` (p. ej. ``ollama:llama3.2:3b``)
sin tocar el catalogo, o una clase propia ``paquete.modulo:Clase`` como backend.
"""
from __future__ import annotations

import importlib
import json
import os
from typing import Any, Dict, Tuple

import yaml

from ..util.config import CONFIG, cargar_rutas, expandir

BACKENDS = {
    "llm": {"ollama": "agente.modelos.llm:OllamaLLM", "openai": "agente.modelos.llm:OpenAILLM",
            "guion": "agente.modelos.llm:GuionLLM"},
    "vlm": {"ollama": "agente.modelos.vlm:OllamaVLM", "openai": "agente.modelos.vlm:OpenAIVLM",
            "transformers": "agente.modelos.vlm:TransformersVLM", "simulado": "agente.modelos.vlm:SimuladoVLM",
            "escena": "agente.modelos.vlm:EscenaVLM"},
    "vla": {"lerobot": "agente.modelos.vla:LeRobotVLA", "postura": "agente.modelos.vla:PosturaVLA",
            "simulado": "agente.modelos.vla:SimuladoVLA"},
}
_CACHE: Dict[Tuple[str, str], Any] = {}


def leer_catalogo(archivo: str = "") -> Dict[str, Dict[str, Any]]:
    cargar_rutas()
    archivo = archivo or os.path.join(CONFIG, "modelos.yaml")
    with open(archivo, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolver(capa: str, nombre) -> Tuple[str, Dict[str, Any]]:
    """(ruta de la clase, argumentos) del modelo ``nombre`` de la ``capa``.
    ``nombre`` puede ser un nombre del catalogo, ``backend:modelo`` o un dict con la
    misma forma que una entrada del catalogo (modelo definido en un experimento)."""
    cat = leer_catalogo().get(capa, {}) or {}
    if isinstance(nombre, dict):
        entrada = expandir(dict(nombre))
        entrada.pop("etiqueta", None)
        backend = entrada.pop("backend")
    elif nombre in cat:
        entrada = expandir(dict(cat[nombre]))
        backend = entrada.pop("backend")
    else:
        backend, sep, resto = str(nombre).partition(":")
        if backend not in BACKENDS[capa]:
            raise KeyError(f"El modelo '{nombre}' no esta en la seccion '{capa}' de config/modelos.yaml. "
                           f"Disponibles: {sorted(cat)}. O usa '<backend>:<modelo>' con backend en "
                           f"{sorted(BACKENDS[capa])}")
        entrada = {("checkpoint" if capa == "vla" else "model"): resto} if resto else {}
        if backend == "ollama" and os.environ.get("OLLAMA_URL"):
            entrada["url"] = os.environ["OLLAMA_URL"]      # el servidor de config/rutas.env
    clase = BACKENDS[capa].get(backend, backend)      # permite "paquete.modulo:Clase"
    if ":" not in clase:
        raise KeyError(f"Backend '{backend}' desconocido para {capa}: {sorted(BACKENDS[capa])}")
    return clase, entrada


def etiqueta_de(nombre) -> str:
    if isinstance(nombre, dict):
        return str(nombre.get("etiqueta") or nombre.get("backend", "?"))
    return str(nombre)


def cargar(capa: str, nombre, usar_cache: bool = True, sobrescribir: Dict[str, Any] = None):
    """Instancia (o reutiliza) el modelo. Los modelos pesados se cargan una sola vez
    por proceso, asi se pueden comparar varios LLM sin recargar el VLA.
    ``sobrescribir`` reemplaza argumentos del catalogo (p. ej. la misma temperatura y
    max_tokens para todos los modelos de un benchmark)."""
    sob = {k: v for k, v in (sobrescribir or {}).items() if v is not None}
    clave = (capa, json.dumps(nombre, sort_keys=True) if isinstance(nombre, dict) else nombre,
             json.dumps(sob, sort_keys=True))
    if usar_cache and clave in _CACHE:
        return _CACHE[clave]
    clase, kwargs = resolver(capa, nombre)
    kwargs.update(sob)
    mod, cls = clase.split(":")
    modelo = getattr(importlib.import_module(mod), cls)(**kwargs)
    modelo.etiqueta = etiqueta_de(nombre)
    modelo.argumentos = {k: v for k, v in kwargs.items() if not isinstance(v, (dict, list)) or len(str(v)) < 300}
    _CACHE[clave] = modelo
    return modelo


def liberar(capa: str = "", nombre: str = "") -> None:
    for k in [k for k in _CACHE if (not capa or k[0] == capa) and (not nombre or k[1] == nombre)]:
        del _CACHE[k]
