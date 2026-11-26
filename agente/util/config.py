"""Carga de configuracion: raiz del proyecto, rutas externas (config/rutas.env),
archivos YAML con ${VARIABLES} y sobreescrituras ``a.b.c=valor`` desde la terminal."""
from __future__ import annotations

import copy
import os
import re
from typing import Any, Dict, Iterable

import yaml

RAIZ = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG = os.path.join(RAIZ, "config")
_LINEA = re.compile(r'^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$')


def cargar_rutas(archivo: str = "") -> Dict[str, str]:
    """Lee config/rutas.env (formato bash VAR="valor") y lo agrega a os.environ
    sin pisar variables ya definidas en la terminal."""
    archivo = archivo or os.path.join(CONFIG, "rutas.env")
    leidas: Dict[str, str] = {}
    if not os.path.exists(archivo):
        return leidas
    with open(archivo, encoding="utf-8") as f:
        for linea in f:
            m = _LINEA.match(linea)
            if not m or linea.strip().startswith("#"):
                continue
            valor = m.group(2).split(" #")[0].strip().strip('"').strip("'")
            valor = os.path.expanduser(os.path.expandvars(valor))
            leidas[m.group(1)] = valor
            os.environ.setdefault(m.group(1), valor)
    return leidas


def expandir(v):
    if isinstance(v, str):
        return os.path.expanduser(os.path.expandvars(v))
    if isinstance(v, dict):
        return {k: expandir(x) for k, x in v.items()}
    if isinstance(v, list):
        return [expandir(x) for x in v]
    return v


def ruta(nombre: str, carpeta: str = "", extension: str = ".yaml") -> str:
    """Resuelve un nombre corto (``omx``) o una ruta a un archivo existente.
    Busca: ruta tal cual -> relativa a la raiz -> config/<carpeta>/<nombre>.yaml"""
    if not nombre:
        return ""
    candidatos = [os.path.expanduser(nombre), os.path.join(RAIZ, nombre)]
    if carpeta:
        candidatos += [os.path.join(CONFIG, carpeta, nombre),
                       os.path.join(CONFIG, carpeta, nombre + extension)]
    for c in candidatos:
        if os.path.isfile(c):
            return os.path.abspath(c)
    raise FileNotFoundError(f"No encuentro '{nombre}' (busque en {candidatos})")


def cargar_yaml(nombre: str, carpeta: str = "") -> Dict[str, Any]:
    with open(ruta(nombre, carpeta), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def fijar(cfg: Dict[str, Any], clave: str, valor: Any) -> None:
    """fijar(cfg, "modelos.llm", "qwen3.8") ; el valor se interpreta como YAML."""
    if isinstance(valor, str):
        try:
            valor = yaml.safe_load(valor)
        except yaml.YAMLError:
            pass
    partes = clave.split(".")
    d = cfg
    for p in partes[:-1]:
        d = d.setdefault(p, {})
    d[partes[-1]] = valor


def aplicar(cfg: Dict[str, Any], asignaciones: Iterable[str]) -> Dict[str, Any]:
    cfg = copy.deepcopy(cfg)
    for a in asignaciones or []:
        if "=" not in a:
            raise ValueError(f"Se esperaba clave=valor y llego '{a}'")
        k, v = a.split("=", 1)
        fijar(cfg, k.strip(), v.strip())
    return cfg
