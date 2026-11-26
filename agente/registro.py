"""Registro de eventos para la evaluacion.

Cada corrida crea resultados/<nombre>_<fecha>/ con:
    corrida.yaml       configuracion completa (modelos por capa, entorno, perfil, ...)
    eventos.jsonl      cada llamada a un modelo y cada paso del agente (entrada, salida,
                       latencia, validez), una linea JSON por evento
    tareas.jsonl       resultado de cada tarea (una linea por tarea)
    imagenes/          (opcional) imagen que vio el VLM en cada consulta
Los indicadores NO se calculan aqui: se calculan despues con evaluacion/evaluar.py,
asi se pueden definir indicadores nuevos y recalcularlos sobre corridas viejas.
"""
from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import asdict, is_dataclass
from typing import Any, Dict, Optional

import numpy as np
import yaml

from .util.config import RAIZ


def _serializable(v):
    if is_dataclass(v):
        return asdict(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    return str(v)


class Registro:
    def __init__(self, nombre: str = "corrida", carpeta: str = "", config: Optional[Dict] = None,
                 guardar_imagenes: bool = False, max_texto: int = 4000):
        base = carpeta or os.path.join(RAIZ, "resultados")
        self.dir = os.path.join(os.path.expanduser(base), f"{nombre}_{time.strftime('%Y%m%d_%H%M%S')}")
        os.makedirs(self.dir, exist_ok=True)
        self.nombre, self.guardar_imagenes, self.max_texto = nombre, guardar_imagenes, max_texto
        self.episodio = ""
        self.n = 0
        self._lock = threading.Lock()
        self._ev = open(os.path.join(self.dir, "eventos.jsonl"), "a", encoding="utf-8")
        self._ta = open(os.path.join(self.dir, "tareas.jsonl"), "a", encoding="utf-8")
        if config is not None:
            with open(os.path.join(self.dir, "corrida.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)

    def iniciar_episodio(self, tarea_id: str):
        self.episodio = tarea_id

    def evento(self, tipo: str, **datos: Any) -> None:
        with self._lock:
            self.n += 1
            for k, v in list(datos.items()):
                if isinstance(v, str) and len(v) > self.max_texto:
                    datos[k] = v[: self.max_texto] + "...[recortado]"
            linea = {"n": self.n, "t": round(time.time(), 4), "corrida": self.nombre,
                     "tarea": self.episodio, "tipo": tipo, **datos}
            self._ev.write(json.dumps(linea, ensure_ascii=False, default=_serializable) + "\n")
            self._ev.flush()

    def imagen(self, img: np.ndarray, etiqueta: str) -> str:
        if not self.guardar_imagenes or img is None:
            return ""
        from .util.imagen import guardar_png
        d = os.path.join(self.dir, "imagenes")
        os.makedirs(d, exist_ok=True)
        archivo = os.path.join(d, f"{self.episodio}_{self.n + 1:05d}_{etiqueta}.png")
        guardar_png(img, archivo)
        return os.path.relpath(archivo, self.dir)

    def tarea(self, resultado, **extra) -> None:
        with self._lock:
            d = asdict(resultado) if is_dataclass(resultado) else dict(resultado)
            d.update(extra)
            d.update({"t": round(time.time(), 4), "corrida": self.nombre})
            self._ta.write(json.dumps(d, ensure_ascii=False, default=_serializable) + "\n")
            self._ta.flush()

    def cerrar(self):
        self._ev.close()
        self._ta.close()
