"""Almacen de resultados del benchmark: una carpeta por corrida, un JSONL por tipo de
registro, y REANUDACION (si una corrida larga se interrumpe, al relanzarla con la
misma carpeta se saltan los casos ya hechos)."""
from __future__ import annotations

import json
import os
import platform
import subprocess
import threading
import time
from typing import Any, Dict, Iterable, Optional, Set, Tuple

import numpy as np
import yaml

from agente.util.config import RAIZ

BASE = os.path.join(RAIZ, "resultados", "benchmark")


def _ser(v):
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, (np.floating, np.integer)):
        return v.item()
    if isinstance(v, np.bool_):
        return bool(v)
    return str(v)


class Corrida:
    """resultados/benchmark/<capa>/<nombre>/  con config.yaml, entorno.yaml y *.jsonl"""

    def __init__(self, capa: str, nombre: str = "", carpeta: str = "", config: Optional[Dict] = None):
        if carpeta:
            self.dir = os.path.abspath(os.path.expanduser(carpeta))
        else:
            nombre = nombre or time.strftime("%Y%m%d_%H%M%S")
            self.dir = os.path.join(BASE, capa, nombre)
        os.makedirs(self.dir, exist_ok=True)
        self.capa = capa
        self._lock = threading.Lock()
        self._archivos: Dict[str, Any] = {}
        if config is not None:
            p = os.path.join(self.dir, "config.yaml")
            if os.path.exists(p):
                previa = yaml.safe_load(open(p, encoding="utf-8")) or {}
                if previa.get("huella") and previa.get("huella") != config.get("huella"):
                    print(f"[aviso] la configuracion cambio respecto a la corrida guardada en {self.dir}; "
                          "los casos nuevos se agregan, los viejos se conservan.")
            with open(p, "w", encoding="utf-8") as f:
                yaml.safe_dump(config, f, allow_unicode=True, sort_keys=False)
            with open(os.path.join(self.dir, "entorno.yaml"), "w", encoding="utf-8") as f:
                yaml.safe_dump(info_equipo(), f, allow_unicode=True, sort_keys=False)

    def ruta(self, nombre: str) -> str:
        return os.path.join(self.dir, nombre)

    def escribir(self, tipo: str, registro: Dict[str, Any]) -> None:
        with self._lock:
            if tipo not in self._archivos:
                self._archivos[tipo] = open(self.ruta(f"{tipo}.jsonl"), "a", encoding="utf-8")
            f = self._archivos[tipo]
            f.write(json.dumps(registro, ensure_ascii=False, default=_ser) + "\n")
            f.flush()

    def leer(self, tipo: str):
        return leer_jsonl(self.ruta(f"{tipo}.jsonl"))

    def hechos(self, tipo: str, claves: Iterable[str]) -> Set[Tuple]:
        claves = list(claves)
        return {tuple(r.get(k) for k in claves) for r in self.leer(tipo)}

    def cerrar(self):
        for f in self._archivos.values():
            f.close()
        self._archivos = {}


def leer_jsonl(p: str):
    if not os.path.exists(p):
        return []
    out = []
    with open(p, encoding="utf-8") as f:
        for linea in f:
            linea = linea.strip()
            if linea:
                try:
                    out.append(json.loads(linea))
                except json.JSONDecodeError:
                    pass          # linea cortada por una interrupcion
    return out


def info_equipo() -> Dict[str, Any]:
    """Datos del equipo para reportar bajo que condiciones se midio."""
    d: Dict[str, Any] = {"fecha": time.strftime("%Y-%m-%d %H:%M:%S"), "sistema": platform.platform(),
                         "python": platform.python_version(), "cpu": platform.processor() or platform.machine()}
    try:
        with open("/proc/meminfo") as f:
            d["ram_gb"] = round(int(f.readline().split()[1]) / 1e6, 1)
    except Exception:
        pass
    try:
        salida = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                                 "--format=csv,noheader"], capture_output=True, text=True, timeout=5).stdout.strip()
        if salida:
            d["gpu"] = salida.splitlines()
    except Exception:
        pass
    for mod in ("torch", "transformers", "lerobot"):
        try:
            m = __import__(mod)
            d[f"version_{mod}"] = getattr(m, "__version__", "?")
        except Exception:
            pass
    return d


def recursos_ollama(url: str, modelo: str) -> Dict[str, Any]:
    """Tamano en memoria y en VRAM del modelo cargado en Ollama (/api/ps)."""
    try:
        import requests
        for m in requests.get(url.rstrip("/") + "/api/ps", timeout=3).json().get("models", []):
            if m.get("name") == modelo or m.get("model") == modelo or m.get("name", "").split(":")[0] == modelo:
                return {"memoria_gb": round((m.get("size") or 0) / 1e9, 2),
                        "vram_gb": round((m.get("size_vram") or 0) / 1e9, 2),
                        "parametros": (m.get("details") or {}).get("parameter_size"),
                        "cuantizacion": (m.get("details") or {}).get("quantization_level")}
    except Exception:
        pass
    return {}


def huella(obj: Any) -> str:
    import hashlib
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:12]
