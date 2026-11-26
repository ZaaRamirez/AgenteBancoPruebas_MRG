"""Capa de PERCEPCION (VLM): los "ojos" del agente.

Responde cualquier pregunta en lenguaje natural sobre la observacion actual
(observacion inicial, verificacion de subtareas, revision final, preguntas del
LLM). Si la pregunta trae un esquema, la respuesta se valida contra el.
"""
from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from ..tipos import Observacion, Respuesta
from ..util.texto import extraer_json, validar_esquema


class CapaPercepcion:
    def __init__(self, vlm, registro=None, salida_estructurada: str = "json", camaras: Optional[List[str]] = None):
        self.vlm = vlm
        self.registro = registro
        self.salida = salida_estructurada          # none | json | schema
        self.camaras = camaras or []               # vacio = todas las camaras del entorno

    @property
    def modelo(self) -> str:
        return getattr(self.vlm, "etiqueta", getattr(self.vlm, "nombre", "?"))

    def _imagenes(self, obs: Observacion):
        if self.camaras:
            return [obs.imagenes[c] for c in self.camaras if c in obs.imagenes]
        return list(obs.imagenes.values())

    def preguntar(self, proposito: str, pregunta: str, obs: Observacion, esquema: Any = None,
                  respuesta_simulada: str = "", contexto_extra: Optional[Dict] = None) -> Respuesta:
        imgs = self._imagenes(obs)
        if esquema is not None:      # el formato pedido forma parte de la pregunta
            pregunta = (f"{pregunta.rstrip()}\nRespond ONLY with a JSON object that follows this JSON schema:\n"
                        f"{json.dumps(esquema, ensure_ascii=False)}")
        if esquema is None:          # pregunta libre del LLM: respuesta en texto
            fmt = None
        else:
            fmt = esquema if (self.salida == "schema" and isinstance(esquema, dict)) else \
                ("json" if self.salida in ("json", "schema") else None)
        r = Respuesta()
        t0 = time.perf_counter()
        try:
            r.texto = self.vlm.consultar(pregunta, imgs, esquema=fmt,
                                         contexto={"proposito": proposito, "respuesta_simulada": respuesta_simulada,
                                                  "esquema": esquema, **(contexto_extra or {})})
        except Exception as exc:
            r.error = str(exc)
        r.latencia_s = time.perf_counter() - t0
        r.datos = extraer_json(r.texto) if (r.texto and esquema is not None) else None
        errores = validar_esquema(r.datos, esquema) if (r.datos is not None and esquema) else []
        r.formato_ok = (r.datos is not None and not errores) if esquema is not None else bool(r.texto.strip())
        if self.registro:
            img = self.registro.imagen(imgs[0], proposito) if imgs else ""
            self.registro.evento("vlm", capa="percepcion", modelo=self.modelo, proposito=proposito,
                                 pregunta=pregunta, respuesta=r.texto, datos=r.datos, latencia_s=r.latencia_s,
                                 formato_ok=r.formato_ok, errores_esquema=errores, error=r.error,
                                 n_imagenes=len(imgs), imagen=img)
        return r
