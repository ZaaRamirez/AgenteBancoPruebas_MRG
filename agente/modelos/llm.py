"""Modelos de la capa de RAZONAMIENTO (LLM). Los modelos NO viven en el banco de
pruebas: estas clases solo se conectan a donde ya estan (servidor Ollama, API
compatible con OpenAI, ...)."""
from __future__ import annotations

import json
from typing import Any, Dict, Optional

import requests

from ..interfaces import ModeloLenguaje


class OllamaLLM(ModeloLenguaje):
    """Modelo servido por Ollama (qwen3.8, deepseek-r1, gemma4, llama, ...)."""

    def __init__(self, model: str, url: str = "http://localhost:11434", temperature: float = 0.3,
                 max_tokens: int = 1024, timeout_s: float = 180.0, think: Optional[bool] = None,
                 keep_alive: str = "30m", seed: Optional[int] = None, num_ctx: Optional[int] = None, **_):
        self.nombre = model
        self.url = url.rstrip("/") + "/api/generate"
        self.opciones = {"temperature": float(temperature), "num_predict": int(max_tokens)}
        if seed is not None:
            self.opciones["seed"] = int(seed)
        if num_ctx:
            self.opciones["num_ctx"] = int(num_ctx)
        self.timeout_s, self.think, self.keep_alive = timeout_s, think, keep_alive
        self.ultimo: Dict[str, Any] = {}     # metadatos de la ultima llamada (evaluacion)

    def generar(self, prompt: str, esquema: Any = None, contexto: Optional[Dict] = None) -> str:
        p = {"model": self.nombre, "prompt": prompt, "stream": False, "options": self.opciones,
             "keep_alive": self.keep_alive}
        if esquema is not None:
            p["format"] = esquema
        if self.think is not None:
            p["think"] = self.think
        r = requests.post(self.url, json=p, timeout=self.timeout_s)
        r.raise_for_status()
        d = r.json()
        self.ultimo = metadatos_ollama(d)
        return d.get("response", "")


def metadatos_ollama(d: Dict[str, Any]) -> Dict[str, Any]:
    """Tokens, tiempos y motivo de fin que reporta Ollama en cada respuesta."""
    ns = lambda k: (d.get(k) or 0) / 1e9
    return {"tokens_prompt": d.get("prompt_eval_count"), "tokens_respuesta": d.get("eval_count"),
            "tiempo_generacion_s": ns("eval_duration") or None, "tiempo_prompt_s": ns("prompt_eval_duration") or None,
            "tiempo_carga_s": ns("load_duration") or None, "motivo_fin": d.get("done_reason"),
            "truncado": d.get("done_reason") == "length", "largo_razonamiento": len(d.get("thinking") or "")}


class OpenAILLM(ModeloLenguaje):
    """Servidor con API compatible con OpenAI (vLLM, llama.cpp, LM Studio, ...)."""

    def __init__(self, model: str, url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 temperature: float = 0.3, max_tokens: int = 1024, timeout_s: float = 180.0, **_):
        self.nombre = model
        self.ultimo: Dict[str, Any] = {}
        self.url = url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.temperature, self.max_tokens, self.timeout_s = temperature, max_tokens, timeout_s

    def generar(self, prompt, esquema=None, contexto=None) -> str:
        p = {"model": self.nombre, "temperature": self.temperature, "max_tokens": self.max_tokens,
             "messages": [{"role": "user", "content": prompt}]}
        if isinstance(esquema, dict):
            p["response_format"] = {"type": "json_schema", "json_schema": {"name": "salida", "schema": esquema}}
        elif esquema == "json":
            p["response_format"] = {"type": "json_object"}
        r = requests.post(self.url, json=p, headers=self.headers, timeout=self.timeout_s)
        r.raise_for_status()
        d = r.json()
        u, ch = d.get("usage") or {}, d["choices"][0]
        self.ultimo = {"tokens_prompt": u.get("prompt_tokens"), "tokens_respuesta": u.get("completion_tokens"),
                       "motivo_fin": ch.get("finish_reason"), "truncado": ch.get("finish_reason") == "length",
                       "largo_razonamiento": len(ch["message"].get("reasoning_content") or "")}
        return ch["message"]["content"] or ""


class GuionLLM(ModeloLenguaje):
    """Razonamiento determinista SIN modelo: linea base y pruebas sin GPU.
    - Con seguimiento por conteo: elige el primer elemento pendiente.
    - Sin conteo: sigue ``plan_de_prueba`` del perfil y luego declara ``done``."""

    def __init__(self, model: str = "guion", **_):
        self.nombre = model

    def generar(self, prompt, esquema=None, contexto=None) -> str:
        c = contexto or {}
        if c.get("respuesta"):            # benchmark: respuesta correcta conocida (oraculo)
            return c["respuesta"]
        pendientes, arg, habilidades = c.get("pendientes") or {}, c.get("argumento"), c.get("habilidades") or []
        if pendientes and arg and habilidades:
            obj = next(iter(pendientes))
            return json.dumps({"thought": f"Next pending element: {obj}.", "action": habilidades[0],
                               "arguments": {arg: obj}})
        plan, k = c.get("plan") or [], int(c.get("hechas", 0))
        if not pendientes and c.get("completa") is not True and k < len(plan):
            paso = plan[k]
            return json.dumps({"thought": f"Scripted step {k + 1}.", "action": paso["action"],
                               "arguments": paso.get("arguments", {})})
        return json.dumps({"thought": "All subtasks are done.", "action": "done", "arguments": {}})
