"""Modelos de la capa de PERCEPCION (VLM): los "ojos" del agente."""
from __future__ import annotations

import json
import random
from typing import Any, Dict, List, Optional

import numpy as np
import requests

from ..interfaces import ModeloVisionLenguaje
from ..util.imagen import png_base64
from ..util.texto import ejemplo_de_esquema


class OllamaVLM(ModeloVisionLenguaje):
    """VLM servido por Ollama (gemma4, qwen2.5vl, llava, ...)."""

    def __init__(self, model: str, url: str = "http://localhost:11434", temperature: float = 0.0,
                 max_tokens: int = 1024, timeout_s: float = 240.0, keep_alive: str = "30m", **_):
        self.nombre = model
        self.url = url.rstrip("/") + "/api/generate"
        self.opciones = {"temperature": float(temperature), "num_predict": int(max_tokens)}
        self.timeout_s, self.keep_alive = timeout_s, keep_alive

    def consultar(self, prompt, imagenes, esquema=None, contexto=None) -> str:
        p = {"model": self.nombre, "prompt": prompt, "stream": False, "options": self.opciones,
             "images": [png_base64(i) for i in imagenes], "keep_alive": self.keep_alive}
        if esquema is not None:
            p["format"] = esquema
        r = requests.post(self.url, json=p, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json().get("response", "")


class OpenAIVLM(ModeloVisionLenguaje):
    """VLM con API compatible con OpenAI (vLLM sirviendo Qwen-VL, Molmo, ...)."""

    def __init__(self, model: str, url: str = "http://localhost:8000/v1", api_key: str = "EMPTY",
                 temperature: float = 0.0, max_tokens: int = 1024, timeout_s: float = 240.0, **_):
        self.nombre = model
        self.url = url.rstrip("/") + "/chat/completions"
        self.headers = {"Authorization": f"Bearer {api_key}"}
        self.temperature, self.max_tokens, self.timeout_s = temperature, max_tokens, timeout_s

    def consultar(self, prompt, imagenes, esquema=None, contexto=None) -> str:
        contenido = [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{png_base64(i)}"}}
                     for i in imagenes] + [{"type": "text", "text": prompt}]
        p = {"model": self.nombre, "temperature": self.temperature, "max_tokens": self.max_tokens,
             "messages": [{"role": "user", "content": contenido}]}
        if esquema is not None:
            p["response_format"] = {"type": "json_object"}
        r = requests.post(self.url, json=p, headers=self.headers, timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"] or ""


class TransformersVLM(ModeloVisionLenguaje):
    """VLM local con Hugging Face transformers (p. ej. el backbone VLM de MolmoAct2),
    cargado igual que en eval_task_decomposition.py. Requiere torch y transformers
    en el entorno de Python activo."""

    def __init__(self, model: str, device: str = "auto", dtype: str = "bfloat16", max_tokens: int = 512, **_):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        self.torch, self.nombre, self.max_tokens = torch, model, max_tokens
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model, dtype=getattr(torch, dtype), trust_remote_code=True).to(self.device).eval()

    def consultar(self, prompt, imagenes, esquema=None, contexto=None) -> str:
        from PIL import Image
        pil = [Image.fromarray(i) for i in imagenes]
        texto = None
        if hasattr(self.processor, "apply_chat_template"):
            try:
                msgs = [{"role": "user", "content": [{"type": "image"} for _ in pil] + [{"type": "text", "text": prompt}]}]
                texto = self.processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            except Exception:
                texto = None
        texto = texto or ("<image>\n" * len(pil) + prompt)
        x = self.processor(text=texto, images=pil, return_tensors="pt")
        x = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v for k, v in x.items()}
        with self.torch.inference_mode():
            out = self.model.generate(**x, max_new_tokens=self.max_tokens, do_sample=False)
        tok = getattr(self.processor, "tokenizer", self.processor)
        return tok.decode(out[0, x["input_ids"].shape[-1]:], skip_special_tokens=True).strip()


class SimuladoVLM(ModeloVisionLenguaje):
    """Percepcion simulada: responde una instancia valida del esquema pedido.
    ``tasa_exito`` controla cuantas verificaciones (campo success) salen verdaderas;
    ``respuestas`` permite fijar la respuesta por proposito (p. ej. inventario)."""

    def __init__(self, model: str = "simulado", tasa_exito: float = 1.0, semilla: int = 0,
                 respuestas: Optional[Dict[str, Any]] = None, **_):
        self.nombre = model
        self.tasa = float(tasa_exito)
        self.semilla = semilla
        self.rng = random.Random(semilla)
        self.respuestas = respuestas or {}

    def reiniciar_semilla(self):
        """Misma secuencia de aciertos y fallos para cada modelo comparado."""
        self.rng = random.Random(self.semilla)

    def consultar(self, prompt, imagenes, esquema=None, contexto=None) -> str:
        c = contexto or {}
        fija = self.respuestas.get(c.get("proposito", "")) or c.get("respuesta_simulada")
        if fija:
            return fija if isinstance(fija, str) else json.dumps(fija, ensure_ascii=False)
        sch = c.get("esquema") if isinstance(c.get("esquema"), dict) else esquema
        sch = sch if isinstance(sch, dict) else {"type": "object", "properties": {
            "success": {"type": "boolean"}, "reason": {"type": "string"}}}
        return json.dumps(ejemplo_de_esquema(sch, self.rng, self.tasa), ensure_ascii=False)


class EscenaVLM(ModeloVisionLenguaje):
    """Oraculo de percepcion para evaluar la capa de RAZONAMIENTO aislada.

    Lleva una escena simbolica coherente por tarea (que hay en el estante o la mesa,
    que hay en el destino) y responde con la verdad a TODAS las preguntas: inventario
    inicial, preguntas libres del LLM (observe), verificaciones y revision final.
    Asi, si el agente falla, el error es del LLM y no de una percepcion incoherente.

    tasa_exito        probabilidad de que una subtarea quede bien hecha (fallos inyectados)
    semilla           los fallos dependen de (semilla, tarea, intento): todos los modelos
                      comparados reciben exactamente los mismos fallos
    existencias_extra unidades de mas en el estante para cada elemento pedido
    distractores      elementos que estan en la escena pero NO se pidieron
    """

    def __init__(self, model: str = "escena", tasa_exito: float = 1.0, semilla: int = 0,
                 existencias_extra: int = 2, distractores: Optional[List[str]] = None, **_):
        self.nombre = model
        self.tasa = float(tasa_exito)
        self.semilla = semilla
        self.extra = int(existencias_extra)
        self.distractores = list(distractores or [])
        self.escenas: Dict[str, Dict[str, Any]] = {}

    def reiniciar_semilla(self):
        self.escenas = {}

    @staticmethod
    def _n(x):
        return " ".join(str(x or "").lower().split())

    def _azar(self, *partes) -> float:
        import hashlib
        h = hashlib.sha256("|".join(str(p) for p in (self.semilla, *partes)).encode()).hexdigest()
        return int(h[:12], 16) / float(16 ** 12)

    def _escena(self, c: Dict[str, Any]) -> Dict[str, Any]:
        t = c.get("tarea", "_")
        if t in self.escenas:
            return self.escenas[t]
        p = c.get("parametros") or {}
        lugares = ["left", "center", "right", "top shelf", "bottom shelf"]
        origen, destinos = {}, {}
        items = p.get("items") if isinstance(p.get("items"), dict) else None
        if items:
            for k, q in items.items():
                origen[k] = int(q) + self.extra
            destinos[p.get("destination", "box")] = {}
        acciones = (c.get("referencia") or {}).get("acciones") or []
        for a in acciones:
            arg = a.get("arguments", {})
            if arg.get("object"):
                origen.setdefault(arg["object"], 1)
            if arg.get("destination"):
                destinos.setdefault(arg["destination"], {})
        if p.get("object"):
            origen.setdefault(p["object"], 1)
        for d in self.distractores:
            if all(self._n(d) != self._n(k) for k in origen):
                origen[d] = 1 + int(3 * self._azar(t, "dist", d))
        ubic = {k: lugares[int(self._azar(t, "ubic", k) * len(lugares))] for k in origen}
        esc = {"origen": origen, "destinos": destinos or {p.get("destination", "box"): {}}, "ubic": ubic}
        self.escenas[t] = esc
        return esc

    def _buscar(self, esc, nombre):
        for k in esc["origen"]:
            if self._n(k) == self._n(nombre) or (self._n(nombre) and self._n(nombre) in self._n(k)):
                return k
        return None

    def _destino(self, esc, nombre):
        for k in esc["destinos"]:
            if self._n(k) == self._n(nombre) or self._n(nombre) in self._n(k) or self._n(k) in self._n(nombre):
                return k
        esc["destinos"][nombre] = {}
        return nombre

    def _describir(self, esc) -> str:
        partes = [f"{k} x{n} ({esc['ubic'].get(k, 'table')})" for k, n in esc["origen"].items() if n > 0]
        dentro = []
        for d, cont in esc["destinos"].items():
            txt = ", ".join(f"{k} x{n}" for k, n in cont.items() if n > 0) or "empty"
            dentro.append(f"{d}: {txt}")
        return ("Visible on the shelf/table: " + (", ".join(partes) or "nothing") + ". " +
                "Destinations -> " + "; ".join(dentro) + ".")

    def consultar(self, prompt, imagenes, esquema=None, contexto=None) -> str:
        c = contexto or {}
        esc = self._escena(c)
        prop = c.get("proposito", "")
        sch = c.get("esquema") if isinstance(c.get("esquema"), dict) else {}
        props = sch.get("properties", {}) if isinstance(sch, dict) else {}
        if prop == "observacion_inicial":
            if "items" in props:
                return json.dumps({"items": [{"name": k, "count": n, "location": esc["ubic"][k]}
                                             for k, n in esc["origen"].items()]}, ensure_ascii=False)
            if "objects" in props:
                return json.dumps({"objects": [{"name": k, "location": esc["ubic"][k]} for k in esc["origen"]],
                                   "description": self._describir(esc)}, ensure_ascii=False)
            return json.dumps({"description": self._describir(esc)}, ensure_ascii=False)
        if prop == "verificacion":
            a = c.get("argumentos") or {}
            obj = self._buscar(esc, a.get("object", ""))
            dest = self._destino(esc, a.get("destination") or next(iter(esc["destinos"])))
            sale = self._azar(c.get("tarea"), "verif", c.get("intento", 0)) < self.tasa
            if obj is None or esc["origen"].get(obj, 0) <= 0:
                return json.dumps({"success": False, "reason": f"There is no {a.get('object')} available to move."})
            if not sale:
                return json.dumps({"success": False, "reason": f"The {obj} was dropped; it is not in the {dest}."})
            esc["origen"][obj] -= 1
            esc["destinos"][dest][obj] = esc["destinos"][dest].get(obj, 0) + 1
            return json.dumps({"success": True, "reason": f"The {obj} is now in the {dest}."})
        if prop == "revision_final":
            if c.get("e2e"):
                ok = self._azar(c.get("tarea"), "e2e") < self.tasa
                return json.dumps({"success": ok, "reason": "simulated end-to-end result"})
            items = (c.get("parametros") or {}).get("items")
            if isinstance(items, dict):
                dest = next(iter(esc["destinos"]))
                ok = all(esc["destinos"][dest].get(k, 0) >= int(q) for k, q in items.items())
            else:
                ok = True
                for a in (c.get("referencia") or {}).get("acciones") or []:
                    arg = a.get("arguments", {})
                    obj, d = self._buscar(esc, arg.get("object", "")), self._destino(esc, arg.get("destination", ""))
                    ok = ok and obj is not None and esc["destinos"][d].get(obj, 0) > 0
            return json.dumps({"success": ok, "reason": self._describir(esc)})
        # pregunta libre del LLM (observe): descripcion veraz de la escena
        return self._describir(esc)
