"""Capa de RAZONAMIENTO (LLM): interpreta la tarea y decide la siguiente accion.

El LLM recibe la tarea, el estado (que lleva el agente), las observaciones de la
percepcion y el historial, y responde UNA accion en JSON:
    {"thought": "...", "action": "<habilidad | observe | done | abort>", "arguments": {...}}
La capa valida la respuesta (formato, habilidad existente, argumentos completos,
coherencia con el estado) antes de entregarla al agente. Nada aqui depende de la
tarea
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List

from ..tareas.perfiles import Profile
from ..tareas.seguimiento import make_tracker
from ..tipos import Decision
from ..util.texto import extraer_json, quitar_razonamiento, rellenar

ACCIONES_BASE = {
    "observe": ("observe(question)", "Ask the perception layer (camera + VLM) a question about the scene."),
    "done": ("done()", "Declare that the whole task is complete."),
    "abort": ("abort(reason)", "Stop because the task cannot be completed (explain why)."),
}

PLANTILLA = """{role}

TASK
- Task ID: {task_id}
- Instruction: {instruction}
- Parameters: {parameters}

AVAILABLE ACTIONS
{actions}

TASK STATE (kept by the agent, trust it)
{state}

OBSERVATIONS FROM THE PERCEPTION LAYER
{observations}

SUBTASKS EXECUTED SO FAR
{history}

Decide the next single action. Respond ONLY with one JSON object in this format:
{{"thought": "brief reasoning", "action": "<action name>", "arguments": {{"<argument>": "<value>"}}}}
"""


@dataclass
class ContextoTarea:
    """Estado de la tarea. Lo lleva el agente, no el modelo."""
    tarea_id: str
    instruccion: str
    parametros: Dict[str, Any]
    perfil: Profile
    seguimiento: Any = None
    historial: List[Dict[str, Any]] = field(default_factory=list)
    observaciones: List[Dict[str, Any]] = field(default_factory=list)
    reintentos: int = 0
    indice_subtarea: int = 0
    llamadas_llm: int = 0
    inicio: float = field(default_factory=time.time)
    referencia: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.seguimiento is None:
            self.seguimiento = make_tracker(self.perfil.tracker, self.parametros)

    @property
    def subtareas_exitosas(self) -> int:
        return sum(1 for h in self.historial if h.get("exito"))


def acciones(p: Profile) -> List[str]:
    return [s.name for s in p.skills] + (["observe"] if p.allow_observe else []) + ["done", "abort"]


def esquema_decision(p: Profile) -> dict:
    return {"type": "object", "properties": {
        "thought": {"type": "string"}, "action": {"type": "string", "enum": acciones(p)},
        "arguments": {"type": "object"}}, "required": ["thought", "action"]}


def describir_acciones(p: Profile) -> str:
    lineas = []
    for s in p.skills:
        args = "; ".join(f"{k}: {v}" for k, v in s.arguments.items())
        lineas.append(f"- {s.signature()}: {s.description}" + (f" Arguments -> {args}" if args else ""))
    for nombre in ("observe", "done", "abort"):
        if nombre == "observe" and not p.allow_observe:
            continue
        firma, desc = ACCIONES_BASE[nombre]
        lineas.append(f"- {firma}: {desc}")
    return "\n".join(lineas)


def construir_prompt(c: ContextoTarea) -> str:
    p = c.perfil
    obs = "\n".join(f"- [{o['proposito']}] Q: {o['pregunta'][:200]}\n  A: {o['respuesta']}"
                    for o in c.observaciones[-6:])
    hist = "\n".join(
        f"{h['indice'] + 1}. {h['habilidad']}({json.dumps(h['argumentos'], ensure_ascii=False)}) -> "
        f"{'success' if h['exito'] else 'FAILED: ' + h.get('detalle', '')}" for h in c.historial) or "None"
    return rellenar(p.reasoning_template or PLANTILLA, c.parametros, {
        "role": rellenar(p.role, c.parametros) or "You are the reasoning layer of a robot agent.",
        "task_id": c.tarea_id, "instruction": c.instruccion, "actions": describir_acciones(p),
        "state": c.seguimiento.summary(), "observations": obs or "None", "history": hist})


def validar(d: Decision, c: ContextoTarea) -> Decision:
    p = c.perfil
    if d.accion not in acciones(p):
        d.motivo = f"accion_desconocida:{d.accion}"
        return d
    if d.accion == "done":
        if p.auto_complete and c.seguimiento.is_complete() is False:
            d.motivo = "fin_prematuro"
            return d
    elif d.accion == "abort":
        d.argumentos.setdefault("reason", d.pensamiento)
    elif d.accion == "observe":
        if not str(d.argumentos.get("question", "")).strip():
            d.motivo = "falta_argumento:question"
            return d
    else:
        s = p.skill(d.accion)
        for k, v in list(d.argumentos.items()):
            if isinstance(v, str) and "{" in v:
                d.argumentos[k] = rellenar(v, c.parametros)
        for k, v in s.defaults.items():
            if not str(d.argumentos.get(k, "")).strip():
                d.argumentos[k] = rellenar(v, c.parametros)
        faltan = [k for k in s.arguments if not str(d.argumentos.get(k, "")).strip()]
        if faltan:
            d.motivo = "falta_argumento:" + ",".join(faltan)
            return d
        ok, por_que = c.seguimiento.validate(d.accion, d.argumentos)
        if not ok:
            d.motivo = por_que
            return d
    d.valida = True
    return d


def interpretar(texto: str, c: ContextoTarea) -> Decision:
    d = Decision(texto=texto)
    if not quitar_razonamiento(texto or ""):
        d.motivo = "respuesta_vacia"        # tipico de modelos que agotan max_tokens razonando
        return d
    datos = extraer_json(texto)
    if not isinstance(datos, dict) or "action" not in datos:
        d.motivo = "formato_invalido"
        return d
    d.formato_ok = True
    d.pensamiento = str(datos.get("thought", ""))
    d.accion = str(datos.get("action", "")).strip()
    args = datos.get("arguments")
    if not isinstance(args, dict):
        args = {k: v for k, v in datos.items() if k not in ("thought", "action", "arguments")}
    d.argumentos = args
    return validar(d, c)


class CapaRazonamiento:
    def __init__(self, llm, registro=None, salida_estructurada: str = "json", max_reintentos_formato: int = 2):
        self.llm = llm
        self.registro = registro
        self.salida = salida_estructurada          # none | json | schema
        self.max_reintentos = int(max_reintentos_formato)

    @property
    def modelo(self) -> str:
        return getattr(self.llm, "etiqueta", getattr(self.llm, "nombre", "?"))

    def _esquema(self, p):
        return esquema_decision(p) if self.salida == "schema" else ("json" if self.salida == "json" else None)

    def decidir(self, c: ContextoTarea) -> Decision:
        """Consulta al LLM; si la respuesta no es valida reintenta hasta max_reintentos_formato."""
        d = Decision()
        for intento in range(1 + self.max_reintentos):
            prompt = construir_prompt(c)
            ctx = {"pendientes": c.seguimiento.remaining(), "argumento": getattr(c.seguimiento, "argument", None),
                   "habilidades": [s.name for s in c.perfil.skills],
                   "plan": c.referencia.get("acciones") or c.perfil.scripted_plan,
                   "hechas": c.subtareas_exitosas, "completa": c.seguimiento.is_complete()}
            t0 = time.perf_counter()
            if hasattr(self.llm, "ultimo"):
                self.llm.ultimo = {}
            try:
                texto = self.llm.generar(prompt, esquema=self._esquema(c.perfil), contexto=ctx)
                d = interpretar(texto, c)
            except Exception as exc:
                d = Decision(motivo=f"error_llm:{exc}")
            d.latencia_s = time.perf_counter() - t0
            c.llamadas_llm += 1
            if self.registro:
                meta = dict(getattr(self.llm, "ultimo", {}) or {})
                self.registro.evento("llm", capa="razonamiento", modelo=self.modelo, intento=intento,
                                     llamada=c.llamadas_llm, prompt=prompt, respuesta=d.texto,
                                     latencia_s=d.latencia_s, formato_ok=d.formato_ok, valida=d.valida,
                                     motivo=d.motivo, accion=d.accion, argumentos=d.argumentos,
                                     pensamiento=d.pensamiento, salida_estructurada=self.salida, **meta)
            if d.valida or d.motivo.startswith("error_llm"):
                return d
        return d

    @staticmethod
    def instruccion_subtarea(c: ContextoTarea, habilidad: str, args: Dict[str, Any]) -> str:
        s = c.perfil.skill(habilidad)
        plantilla = s.instruction_template or (s.name + " " + " ".join(str(v) for v in args.values()))
        return rellenar(plantilla, c.parametros, dict(args))
