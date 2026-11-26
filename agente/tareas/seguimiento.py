"""Seguimiento del estado de la tarea (lo lleva el agente, no el modelo).

Cada perfil elige un tipo en ``tracker.type``:
- ``none``   : el agente solo guarda el historial; el LLM decide cuando terminar.
- ``counts`` : la tarea pide N unidades de cada elemento de un parametro tipo
               diccionario (p. ej. pedidos, clasificacion por cantidad). El agente
               rechaza elegir elementos ya completos y declara el fin al completar.
Para una tarea nueva con otra logica se agrega aqui una clase y se registra en TRACKERS.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from ..util.texto import norm


class NoTracker:
    type = "none"

    def __init__(self, params: Dict[str, Any], **_):
        self.params = params

    def summary(self) -> str:
        return "No numeric tracking for this task; use the history and observations."

    def validate(self, skill: str, args: Dict[str, Any]) -> Tuple[bool, str]:
        return True, ""

    def register(self, skill: str, args: Dict[str, Any]) -> None:
        pass

    def is_complete(self) -> Optional[bool]:
        return None                      # desconocido: lo decide el LLM

    def progress(self) -> float:
        return -1.0

    def remaining(self) -> Dict[str, int]:
        return {}

    def check_observation(self, answer: Any) -> Tuple[bool, str]:
        return True, ""


class CountTracker(NoTracker):
    """Cuenta unidades por elemento. Configuracion en el perfil:
        tracker: {type: counts, parameter: items, argument: object, skills: [pick_and_place]}
    """
    type = "counts"

    def __init__(self, params, parameter: str = "items", argument: str = "object", skills=None,
                 inventory_field: str = "items", **_):
        super().__init__(params)
        self.argument = argument
        self.skills = set(skills or [])
        self.inventory_field = inventory_field
        self.required = {str(k): int(v) for k, v in (params.get(parameter) or {}).items() if int(v) > 0}
        self.done: Dict[str, int] = {k: 0 for k in self.required}

    def _canonical(self, name: str) -> Optional[str]:
        for k in self.required:
            if norm(k) == norm(name):
                return k
        return None

    def remaining(self) -> Dict[str, int]:
        return {k: q - self.done[k] for k, q in self.required.items() if q - self.done[k] > 0}

    def summary(self) -> str:
        req = ", ".join(f"{k} x{q}" for k, q in self.required.items())
        rem = ", ".join(f"{k} x{q}" for k, q in self.remaining().items()) or "None - everything is complete"
        return f"Required: {req}\nAlready completed: {sum(self.done.values())} unit(s)\nRemaining: {rem}"

    def _applies(self, skill):
        return not self.skills or skill in self.skills

    def validate(self, skill, args):
        if not self._applies(skill):
            return True, ""
        obj = self._canonical(str(args.get(self.argument, "")))
        if obj is None:
            return False, f"unknown_{self.argument}:{args.get(self.argument)}"
        if obj not in self.remaining():
            return False, f"already_complete:{obj}"
        args[self.argument] = obj
        return True, ""

    def register(self, skill, args):
        if self._applies(skill):
            obj = self._canonical(str(args.get(self.argument, "")))
            if obj:
                self.done[obj] += 1

    def is_complete(self):
        return not self.remaining()

    def progress(self):
        total = sum(self.required.values())
        return sum(self.done.values()) / total if total else 1.0

    def check_observation(self, answer):
        """Comprueba que la observacion inicial reporte suficientes unidades."""
        items = answer.get(self.inventory_field) if isinstance(answer, dict) else None
        if not isinstance(items, list):
            return True, ""
        seen: Dict[str, int] = {}
        for it in items:
            if isinstance(it, dict):
                try:
                    seen[norm(str(it.get("name", "")))] = seen.get(norm(str(it.get("name", ""))), 0) + int(it.get("count", 0))
                except (TypeError, ValueError):
                    pass
        missing = {k: q - seen.get(norm(k), 0) for k, q in self.required.items() if seen.get(norm(k), 0) < q}
        if missing:
            return False, "missing: " + ", ".join(f"{k} x{v}" for k, v in missing.items())
        return True, ""


TRACKERS = {"none": NoTracker, "counts": CountTracker}


def make_tracker(cfg: Dict[str, Any], params: Dict[str, Any]):
    cfg = dict(cfg or {})
    kind = cfg.pop("type", "none")
    if kind not in TRACKERS:
        raise ValueError(f"tracker '{kind}' desconocido; opciones {sorted(TRACKERS)}")
    return TRACKERS[kind](params, **cfg)
