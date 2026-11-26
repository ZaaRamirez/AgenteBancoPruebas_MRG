"""Perfiles de tarea (config/tareas/<perfil>.yaml).

Un perfil es lo UNICO que cambia entre tareas. Define el rol de la capa de
razonamiento, las habilidades que el agente puede ejecutar y como se traducen a
instrucciones para el VLA, las consultas a la percepcion (observacion inicial,
verificaciones, revision final) y el seguimiento del estado que lleva el agente.
El ciclo del agente y las capas no contienen nada especifico de una tarea.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import yaml

from ..util.config import ruta
from ..util.texto import rellenar as fill


def escape(text: str) -> str:
    return text.replace("{", "{{").replace("}", "}}")


@dataclass
class Query:
    """Consulta a la capa de percepcion definida en el perfil."""
    question: str
    schema: Optional[dict] = None
    simulated_answer: str = ""
    abort_if_false: str = ""        # campo booleano de la respuesta que, si es falso, aborta
    check_with_tracker: bool = False

    @staticmethod
    def from_dict(d) -> Optional["Query"]:
        if not d:
            return None
        sa = d.get("simulated_answer", "")
        return Query(question=d["question"], schema=d.get("schema"),
                     simulated_answer=escape(json.dumps(sa)) if isinstance(sa, (dict, list)) else str(sa or ""),
                     abort_if_false=d.get("abort_if_false", ""),
                     check_with_tracker=bool(d.get("check_with_tracker", False)))


@dataclass
class Skill:
    name: str
    description: str
    arguments: Dict[str, str] = field(default_factory=dict)   # nombre -> descripcion
    defaults: Dict[str, str] = field(default_factory=dict)    # valores por omision (plantillas)
    instruction_template: str = ""
    verify: Optional[Query] = None
    executor: str = "vla"           # vla | none (la habilidad no mueve el robot)

    def signature(self) -> str:
        args = ", ".join(self.arguments)
        return f"{self.name}({args})"


@dataclass
class Profile:
    name: str
    description: str = ""
    role: str = ""
    instruction_template: str = ""
    reasoning_template: str = ""
    skills: List[Skill] = field(default_factory=list)
    initial_observation: Optional[Query] = None
    final_check: Optional[Query] = None
    tracker: Dict[str, Any] = field(default_factory=dict)
    auto_complete: bool = True
    allow_observe: bool = True
    scripted_plan: List[Dict[str, Any]] = field(default_factory=list)
    limits: Dict[str, int] = field(default_factory=dict)
    e2e_instruction_template: str = ""
    path: str = ""

    def skill(self, name: str) -> Optional[Skill]:
        for s in self.skills:
            if s.name == name:
                return s
        return None

    def instruction(self, params: Dict[str, Any], given: str = "") -> str:
        return given or fill(self.instruction_template, params)


def load_profile(name_or_path: str) -> Profile:
    path = ruta(name_or_path, "tareas")
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f) or {}
    skills = []
    for s in d.get("skills", []):
        skills.append(Skill(name=s["name"], description=s.get("description", ""),
                            arguments=dict(s.get("arguments", {}) or {}),
                            defaults=dict(s.get("defaults", {}) or {}),
                            instruction_template=s.get("instruction_template", ""),
                            verify=Query.from_dict(s.get("verify")),
                            executor=s.get("executor", "vla")))
    return Profile(name=d.get("name", os.path.splitext(os.path.basename(path))[0]),
                   description=d.get("description", ""), role=d.get("role", ""),
                   instruction_template=d.get("instruction_template", ""),
                   reasoning_template=d.get("reasoning_template", ""),
                   skills=skills, initial_observation=Query.from_dict(d.get("initial_observation")),
                   final_check=Query.from_dict(d.get("final_check")), tracker=d.get("tracker", {}) or {},
                   auto_complete=bool(d.get("auto_complete", True)),
                   allow_observe=bool(d.get("allow_observe", True)),
                   scripted_plan=d.get("scripted_plan", []) or [], limits=d.get("limits", {}) or {},
                   e2e_instruction_template=d.get("e2e_instruction_template", ""), path=path)
