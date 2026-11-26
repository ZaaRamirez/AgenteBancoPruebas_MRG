"""Conversion de entradas libres (dict, JSON o texto) a Tarea."""
from __future__ import annotations

import json
import uuid
from typing import Any, Dict, Union

from ..tipos import Tarea


def nuevo_id(base: str = "T") -> str:
    return f"{base}-{uuid.uuid4().hex[:6]}"


def a_tarea(x: Union[str, Dict[str, Any]], perfil_por_omision: str = "manipulacion_general") -> Union[Tarea, Dict]:
    """Acepta un dict / JSON {"id","perfil","parametros","instruccion","solicitante"},
    un comando {"cambiar": {"llm": "..."}} o texto libre (instruccion)."""
    if isinstance(x, str):
        s = x.strip()
        if s.startswith("{"):
            x = json.loads(s)
        else:
            return Tarea(id=nuevo_id(), perfil=perfil_por_omision, instruccion=s)
    if "cambiar" in x:
        return {"cambiar": dict(x["cambiar"])}
    return Tarea(id=str(x.get("id") or nuevo_id()), perfil=x.get("perfil", perfil_por_omision),
                 parametros=dict(x.get("parametros", {}) or {}), instruccion=x.get("instruccion", ""),
                 solicitante=x.get("solicitante", ""), referencia=dict(x.get("referencia", {}) or {}))
