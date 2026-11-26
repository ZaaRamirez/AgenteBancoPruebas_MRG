"""Utilidades de texto: limpieza de salidas de modelos, JSON, esquemas y plantillas."""
from __future__ import annotations

import json
import random
import re
from typing import Any, Dict, Optional

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_TOKEN = re.compile(r"\{\{|\}\}|\{([A-Za-z_][A-Za-z0-9_]*)\}")
_TIPOS = {"string": str, "boolean": bool, "integer": int, "number": (int, float), "array": list, "object": dict}


def quitar_razonamiento(texto: str) -> str:
    """Elimina bloques <think>...</think> (modelos de razonamiento como DeepSeek-R1)."""
    return _THINK.sub("", texto or "").strip()


def extraer_json(texto: str) -> Optional[Any]:
    """Primer objeto JSON valido dentro del texto (ignora texto alrededor y ```json```)."""
    texto = quitar_razonamiento(texto)
    for inicio in [m.start() for m in re.finditer(r"\{", texto)]:
        prof, en_cadena, esc = 0, False, False
        for i in range(inicio, len(texto)):
            ch = texto[i]
            if en_cadena:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    en_cadena = False
                continue
            if ch == '"':
                en_cadena = True
            elif ch == "{":
                prof += 1
            elif ch == "}":
                prof -= 1
                if prof == 0:
                    try:
                        valor = json.loads(texto[inicio:i + 1])
                    except json.JSONDecodeError:
                        break
                    if isinstance(valor, dict):
                        return valor
                    break
    return None


def norm(nombre: str) -> str:
    """Normaliza nombres para compararlos (minusculas, espacios simples)."""
    return " ".join(str(nombre or "").strip().lower().split())


def validar_esquema(valor, esquema) -> list:
    """Validacion ligera de JSON Schema (type, required, properties, items, enum).
    Devuelve la lista de errores; vacia si cumple."""
    errores = []
    if not isinstance(esquema, dict):
        return errores
    t = esquema.get("type")
    if t in _TIPOS:
        ok = isinstance(valor, _TIPOS[t]) and not (t in ("integer", "number") and isinstance(valor, bool))
        if not ok:
            return [f"se esperaba {t}"]
    if "enum" in esquema and valor not in esquema["enum"]:
        errores.append(f"valor fuera de {esquema['enum']}")
    if isinstance(valor, dict):
        errores += [f"falta '{k}'" for k in esquema.get("required", []) if k not in valor]
        for k, sub in esquema.get("properties", {}).items():
            if k in valor:
                errores += [f"{k}: {e}" for e in validar_esquema(valor[k], sub)]
    if isinstance(valor, list) and "items" in esquema:
        for i, v in enumerate(valor):
            errores += [f"[{i}]: {e}" for e in validar_esquema(v, esquema["items"])]
    return errores


def ejemplo_de_esquema(esquema, rng=None, tasa_exito: float = 1.0):
    """Instancia minima valida de un esquema (respuesta del VLM simulado)."""
    rng = rng or random
    if not isinstance(esquema, dict):
        return "simulated"
    if "enum" in esquema:
        return esquema["enum"][0]
    t = esquema.get("type", "object")
    if t == "object":
        out = {}
        for k, sub in esquema.get("properties", {}).items():
            if k == "success" and isinstance(sub, dict) and sub.get("type") == "boolean":
                out[k] = rng.random() < tasa_exito
            else:
                out[k] = ejemplo_de_esquema(sub, rng, tasa_exito)
        return out
    return {"string": "simulated", "boolean": True, "integer": 1, "number": 1.0, "array": []}.get(t)


def variables(parametros: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Variables de plantilla. Para un parametro diccionario ``x`` (p. ej. items):
    {x} JSON, {x_text} "1 A, 2 B", {x_names} "A, B", {x_as_list} '[{"name":"A","count":1}]'.
    Para listas: {x} JSON y {x_text}."""
    v: Dict[str, Any] = {}
    for k, val in (parametros or {}).items():
        if isinstance(val, dict):
            v[k] = json.dumps(val, ensure_ascii=False)
            v[f"{k}_as_list"] = json.dumps([{"name": n, "count": q} for n, q in val.items()], ensure_ascii=False)
            v[f"{k}_text"] = ", ".join(f"{q} {n}" for n, q in val.items())
            v[f"{k}_names"] = ", ".join(str(n) for n in val)
        elif isinstance(val, list):
            v[k] = json.dumps(val, ensure_ascii=False)
            v[f"{k}_text"] = ", ".join(str(x) for x in val)
        else:
            v[k] = val
    v["parameters"] = json.dumps(parametros or {}, ensure_ascii=False)
    v.update(extra or {})
    return v


def rellenar(plantilla: str, parametros: Dict[str, Any], extra: Optional[Dict[str, Any]] = None) -> str:
    """Sustituye {variable}; {{ y }} dan llaves literales; lo demas queda intacto."""
    v = variables(parametros, extra)

    def rep(m):
        tok = m.group(0)
        if tok == "{{":
            return "{"
        if tok == "}}":
            return "}"
        return str(v[m.group(1)]) if m.group(1) in v else tok
    return _TOKEN.sub(rep, plantilla or "")
