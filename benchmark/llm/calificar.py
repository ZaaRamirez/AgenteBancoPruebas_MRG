"""Calificacion de una respuesta del LLM contra la respuesta correcta del caso.

Devuelve correcto (bool), tipo de error y la accion interpretada. Tipos de error:
  respuesta_vacia       el modelo no devolvio texto util (p. ej. agoto max_tokens razonando)
  formato               no hay un objeto JSON con el campo "action"
  accion_inexistente    accion que no esta en la lista de acciones disponibles
  argumentos_incompletos falta el objeto (o la instruccion) de la habilidad
  repite_lo_hecho       elige algo ya completado
  elemento_no_pedido    elige algo que esta en la escena pero no se pidio (distractor)
  elemento_inventado    elige algo que no existe ni en el pedido ni en la escena
  orden_incorrecto      elige un paso pendiente pero no el que sigue (instrucciones en orden)
  destino_incorrecto    objeto correcto, destino equivocado
  fin_prematuro         responde "done" con pasos pendientes
  no_termina            todo esta hecho y no responde "done"
  aborto_injustificado  responde "abort" cuando la tarea es realizable
  ignora_percepcion     la percepcion dijo que el objeto no esta y aun asi intenta tomarlo
  observa_sin_necesidad pregunta a la percepcion cuando ya tenia todo para decidir
"""
from __future__ import annotations

from typing import Any, Dict

from agente.util.texto import extraer_json, norm, quitar_razonamiento

TIPOS_ERROR = ["respuesta_vacia", "formato", "accion_inexistente", "argumentos_incompletos", "repite_lo_hecho",
               "elemento_no_pedido", "elemento_inventado", "orden_incorrecto", "destino_incorrecto", "fin_prematuro",
               "no_termina", "aborto_injustificado", "ignora_percepcion", "observa_sin_necesidad", "error_servidor"]
# errores que dejarian al agente en un estado incorrecto si no los atrapara su validacion
ERRORES_DE_ESTADO = {"repite_lo_hecho", "elemento_no_pedido", "elemento_inventado", "fin_prematuro",
                     "ignora_percepcion", "orden_incorrecto", "destino_incorrecto"}


def _igual(a, b) -> bool:
    a, b = norm(a), norm(b)
    return bool(a) and bool(b) and (a == b or a in b or b in a)


def _en(nombre, lista) -> bool:
    return any(_igual(nombre, x) for x in lista)


def calificar(caso: Dict[str, Any], texto: str) -> Dict[str, Any]:
    r = {"correcto": False, "error": "", "formato_ok": False, "accion": "", "objeto": "", "destino": ""}
    if not quitar_razonamiento(texto or ""):
        r["error"] = "respuesta_vacia"
        return r
    datos = extraer_json(texto)
    if not isinstance(datos, dict) or "action" not in datos:
        r["error"] = "formato"
        return r
    r["formato_ok"] = True
    accion = str(datos.get("action", "")).strip()
    args = datos.get("arguments") if isinstance(datos.get("arguments"), dict) else \
        {k: v for k, v in datos.items() if k not in ("thought", "action", "arguments")}
    r["accion"] = accion
    esp = caso["esperado"]
    disponibles = [p["enum"] for p in [caso["esquema"]["properties"]["action"]]][0]
    if accion not in disponibles:
        r["error"] = "accion_inexistente"
        return r
    pendientes = caso.get("pendientes") or {}
    hay_pendientes = bool(pendientes)
    if accion == "done":
        r["correcto"] = "done" in esp["acciones"]
        r["error"] = "" if r["correcto"] else "fin_prematuro"
        return r
    if accion == "abort":
        r["correcto"] = "abort" in esp["acciones"]
        r["error"] = "" if r["correcto"] else "aborto_injustificado"
        return r
    if accion == "observe":
        r["correcto"] = "observe" in esp["acciones"]
        r["error"] = "" if r["correcto"] else ("no_termina" if "done" in esp["acciones"] else "observa_sin_necesidad")
        return r
    # ---- habilidad de manipulacion
    if accion == "vla_instruction":
        instr = str(args.get("instruction", ""))
        objeto = next((o for o in list(pendientes) + caso.get("completados", []) + caso.get("conocidos", [])
                       if norm(o) in norm(instr)), "")
        destino = instr
    else:
        objeto, destino = str(args.get("object", "")), str(args.get("destination", ""))
    r["objeto"], r["destino"] = objeto, destino
    if not norm(objeto):
        r["error"] = "argumentos_incompletos"
        return r
    if "done" in esp["acciones"]:
        r["error"] = "no_termina"
        return r
    if "abort" in esp["acciones"]:                       # el objeto pedido no esta en la escena
        r["error"] = "ignora_percepcion"
        return r
    if "objetos" in esp:                                 # medicamentos: cualquier pendiente
        if _en(objeto, esp["objetos"]):
            if norm(destino) and not _igual(destino, esp.get("destino", destino)):
                r["error"] = "destino_incorrecto"
            else:
                r["correcto"] = True
        elif _en(objeto, caso.get("completados", [])):
            r["error"] = "repite_lo_hecho"
        elif _en(objeto, caso.get("no_pedidos", [])):
            r["error"] = "elemento_no_pedido"
        else:
            r["error"] = "elemento_inventado"
        return r
    opciones = esp.get("opciones", [])
    ok_obj = [o for o in opciones if _igual(objeto, o["object"])]
    if ok_obj:
        if accion == "vla_instruction" or _igual(destino, ok_obj[0]["destination"]):
            r["correcto"] = True
        else:
            r["error"] = "destino_incorrecto"
    elif _en(objeto, caso.get("completados", [])) and not _en(objeto, list(pendientes)):
        r["error"] = "repite_lo_hecho"
    elif _en(objeto, list(pendientes)):
        r["error"] = "orden_incorrecto"
    elif _en(objeto, caso.get("no_pedidos", [])):
        r["error"] = "elemento_no_pedido"
    else:
        r["error"] = "elemento_inventado"
    if not hay_pendientes and not r["error"]:
        r["error"] = "no_termina"
    return r
