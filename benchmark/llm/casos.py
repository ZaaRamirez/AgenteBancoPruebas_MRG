"""Generacion de CASOS DE DECISION para el benchmark LLM.

Un caso congela un momento de una tarea (que se pidio, que ya se hizo, que fallo, que
ve la percepcion) y pregunta al LLM la SIGUIENTE accion. La respuesta correcta se
conoce de antemano, asi que cada respuesta se califica como acierto o error, y el
error se clasifica por tipo. El prompt se construye con el MISMO codigo que usa el
agente (agente/capas/razonamiento.py), de modo que se evalua exactamente lo que el
LLM vera en operacion. Con la misma semilla todos los modelos reciben los mismos casos.

Categorias (empaquetado_medicamentos, el agente lleva el conteo):
  med_inicio        nada hecho todavia
  med_intermedio    avance parcial al azar
  med_ultimo        falta una sola unidad
  med_tras_fallo    la ultima subtarea fallo (el elemento sigue pendiente)
  med_distractores  el estante tiene medicamentos que NO se pidieron
  med_grande        5-6 medicamentos y muchas unidades
  med_real          pedidos reales de comanda.xlsx con avance al azar
Categorias (manipulacion_general, el LLM decide los pasos y cuando terminar):
  gen_secuencial    instruccion de 1-4 pasos en orden ("..., then ..."): solo el siguiente paso es correcto
  gen_conjunto      varios objetos al mismo destino: cualquier pendiente es correcto
  gen_completo      ya se hizo todo: lo correcto es "done"
  gen_tras_fallo    el ultimo paso fallo: hay que reintentarlo
  gen_ausente       la percepcion dice que el objeto pedido no esta: lo correcto es abort u observe
"""
from __future__ import annotations

import copy
import json
import random
from typing import Any, Dict, List

from agente.capas.razonamiento import ContextoTarea, construir_prompt, esquema_decision
from agente.tareas.perfiles import load_profile

CATEGORIAS_MED = ["med_inicio", "med_intermedio", "med_ultimo", "med_tras_fallo", "med_distractores",
                  "med_grande", "med_real"]
CATEGORIAS_GEN = ["gen_secuencial", "gen_conjunto", "gen_completo", "gen_tras_fallo", "gen_ausente"]
CATEGORIAS = CATEGORIAS_MED + CATEGORIAS_GEN
LUGARES = ["left", "center", "right", "top shelf", "bottom shelf"]


# ----------------------------------------------------------------------------- medicamentos
def _pedido(rng, meds, qmax, k=None):
    k = k or rng.randint(1, min(4, len(meds)))
    return {m: rng.randint(1, qmax) for m in rng.sample(meds, k)}


def _avance(rng, items: Dict[str, int], minimo_restante: int = 1):
    """Elige cuantas unidades de cada elemento ya se colocaron (al menos minimo_restante pendientes)."""
    unidades = [m for m, q in items.items() for _ in range(q)]
    rng.shuffle(unidades)
    total = len(unidades)
    hechas = rng.randint(1, max(1, total - minimo_restante)) if total > minimo_restante else 0
    return unidades[:hechas]


def _inventario(rng, items, distractores=()):
    filas = [{"name": m, "count": q + rng.randint(0, 3), "location": rng.choice(LUGARES)} for m, q in items.items()]
    filas += [{"name": d, "count": rng.randint(1, 4), "location": rng.choice(LUGARES)} for d in distractores]
    rng.shuffle(filas)
    return json.dumps({"items": filas}, ensure_ascii=False)


def _caso_med(rng, cat, i, vocab, comandas):
    meds, qmax, destino = vocab["medicamentos"], int(vocab.get("cantidad_maxima", 5)), vocab["destino_medicamentos"]
    distractores = []
    solicitante = ""
    if cat == "med_real" and comandas.get("pedidos"):
        ped = comandas["pedidos"][i % len(comandas["pedidos"])]
        items, solicitante = dict(ped["items"]), ped.get("solicitante", "")
        destino = comandas.get("destino", destino)
    elif cat == "med_grande":
        items = _pedido(rng, meds, qmax, k=rng.randint(5, len(meds)))
    elif cat == "med_ultimo":
        items = _pedido(rng, meds, qmax)
    else:
        items = _pedido(rng, meds, qmax)
    if cat == "med_inicio":
        hechas = []
    elif cat == "med_ultimo":
        unidades = [m for m, q in items.items() for _ in range(q)]
        rng.shuffle(unidades)
        hechas = unidades[:-1]
    else:
        hechas = _avance(rng, items)
    if cat == "med_distractores":
        distractores = rng.sample(vocab.get("medicamentos_distractores", []), k=min(2, len(vocab.get("medicamentos_distractores", []))))
    historial = [{"habilidad": "pick_and_place", "argumentos": {"object": m, "destination": destino}, "exito": True,
                  "detalle": f"The {m} is now in the {destino}."} for m in hechas]
    pendientes = dict(items)
    for m in hechas:
        pendientes[m] -= 1
    pendientes = {m: q for m, q in pendientes.items() if q > 0}
    if cat == "med_tras_fallo":
        fallo = rng.choice(sorted(pendientes))
        pos = rng.randint(0, len(historial))
        historial.insert(pos, {"habilidad": "pick_and_place", "argumentos": {"object": fallo, "destination": destino},
                               "exito": False, "detalle": f"The {fallo} was dropped; it is not in the {destino}."})
        historial.append({"habilidad": "pick_and_place", "argumentos": {"object": fallo, "destination": destino},
                          "exito": False, "detalle": f"The {fallo} was dropped; it is not in the {destino}."})
    params = {"items": items, "destination": destino}
    if solicitante:
        params["requester"] = solicitante
    obs = [{"proposito": "inicial", "pregunta": "inventory", "respuesta": _inventario(rng, items, distractores)}]
    return {"perfil": "empaquetado_medicamentos", "parametros": params, "instruccion": "", "historial": historial,
            "observaciones": obs, "esperado": {"acciones": ["pick_and_place"], "objetos": sorted(pendientes),
                                               "destino": destino, "ordenado": False},
            "conocidos": sorted(set(items) | set(distractores)), "no_pedidos": distractores,
            "completados": sorted(m for m in items if m not in pendientes),
            "pendientes": pendientes, "solicitante": solicitante}


# ----------------------------------------------------------------------------- instrucciones libres
def _instruccion(rng, pasos, vocab, conjunto=False):
    verbos = vocab["verbos"]
    if conjunto:
        objs = [o for o, _ in pasos]
        lista = ", ".join(f"the {o}" for o in objs[:-1]) + f" and the {objs[-1]}" if len(objs) > 1 else f"the {objs[0]}"
        return f"Put {lista} in the {pasos[0][1]}"
    frases = [rng.choice(verbos).format(o=o, d=d) for o, d in pasos]
    if len(frases) == 1:
        return frases[0]
    if rng.random() < 0.5:
        return ", then ".join([frases[0]] + [f[0].lower() + f[1:] for f in frases[1:]])
    conect = ["First", "Next", "Then", "After that", "Finally"]
    partes = [f"{conect[0]}, {frases[0][0].lower() + frases[0][1:]}"]
    for j, f in enumerate(frases[1:], 1):
        c = conect[-1] if j == len(frases) - 1 else conect[min(j, 3)]
        partes.append(f"{c}, {f[0].lower() + f[1:]}")
    return ". ".join(partes) + "."


def _descripcion(rng, objetos, ausente=None):
    vis = [o for o in objetos if o != ausente]
    filas = [{"name": o, "location": rng.choice(["left", "center", "right", "front", "back"])} for o in vis]
    desc = "Objects on the workspace: " + ", ".join(f"{f['name']} ({f['location']})" for f in filas) + "."
    if ausente:
        desc += f" There is no {ausente} anywhere on the workspace."
    return json.dumps({"objects": filas, "description": desc}, ensure_ascii=False)


def _caso_gen(rng, cat, i, vocab):
    n = rng.randint(1, 4) if cat in ("gen_secuencial", "gen_completo", "gen_ausente") else rng.randint(2, 4)
    objs = rng.sample(vocab["objetos"], n)
    conjunto = cat == "gen_conjunto"
    if conjunto:
        d = rng.choice(vocab["destinos"])
        pasos = [(o, d) for o in objs]
    else:
        pasos = [(o, rng.choice(vocab["destinos"])) for o in objs]
    instr = _instruccion(rng, pasos, vocab, conjunto)
    extra = rng.sample([o for o in vocab["objetos"] if o not in objs], k=1)
    if cat == "gen_completo":
        k = n
    elif cat == "gen_ausente":
        k = rng.randint(0, n - 1)
    elif cat == "gen_secuencial":
        k = rng.randint(0, n - 1)
    else:
        k = rng.randint(0, n - 1) if conjunto else rng.randint(0, n - 1)
    orden = list(pasos)
    if conjunto:
        rng.shuffle(orden)
    hechos = orden[:k]
    historial = [{"habilidad": "pick_and_place", "argumentos": {"object": o, "destination": d}, "exito": True,
                  "detalle": f"The {o} is now in the {d}."} for o, d in hechos]
    restantes = orden[k:]
    ausente = None
    if cat == "gen_tras_fallo":
        o, d = restantes[0]
        historial.append({"habilidad": "pick_and_place", "argumentos": {"object": o, "destination": d},
                          "exito": False, "detalle": f"The {o} slipped; it is not in the {d}."})
    if cat == "gen_ausente":
        ausente = restantes[0][0]
    obs = [{"proposito": "inicial", "pregunta": "objects", "respuesta": _descripcion(rng, objs + extra, ausente)}]
    if cat == "gen_completo":
        esperado = {"acciones": ["done"]}
    elif cat == "gen_ausente":
        esperado = {"acciones": ["abort", "observe"]}
    else:
        validos = restantes if conjunto else restantes[:1]
        esperado = {"acciones": ["pick_and_place"], "opciones": [{"object": o, "destination": d} for o, d in validos],
                    "ordenado": not conjunto}
    return {"perfil": "manipulacion_general", "parametros": {}, "instruccion": instr, "historial": historial,
            "observaciones": obs, "esperado": esperado, "conocidos": objs + extra, "no_pedidos": extra,
            "completados": [o for o, _ in hechos], "pendientes": {o: 1 for o, _ in restantes},
            "referencia": {"acciones": [{"action": "pick_and_place", "arguments": {"object": o, "destination": d}}
                                        for o, d in pasos]}}


# ----------------------------------------------------------------------------- prompt y contexto
def contexto_de(caso: Dict[str, Any]) -> ContextoTarea:
    p = load_profile(caso["perfil"])
    instr = p.instruction(caso["parametros"], caso["instruccion"])
    c = ContextoTarea(caso["id"], instr, copy.deepcopy(caso["parametros"]), p,
                      referencia=caso.get("referencia", {}))
    for j, h in enumerate(caso["historial"]):
        c.historial.append({"indice": j, **h})
        if h["exito"]:
            c.seguimiento.register(h["habilidad"], h["argumentos"])
    c.indice_subtarea = len(caso["historial"])
    c.observaciones = list(caso["observaciones"])
    return c


def contexto_guion(c: ContextoTarea) -> Dict[str, Any]:
    """Lo que el LLM de referencia 'guion' necesita para decidir (no lo ve ningun LLM real)."""
    return {"pendientes": c.seguimiento.remaining(), "argumento": getattr(c.seguimiento, "argument", None),
            "habilidades": [s.name for s in c.perfil.skills],
            "plan": c.referencia.get("acciones") or c.perfil.scripted_plan,
            "hechas": c.subtareas_exitosas, "completa": c.seguimiento.is_complete()}


def _respuesta_correcta(caso) -> str:
    """Una respuesta correcta del caso: la usa el modelo de referencia "guion" para
    comprobar que el banco califica bien (guion debe obtener 100 %)."""
    esp = caso["esperado"]
    if esp["acciones"][0] in ("done", "abort"):
        a = esp["acciones"][0]
        return json.dumps({"thought": "reference", "action": a, "arguments": {"reason": "object missing"} if a == "abort" else {}})
    if "objetos" in esp:
        return json.dumps({"thought": "reference", "action": "pick_and_place",
                           "arguments": {"object": esp["objetos"][0], "destination": esp.get("destino", "")}})
    o = esp["opciones"][0]
    return json.dumps({"thought": "reference", "action": "pick_and_place", "arguments": dict(o)})


def generar(vocab: Dict, comandas: Dict, por_categoria: int, semilla: int = 2026) -> List[Dict[str, Any]]:
    casos = []
    for cat in CATEGORIAS:
        for i in range(por_categoria):
            rng = random.Random(f"{semilla}|{cat}|{i}")
            caso = _caso_med(rng, cat, i, vocab, comandas) if cat.startswith("med") else _caso_gen(rng, cat, i, vocab)
            caso["id"] = f"{cat}-{i:04d}"
            caso["categoria"] = cat
            c = contexto_de(caso)
            caso["prompt"] = construir_prompt(c)
            caso["esquema"] = esquema_decision(c.perfil)
            caso["contexto_guion"] = {**contexto_guion(c), "respuesta": _respuesta_correcta(caso)}
            casos.append(caso)
    return casos
