"""Ejecucion del benchmark LLM.

  casos      cada modelo responde todos los casos de decision, en cada condicion y repeticion
  episodios  cada modelo resuelve tareas completas con el agente real (percepcion = oraculo
             de escena, ejecucion simulada), con y sin fallos inyectados

Las corridas son REANUDABLES: si se interrumpen, al relanzar el mismo comando con
--corrida <carpeta> se saltan las respuestas ya guardadas.
"""
from __future__ import annotations

import json
import os
import random
import time
from typing import Any, Dict, List

from agente.agente import Agente
from agente.registro import Registro
from agente.tipos import Tarea
from agente.util.texto import norm
from ..comun import config as bcfg
from ..comun.almacen import Corrida, huella, leer_jsonl, recursos_ollama
from .calificar import calificar
from .casos import generar


def _progreso(i, total, t0, prefijo):
    if i == 1 or i % 25 == 0 or i == total:
        dt = time.time() - t0
        eta = dt / i * (total - i) if i else 0
        print(f"  {prefijo}: {i}/{total}  ({dt / 60:.1f} min, faltan ~{eta / 60:.1f} min)", flush=True)


def _llamar(llm, prompt, esquema, contexto, reintentos=1):
    for k in range(reintentos + 1):
        try:
            if hasattr(llm, "ultimo"):
                llm.ultimo = {}
            t0 = time.perf_counter()
            texto = llm.generar(prompt, esquema=esquema, contexto=contexto)
            return texto, time.perf_counter() - t0, ""
        except Exception as exc:
            if k == reintentos:
                return "", 0.0, f"{type(exc).__name__}: {exc}"
            time.sleep(5)


# ============================================================================ casos de decision
def correr_casos(cfg: Dict[str, Any], corrida: Corrida, modelos: List[str], rapido: bool = False) -> None:
    cc = cfg["casos"]
    n = int(cfg.get("rapido", {}).get("por_categoria", 5)) if rapido else int(cc["por_categoria"])
    ruta_casos = corrida.ruta("casos.jsonl")
    casos = leer_jsonl(ruta_casos)
    if not casos:
        casos = generar(bcfg.vocabulario(cc.get("objetos", "objetos")), bcfg.vocabulario(cc.get("comandas", "comandas")),
                        n, int(cc.get("semilla", 2026)))
        for c in casos:
            corrida.escribir("casos", c)
    print(f"{len(casos)} casos ({len(casos) // n} categorias x {n}) en {corrida.dir}")
    hechos = corrida.hechos("respuestas", ["modelo", "condicion", "rep", "caso"])
    comunes = dict(cfg.get("parametros_comunes", {}))
    for modelo in modelos:
        for cond in cfg["condiciones"]:
            reps = int(cond.get("repeticiones", 1))
            if rapido:
                reps = min(reps, int(cfg.get("rapido", {}).get("repeticiones_maximas", 1)))
            pendientes = [(r, c) for r in range(reps) for c in casos
                          if (modelo, cond["nombre"], r, c["id"]) not in hechos]
            if not pendientes:
                continue
            params = {**comunes, "temperature": cond.get("temperature", 0.0)}
            for k in ("think", "max_tokens", "top_p"):
                if k in cond:
                    params[k] = cond[k]
            llm = bcfg.modelo("llm", modelo, params)
            print(f"\n=== {modelo} | {cond['nombre']} | {len(pendientes)} llamadas")
            if cfg.get("calentamiento", True) and hasattr(llm, "url"):
                _llamar(llm, "Reply with the JSON {\"ok\": true}", "json", {})
            rec = recursos_ollama(os.path.dirname(getattr(llm, "url", "")).replace("/api", ""), getattr(llm, "nombre", ""))
            if rec:
                corrida.escribir("recursos", {"modelo": modelo, "condicion": cond["nombre"], **rec})
            sal = cond.get("salida_estructurada", "json")
            t0 = time.time()
            for i, (rep, caso) in enumerate(pendientes, 1):
                esquema = caso["esquema"] if sal == "schema" else ("json" if sal == "json" else None)
                texto, lat, err = _llamar(llm, caso["prompt"], esquema, caso["contexto_guion"])
                meta = dict(getattr(llm, "ultimo", {}) or {})
                cal = calificar(caso, texto) if not err else {"correcto": False, "error": "error_servidor",
                                                              "formato_ok": False, "accion": ""}
                corrida.escribir("respuestas", {
                    "modelo": modelo, "condicion": cond["nombre"], "rep": rep, "caso": caso["id"],
                    "categoria": caso["categoria"], "perfil": caso["perfil"], "temperature": params["temperature"],
                    "salida_estructurada": sal, "latencia_s": lat, **cal, "detalle_error": err,
                    "largo_respuesta": len(texto or ""), "respuesta": (texto or "")[:3000], **meta})
                _progreso(i, len(pendientes), t0, f"{modelo}/{cond['nombre']}")


# ============================================================================ episodios
def _tareas_episodios(cfg, semilla, rapido) -> List[Tarea]:
    ce = cfg["episodios"]
    vocab = bcfg.vocabulario(cfg["casos"].get("objetos", "objetos"))
    comandas = bcfg.vocabulario(cfg["casos"].get("comandas", "comandas"))
    rr = cfg.get("rapido", {})
    n_ped = int(rr.get("pedidos_generados", 2)) if rapido else int(ce.get("pedidos_generados", 24))
    n_gen = int(rr.get("instrucciones_generales", 2)) if rapido else int(ce.get("instrucciones_generales", 24))
    destino = comandas.get("destino", vocab.get("destino_medicamentos", "brown box"))
    tareas = []
    for p in comandas.get("pedidos", [])[: (2 if rapido else None)]:
        tareas.append(Tarea(id=f"real-{p['id']}", perfil="empaquetado_medicamentos", solicitante=p.get("solicitante", ""),
                            parametros={"items": dict(p["items"]), "destination": destino}))
    for i in range(n_ped):
        rng = random.Random(f"{semilla}|pedido|{i}")
        meds = rng.sample(vocab["medicamentos"], rng.randint(1, len(vocab["medicamentos"])))
        items = {m: rng.randint(1, int(vocab.get("cantidad_maxima", 5))) for m in meds}
        tareas.append(Tarea(id=f"gen-{i:03d}", perfil="empaquetado_medicamentos",
                            parametros={"items": items, "destination": destino}))
    from .casos import _instruccion
    for i in range(n_gen):
        rng = random.Random(f"{semilla}|instr|{i}")
        n = rng.randint(1, 4)
        objs = rng.sample(vocab["objetos"], n)
        conjunto = n > 1 and rng.random() < 0.3
        d0 = rng.choice(vocab["destinos"])
        pasos = [(o, d0 if conjunto else rng.choice(vocab["destinos"])) for o in objs]
        tareas.append(Tarea(id=f"instr-{i:03d}", perfil="manipulacion_general",
                            instruccion=_instruccion(rng, pasos, vocab, conjunto),
                            referencia={"ordenado": not conjunto, "acciones": [
                                {"action": "pick_and_place", "arguments": {"object": o, "destination": d}}
                                for o, d in pasos]}))
    return tareas


def plan_correcto(subtareas_ok: List[Dict], tarea: Dict, resultado: Dict) -> Any:
    """Hizo exactamente lo pedido. Medicamentos: todas las unidades, ninguna de mas.
    Instrucciones: los pasos de la referencia (en orden si la instruccion lo exige)."""
    ref = (tarea.get("referencia") or {}).get("acciones")
    if ref:
        hechos = [(norm(s["argumentos"].get("object", s["argumentos"].get("instruction", ""))),
                   norm(s["argumentos"].get("destination", ""))) for s in subtareas_ok]
        esperados = [(norm(a["arguments"]["object"]), norm(a["arguments"]["destination"])) for a in ref]
        coincide = lambda h, e: (e[0] in h[0] or h[0] in e[0]) and (not h[1] or e[1] in h[1] or h[1] in e[1])
        if len(hechos) != len(esperados):
            return False
        if (tarea.get("referencia") or {}).get("ordenado", True):
            return all(coincide(h, e) for h, e in zip(hechos, esperados)) and bool(resultado.get("exito"))
        libres = list(esperados)
        for h in hechos:
            k = next((j for j, e in enumerate(libres) if coincide(h, e)), None)
            if k is None:
                return False
            libres.pop(k)
        return bool(resultado.get("exito"))
    if resultado.get("unidades_requeridas", -1) > 0:
        return bool(resultado.get("exito")) and resultado.get("subtareas_exitosas") == resultado["unidades_requeridas"]
    return None


def _resumen_episodio(eventos: List[Dict], tarea: Tarea, r) -> Dict[str, Any]:
    llm = [e for e in eventos if e["tipo"] == "llm"]
    sub = [e for e in eventos if e["tipo"] == "subtarea"]
    motivos: Dict[str, int] = {}
    for e in llm:
        if e.get("motivo"):
            m = str(e["motivo"]).split(":")[0]
            motivos[m] = motivos.get(m, 0) + 1
    fallos = sum(1 for s in sub if not s.get("exito"))
    d = r.__dict__ if hasattr(r, "__dict__") else dict(r)
    return {
        "exito": d["exito"], "estado": d["estado"], "detalle": d["detalle"][:300], "duracion_s": d["duracion_s"],
        "subtareas": d["subtareas"], "subtareas_exitosas": d["subtareas_exitosas"],
        "unidades_requeridas": d["unidades_requeridas"], "reintentos": d["reintentos"], "progreso": d["progreso"],
        "llamadas_llm": len(llm), "decisiones_invalidas": sum(1 for e in llm if not e.get("valida")),
        "formato_invalido": sum(1 for e in llm if not e.get("formato_ok")),
        "observaciones_pedidas": sum(1 for e in llm if e.get("accion") == "observe" and e.get("valida")),
        "motivos": motivos, "fallos_inyectados": fallos,
        "plan_correcto": plan_correcto([s for s in sub if s.get("exito")], tarea.como_dict(), d),
        "tiempo_razonamiento_s": sum(e.get("latencia_s", 0) for e in llm),
        "tokens_respuesta": sum(e.get("tokens_respuesta") or 0 for e in llm),
        "truncadas": sum(1 for e in llm if e.get("truncado")),
    }


def correr_episodios(cfg: Dict[str, Any], corrida: Corrida, modelos: List[str], rapido: bool = False) -> None:
    ce = cfg["episodios"]
    semilla = int(ce.get("semilla", 2026))
    tareas = _tareas_episodios(cfg, semilla, rapido)
    vocab = bcfg.vocabulario(cfg["casos"].get("objetos", "objetos"))
    reps = 1 if rapido else int(ce.get("repeticiones", 3))
    print(f"{len(tareas)} tareas x {len(ce['condiciones'])} condiciones x {reps} repeticiones por modelo")
    hechos = corrida.hechos("episodios", ["modelo", "condicion", "rep", "tarea"])
    comunes = {**cfg.get("parametros_comunes", {}), "temperature": ce.get("temperature", 0.0)}
    for modelo in modelos:
        spec = bcfg.especificacion("llm", modelo, comunes)
        for cond in ce["condiciones"]:
            for rep in range(reps):
                faltan = [t for t in tareas if (modelo, cond["nombre"], rep, t.id) not in hechos]
                if not faltan:
                    continue
                vlm = {"backend": "escena", "etiqueta": f"escena_{cond['nombre']}", "tasa_exito": cond["tasa_exito"],
                       "semilla": semilla * 100 + rep,
                       "distractores": random.Random(semilla).sample(vocab.get("medicamentos_distractores", []),
                                                                     k=min(int(ce.get("distractores", 0)),
                                                                           len(vocab.get("medicamentos_distractores", []))))}
                acfg = {"nombre": f"llm_episodios_{bcfg.nombre_archivo(modelo)}", "modo": "modular",
                        "modelos": {"llm": spec, "vlm": vlm, "vla": "simulado"},
                        "entorno": {"tipo": "simulado", "robot": "omx", "tiempo_real": False},
                        "razonamiento": {"salida_estructurada": ce.get("salida_estructurada", "json"),
                                         "max_reintentos_formato": 2},
                        "percepcion": {"salida_estructurada": "json", "verificar_subtareas": True},
                        "ejecucion": {"umbral_estable": 3.0, "ciclos_estables": 5, "min_pasos": 3, "max_duracion_s": 20},
                        "agente": {"max_reintentos": ce.get("max_reintentos", 3),
                                   "max_llamadas_llm": ce.get("max_llamadas_llm", 80)}}
                reg = Registro(f"{bcfg.nombre_archivo(modelo)}__{cond['nombre']}__r{rep}",
                               corrida.ruta("registros"), acfg)
                agente = Agente(acfg, reg)
                for m in (agente.percepcion.vlm, agente.ejecucion.vla):   # escena limpia, mismos fallos
                    if hasattr(m, "reiniciar_semilla"):
                        m.reiniciar_semilla()
                print(f"\n=== {modelo} | {cond['nombre']} | rep {rep} | {len(faltan)} tareas")
                t0 = time.time()
                for i, t in enumerate(faltan, 1):
                    r = agente.resolver(t)
                    eventos = [e for e in leer_jsonl(os.path.join(reg.dir, "eventos.jsonl")) if e.get("tarea") == t.id]
                    corrida.escribir("episodios", {"modelo": modelo, "condicion": cond["nombre"], "rep": rep,
                                                   "tarea": t.id, "perfil": t.perfil,
                                                   "tipo": t.id.split("-")[0], "registro": os.path.relpath(reg.dir, corrida.dir),
                                                   **_resumen_episodio(eventos, t, r)})
                    _progreso(i, len(faltan), t0, f"{modelo}/{cond['nombre']}/r{rep}")
                reg.cerrar()
                agente.cerrar()


def configuracion_corrida(cfg, modelos, tipo, rapido):
    c = {"tipo": tipo, "modelos": modelos, "rapido": rapido, **cfg}
    c["huella"] = huella({k: v for k, v in c.items() if k != "modelos"})
    return c
