"""Metricas del benchmark LLM (capa de razonamiento).

QUE SE MIDE (por modelo y condicion; IC 95 %):
  Exactitud de decision   fraccion de casos donde la accion propuesta es correcta.   PRINCIPAL
  Formato valido          fraccion de respuestas con JSON y campo "action".
  Errores de estado       fraccion de respuestas que dejarian la tarea en un estado incorrecto.
  Tasa por tipo de error  ver calificar.py.
  Exactitud por categoria donde falla cada modelo (pedidos grandes, tras fallo, ...).
  Estabilidad             (condiciones con varias repeticiones) fraccion de casos en que
                          todas las repeticiones dan la misma decision; y exactitud "siempre".
  Latencia                media [IC bootstrap], mediana y p95 por consulta.
  Tokens y truncadas      largo de respuesta; fraccion cortada por max_tokens.
  Memoria                 RAM/VRAM del modelo segun Ollama (/api/ps).
EPISODIOS (tareas completas):
  Exito de tarea, plan correcto (PRINCIPAL), progreso, llamadas por subtarea exitosa,
  decisiones invalidas por tarea, recuperacion (exito en tareas con fallos inyectados),
  tiempo de razonamiento por tarea.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

import numpy as np

from ..comun import estadistica as est
from ..comun.almacen import leer_jsonl
from ..comun.tablas import Tabla, num, pct, valor_p
from .calificar import ERRORES_DE_ESTADO, TIPOS_ERROR
from .casos import CATEGORIAS


def _agrupar(registros, *claves):
    g = defaultdict(list)
    for r in registros:
        g[tuple(r.get(k) for k in claves)].append(r)
    return g


def casos(dir_corrida: str) -> Dict[str, Any]:
    resp = leer_jsonl(f"{dir_corrida}/respuestas.jsonl")
    recursos = {(r["modelo"]): r for r in leer_jsonl(f"{dir_corrida}/recursos.jsonl")}
    if not resp:
        return {"secciones": [], "resumen": {}}
    secciones, resumen = [], {}
    condiciones = list(dict.fromkeys(r["condicion"] for r in resp))
    for cond in condiciones:
        rc = [r for r in resp if r["condicion"] == cond]
        por_modelo = _agrupar(rc, "modelo")
        # puntaje por caso = fraccion de repeticiones correctas (pareado por caso)
        puntajes = {}
        for (m,), rs in por_modelo.items():
            pc = defaultdict(list)
            for r in rs:
                pc[r["caso"]].append(1.0 if r["correcto"] else 0.0)
            puntajes[m] = {c: float(np.mean(v)) for c, v in pc.items()}
        lat_mediana = {m: est.percentil([r["latencia_s"] for r in rs], 50) for (m,), rs in por_modelo.items()}
        rk = est.ranking(puntajes, desempate=lat_mediana)
        reps = max(r["rep"] for r in rc) + 1
        t = Tabla(f"Ranking — condición `{cond}` (exactitud de decisión, {len(puntajes[rk[0]['modelo']])} casos"
                  f"{f' × {reps} repeticiones' if reps > 1 else ''})",
                  ["#", "Grupo", "Modelo", "Exactitud", "Formato válido", "Errores de estado", "Respuestas vacías",
                   "Latencia media s", "Latencia p95 s", "Tokens resp.", "Truncadas", "p vs. mejor"],
                  "Grupo: modelos con la misma letra no difieren significativamente (prueba de signos pareada, "
                  "Holm, α=0.05). Proporciones con IC 95 % de Wilson; latencia media con IC bootstrap.")
        for f in rk:
            rs = por_modelo[(f["modelo"],)]
            t.agregar(f["lugar"], f["grupo"], f["modelo"], pct(est.proporcion([r["correcto"] for r in rs])),
                      pct(est.proporcion([r["formato_ok"] for r in rs]), ic=False),
                      pct(est.proporcion([r["error"] in ERRORES_DE_ESTADO for r in rs])),
                      pct(est.proporcion([r["error"] == "respuesta_vacia" for r in rs]), ic=False),
                      num(est.bootstrap([r["latencia_s"] for r in rs])),
                      num(est.percentil([r["latencia_s"] for r in rs], 95)),
                      num(float(np.mean([r.get("tokens_respuesta") or 0 for r in rs])), 0),
                      pct(est.proporcion([r.get("truncado") for r in rs if "truncado" in r]), ic=False),
                      valor_p(f["p_vs_mejor"]))
        tablas = [t]
        # exactitud por categoria
        tc = Tabla(f"Exactitud por categoría — `{cond}`", ["Categoría"] + [f["modelo"] for f in rk],
                   "Cada celda: exactitud [IC 95 %]. Ver BENCHMARK.md para la definición de cada categoría.")
        for cat in CATEGORIAS:
            fila = [cat]
            for f in rk:
                rs = [r for r in por_modelo[(f["modelo"],)] if r["categoria"] == cat]
                fila.append(pct(est.proporcion([r["correcto"] for r in rs]), dec=0))
            if any(c != "–" for c in fila[1:]):
                tc.agregar(*fila)
        tablas.append(tc)
        # errores por tipo
        te = Tabla(f"Errores por tipo — `{cond}` (porcentaje de respuestas)", ["Error"] + [f["modelo"] for f in rk])
        for tipo in TIPOS_ERROR:
            fila = [tipo]
            hay = False
            for f in rk:
                rs = por_modelo[(f["modelo"],)]
                k = sum(1 for r in rs if r["error"] == tipo)
                hay = hay or k > 0
                fila.append(f"{100 * k / len(rs):.1f} %" if rs else "–")
            if hay:
                te.agregar(*fila)
        tablas.append(te)
        # estabilidad
        if reps > 1:
            ts = Tabla(f"Estabilidad — `{cond}` ({reps} repeticiones por caso)",
                       ["Modelo", "Misma decisión en todas", "Correcto en todas", "Correcto en alguna"])
            for f in rk:
                pc = defaultdict(list)
                for r in por_modelo[(f["modelo"],)]:
                    pc[r["caso"]].append(r)
                igual = [len({(x["accion"], str(x.get("objeto", "")).lower()) for x in v}) == 1 for v in pc.values()]
                todas = [all(x["correcto"] for x in v) for v in pc.values()]
                alguna = [any(x["correcto"] for x in v) for v in pc.values()]
                ts.agregar(f["modelo"], pct(est.proporcion(igual)), pct(est.proporcion(todas)),
                           pct(est.proporcion(alguna)))
            tablas.append(ts)
        resumen[cond] = rk
        secciones.append({"titulo": f"Casos de decisión — {cond}", "tablas": tablas})
    if recursos:
        tr = Tabla("Recursos de cada modelo (Ollama /api/ps)", ["Modelo", "Parámetros", "Cuantización",
                                                              "Memoria GB", "VRAM GB"])
        for m, r in recursos.items():
            tr.agregar(m, r.get("parametros"), r.get("cuantizacion"), r.get("memoria_gb"), r.get("vram_gb"))
        secciones.append({"titulo": "Recursos", "tablas": [tr]})
    return {"secciones": secciones, "resumen": resumen}


def episodios(dir_corrida: str) -> Dict[str, Any]:
    eps = leer_jsonl(f"{dir_corrida}/episodios.jsonl")
    if not eps:
        return {"secciones": [], "resumen": {}}
    secciones, resumen = [], {}
    for cond in list(dict.fromkeys(e["condicion"] for e in eps)):
        ec = [e for e in eps if e["condicion"] == cond]
        pm = _agrupar(ec, "modelo")
        puntajes = {m: {f"{e['tarea']}|{e['rep']}": 1.0 if e["plan_correcto"] or (e["plan_correcto"] is None and e["exito"])
                        else 0.0 for e in rs} for (m,), rs in pm.items()}
        t_med = {m: float(np.median([e["tiempo_razonamiento_s"] for e in rs])) for (m,), rs in pm.items()}
        rk = est.ranking(puntajes, desempate=t_med)
        n = len(next(iter(puntajes.values())))
        t = Tabla(f"Ranking — episodios `{cond}` (plan correcto, {n} tareas×repeticiones por modelo)",
                  ["#", "Grupo", "Modelo", "Plan correcto", "Éxito", "Progreso", "Llamadas/subtarea",
                   "Decisiones inválidas/tarea", "Recuperación tras fallo", "Razonamiento s/tarea", "p vs. mejor"],
                  "Plan correcto: hizo exactamente lo pedido (sin pasos de más, de menos ni fuera de orden). "
                  "Recuperación: éxito en tareas donde se inyectó al menos un fallo de subtarea.")
        for f in rk:
            rs = pm[(f["modelo"],)]
            con_fallo = [e for e in rs if e["fallos_inyectados"] > 0]
            t.agregar(f["lugar"], f["grupo"], f["modelo"],
                      pct(est.proporcion([e["plan_correcto"] if e["plan_correcto"] is not None else e["exito"] for e in rs])),
                      pct(est.proporcion([e["exito"] for e in rs])),
                      num(est.bootstrap([e["progreso"] for e in rs if e["progreso"] >= 0])),
                      num(est.bootstrap([e["llamadas_llm"] / e["subtareas_exitosas"] for e in rs if e["subtareas_exitosas"]])),
                      num(est.bootstrap([e["decisiones_invalidas"] for e in rs])),
                      pct(est.proporcion([e["exito"] for e in con_fallo])) if con_fallo else "–",
                      num(est.bootstrap([e["tiempo_razonamiento_s"] for e in rs]), 1),
                      valor_p(f["p_vs_mejor"]))
        tt = Tabla(f"Plan correcto por tipo de tarea — `{cond}`", ["Tipo"] + [f["modelo"] for f in rk],
                   "real: pedidos de comanda.xlsx; gen: pedidos generados; instr: instrucciones libres.")
        for tipo in ("real", "gen", "instr"):
            fila = [tipo]
            for f in rk:
                rs = [e for e in pm[(f["modelo"],)] if e["tipo"] == tipo]
                fila.append(pct(est.proporcion([e["plan_correcto"] if e["plan_correcto"] is not None else e["exito"]
                                                for e in rs]), dec=0))
            tt.agregar(*fila)
        tm = Tabla(f"Motivos de rechazo de decisiones — `{cond}` (total en todas las tareas)",
                   ["Motivo"] + [f["modelo"] for f in rk])
        motivos = Counter()
        for e in ec:
            motivos.update(e["motivos"])
        for mot, _ in motivos.most_common():
            tm.agregar(mot, *[sum(e["motivos"].get(mot, 0) for e in pm[(f["modelo"],)]) for f in rk])
        resumen[cond] = rk
        secciones.append({"titulo": f"Episodios — {cond}", "tablas": [t, tt, tm]})
    return {"secciones": secciones, "resumen": resumen}
