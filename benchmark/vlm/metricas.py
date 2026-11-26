"""Metricas del benchmark VLM (capa de percepcion).

VERIFICACION (¿la subtarea se cumplio?; clasificacion binaria contra la etiqueta):
  Exactitud balanceada  (sensibilidad + especificidad) / 2; no se infla si hay mas casos
                        de una clase que de otra.                                   PRINCIPAL
  Exactitud, precision, sensibilidad (recall), especificidad, F1, MCC.
  Falsos positivos (FPR) dice "si" cuando la subtarea FALLO: el agente avanza con un error
                         sin reintentar. Es el error mas costoso para el agente.
  Falsos negativos (FNR) dice "no" cuando SI se cumplio: reintentos innecesarios.
  Sin respuesta         respuesta sin JSON valido (el agente lo trata como fallo).
INVENTARIO (¿cuantas cajas de cada medicamento hay?):
  Conteo exacto por imagen (todas las cantidades correctas)                        PRINCIPAL
  Conteo exacto por elemento, error absoluto medio (MAE), deteccion de presencia
  (precision / sensibilidad / F1), alucinaciones (reporta cantidad > 0 de algo ausente),
  omisiones (reporta 0 de algo presente), nombres no pedidos.
ROBUSTEZ Y CONSISTENCIA:
  Exactitud por perturbacion de la imagen, invariancia (misma respuesta que con la imagen
  original), consistencia entre parafrasis de la pregunta.
COSTO: latencia media, mediana y p95; formato valido; memoria (Ollama).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any, Dict, List

import numpy as np

from ..comun import estadistica as est
from ..comun.almacen import leer_jsonl
from ..comun.tablas import Tabla, num, pct, valor_p
from .perturbaciones import DESCRIPCION


def matriz(rs: List[Dict]) -> Dict[str, float]:
    tp = sum(1 for r in rs if r["prediccion"] is True and r["esperado"] is True)
    tn = sum(1 for r in rs if r["prediccion"] is False and r["esperado"] is False)
    fp = sum(1 for r in rs if r["prediccion"] is True and r["esperado"] is False)
    fn = sum(1 for r in rs if r["prediccion"] is not True and r["esperado"] is True)   # sin respuesta = no
    fn_sin = sum(1 for r in rs if r["prediccion"] is None and r["esperado"] is False)   # cuenta como negativo
    tn += fn_sin
    pos, neg = tp + fn, tn + fp
    sens = tp / pos if pos else float("nan")
    esp = tn / neg if neg else float("nan")
    prec = tp / (tp + fp) if tp + fp else float("nan")
    f1 = 2 * prec * sens / (prec + sens) if prec == prec and sens == sens and prec + sens else float("nan")
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (tp * tn - fp * fn) / den if den else float("nan")
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "sensibilidad": sens, "especificidad": esp, "precision": prec,
            "f1": f1, "mcc": mcc, "bal": (sens + esp) / 2 if sens == sens and esp == esp else float("nan"),
            "pos": pos, "neg": neg}


def _bal_ic(rs, b=1000, semilla=0):
    """IC bootstrap de la exactitud balanceada, remuestreando casos (imagenes)."""
    por_caso = defaultdict(list)
    for r in rs:
        por_caso[r["caso"]].append(r)
    claves = list(por_caso)
    if len(claves) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(semilla)
    vals = []
    for _ in range(b):
        muestra = [x for k in rng.choice(claves, len(claves)) for x in por_caso[k]]
        v = matriz(muestra)["bal"]
        if v == v:
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2


def _f(x, d=3):
    return "–" if x is None or x != x else f"{x:.{d}f}"


def _p100(x):
    return "–" if x is None or x != x else f"{100 * x:.1f} %"


def calcular(dir_corrida: str) -> Dict[str, Any]:
    resp = leer_jsonl(f"{dir_corrida}/respuestas.jsonl")
    if not resp:
        return {"secciones": [], "resumen": {}}
    secciones, resumen = [], {}
    modelos = list(dict.fromkeys(r["modelo"] for r in resp))
    rec = {r["modelo"]: r for r in leer_jsonl(f"{dir_corrida}/recursos.jsonl")}

    # ------------------------------------------------------------------ verificacion
    ver = [r for r in resp if r["tipo"] == "verificacion"]
    if ver:
        pm = defaultdict(list)
        for r in ver:
            pm[r["modelo"]].append(r)
        puntajes = {}
        for m, rs in pm.items():
            pc = defaultdict(list)
            for r in rs:
                pc[r["caso"]].append(1.0 if r["correcto"] else 0.0)
            puntajes[m] = {k: float(np.mean(v)) for k, v in pc.items()}
        lat = {m: est.percentil([r["latencia_s"] for r in rs], 50) for m, rs in pm.items()}
        bal = {m: matriz(rs)["bal"] for m, rs in pm.items()}
        # el ranking se ordena por exactitud balanceada; grupos con prueba pareada por imagen
        rk = est.ranking(puntajes, desempate=lat)
        rk.sort(key=lambda f: (-(bal[f["modelo"]] if bal[f["modelo"]] == bal[f["modelo"]] else -1), lat[f["modelo"]]))
        for i, f in enumerate(rk):
            f["lugar"] = i + 1
        ncasos = len(next(iter(puntajes.values())))
        t = Tabla(f"Ranking — verificación de subtareas ({ncasos} imágenes etiquetadas; "
                  f"{len(pm[rk[0]['modelo']])} consultas por modelo con parafrasis y perturbaciones)",
                  ["#", "Grupo", "Modelo", "Exactitud balanceada", "Exactitud", "Sensibilidad", "Especificidad",
                   "Falsos positivos", "F1", "MCC", "Sin respuesta", "Latencia mediana s", "p95 s", "p vs. mejor"],
                  "Exactitud balanceada con IC bootstrap por imagen; exactitud con IC de Wilson. Falsos positivos = "
                  "fracción de subtareas FALLIDAS que el modelo da por buenas (el agente no reintentaría). Grupo: "
                  "misma letra = sin diferencia significativa en acierto por imagen (prueba de signos, Holm).")
        for f in rk:
            rs = pm[f["modelo"]]
            mz = matriz(rs)
            lo, hi = _bal_ic(rs)
            t.agregar(f["lugar"], f["grupo"], f["modelo"], f"{100 * mz['bal']:.1f} % [{100 * lo:.1f}–{100 * hi:.1f}]",
                      pct(est.proporcion([r["correcto"] for r in rs])), _p100(mz["sensibilidad"]),
                      _p100(mz["especificidad"]), _p100(1 - mz["especificidad"] if mz["especificidad"] == mz["especificidad"] else None),
                      _f(mz["f1"]), _f(mz["mcc"]), pct(est.proporcion([r["prediccion"] is None for r in rs]), ic=False),
                      num(est.percentil([r["latencia_s"] for r in rs], 50)),
                      num(est.percentil([r["latencia_s"] for r in rs], 95)), valor_p(f["p_vs_mejor"]))
        tablas = [t]
        # matriz de confusion
        tm = Tabla("Matriz de confusión — verificación (todas las consultas)",
                   ["Modelo", "VP (sí y era sí)", "FN (no o sin respuesta, era sí)", "FP (sí, era no)",
                    "VN (no, era no)", "Positivos", "Negativos"])
        for f in rk:
            mz = matriz(pm[f["modelo"]])
            tm.agregar(f["modelo"], mz["tp"], mz["fn"], mz["fp"], mz["tn"], mz["pos"], mz["neg"])
        tablas.append(tm)
        # robustez por perturbacion
        perts = list(dict.fromkeys(r["perturbacion"] for r in ver))
        if len(perts) > 1:
            tp_ = Tabla("Robustez — exactitud balanceada por perturbación de imagen", ["Perturbación", "Descripción"] +
                        [f["modelo"] for f in rk])
            for p in perts:
                tp_.agregar(p, DESCRIPCION.get(p, ""), *[_p100(matriz([r for r in pm[f["modelo"]] if r["perturbacion"] == p])["bal"])
                                                         for f in rk])
            tablas.append(tp_)
            ti = Tabla("Invariancia — misma respuesta que con la imagen original", ["Modelo", "Invariancia"])
            for f in rk:
                base = {(r["caso"], r["pregunta"], r["rep"]): r["prediccion"] for r in pm[f["modelo"]] if r["perturbacion"] == "original"}
                igual = [r["prediccion"] == base[(r["caso"], r["pregunta"], r["rep"])] for r in pm[f["modelo"]]
                         if r["perturbacion"] != "original" and (r["caso"], r["pregunta"], r["rep"]) in base]
                ti.agregar(f["modelo"], pct(est.proporcion(igual)))
            tablas.append(ti)
        pregs = list(dict.fromkeys(r["pregunta"] for r in ver))
        if len(pregs) > 1:
            tq = Tabla("Sensibilidad a la redacción — exactitud balanceada por variante de pregunta (q0 = la del agente)",
                       ["Pregunta"] + [f["modelo"] for f in rk] )
            for q in pregs:
                tq.agregar(q, *[_p100(matriz([r for r in pm[f["modelo"]] if r["pregunta"] == q])["bal"]) for f in rk])
            tc = Tabla("Consistencia entre paráfrasis (imagen original): misma respuesta con todas las redacciones",
                       ["Modelo", "Consistencia"])
            for f in rk:
                g = defaultdict(set)
                for r in pm[f["modelo"]]:
                    if r["perturbacion"] == "original":
                        g[(r["caso"], r["rep"])].add(r["prediccion"])
                tc.agregar(f["modelo"], pct(est.proporcion([len(v) == 1 for v in g.values()])))
            tablas += [tq, tc]
        filas_res = []
        for f in rk:
            lo, hi = _bal_ic(pm[f["modelo"]])
            filas_res.append({**f, "media": bal[f["modelo"]], "ic_bajo": lo, "ic_alto": hi})
        resumen["verificacion"] = filas_res
        secciones.append({"titulo": "Verificación de subtareas", "tablas": tablas})

    # ------------------------------------------------------------------ inventario
    inv = [r for r in resp if r["tipo"] == "inventario"]
    if inv:
        pm = defaultdict(list)
        for r in inv:
            pm[r["modelo"]].append(r)
        puntajes = {}
        for m, rs in pm.items():
            pc = defaultdict(list)
            for r in rs:
                pc[r["caso"]].append(1.0 if r["correcto"] else 0.0)
            puntajes[m] = {k: float(np.mean(v)) for k, v in pc.items()}
        lat = {m: est.percentil([r["latencia_s"] for r in rs], 50) for m, rs in pm.items()}
        rk = est.ranking(puntajes, desempate=lat)
        t = Tabla(f"Ranking — inventario ({len(next(iter(puntajes.values())))} imágenes etiquetadas)",
                  ["#", "Grupo", "Modelo", "Conteo exacto (imagen)", "Conteo exacto (elemento)", "MAE cajas",
                   "Presencia F1", "Alucinaciones", "Omisiones", "Nombres no pedidos", "Sin respuesta",
                   "Latencia mediana s", "p vs. mejor"],
                  "Conteo exacto (imagen): todas las cantidades correctas. Alucinación: reporta cajas de un "
                  "medicamento ausente. Omisión: reporta 0 de uno presente.")
        for f in rk:
            rs = pm[f["modelo"]]
            elem, err, tp, fp, fn, aluc, omis = [], [], 0, 0, 0, [], []
            for r in rs:
                pred = r.get("conteos_pred")
                for k, v in (r["esperado"] or {}).items():
                    pv = pred.get(k, 0) if pred else None
                    elem.append(pv == v if pv is not None else False)
                    err.append(abs((pv or 0) - v))
                    presente_pred = bool(pv) if pv is not None else False
                    tp += presente_pred and v > 0
                    fp += presente_pred and v == 0
                    fn += (not presente_pred) and v > 0
                    if v == 0:
                        aluc.append(presente_pred)
                    else:
                        omis.append(pv == 0 if pv is not None else True)
            prec = tp / (tp + fp) if tp + fp else float("nan")
            sens = tp / (tp + fn) if tp + fn else float("nan")
            f1 = 2 * prec * sens / (prec + sens) if prec == prec and sens == sens and prec + sens else float("nan")
            t.agregar(f["lugar"], f["grupo"], f["modelo"], pct(est.proporcion([r["correcto"] for r in rs])),
                      pct(est.proporcion(elem)), num(est.bootstrap(err)), _f(f1),
                      pct(est.proporcion(aluc)) if aluc else "–", pct(est.proporcion(omis)) if omis else "–",
                      pct(est.proporcion([bool(r.get("no_pedidos")) for r in rs]), ic=False),
                      pct(est.proporcion([r["prediccion"] is None for r in rs]), ic=False),
                      num(est.percentil([r["latencia_s"] for r in rs], 50)), valor_p(f["p_vs_mejor"]))
        tablas = [t]
        perts = list(dict.fromkeys(r["perturbacion"] for r in inv))
        if len(perts) > 1:
            tp_ = Tabla("Robustez — conteo exacto por perturbación", ["Perturbación"] + [f["modelo"] for f in rk])
            for p in perts:
                tp_.agregar(p, *[pct(est.proporcion([r["correcto"] for r in pm[f["modelo"]] if r["perturbacion"] == p]), dec=0)
                                 for f in rk])
            tablas.append(tp_)
        resumen["inventario"] = rk
        secciones.append({"titulo": "Inventario", "tablas": tablas})
    if rec:
        tr = Tabla("Recursos de cada modelo (Ollama /api/ps)", ["Modelo", "Parámetros", "Cuantización", "Memoria GB", "VRAM GB"])
        for m, r in rec.items():
            tr.agregar(m, r.get("parametros"), r.get("cuantizacion"), r.get("memoria_gb"), r.get("vram_gb"))
        secciones.append({"titulo": "Recursos", "tablas": [tr]})
    return {"secciones": secciones, "resumen": resumen}
