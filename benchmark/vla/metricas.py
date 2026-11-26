"""Metricas del benchmark VLA (capa de ejecucion).

FUERA DE LINEA (contra las demostraciones grabadas; unidades del conjunto: grados y 0-80 pinza):
  MAE articular por episodio (grados)                                            PRINCIPAL (menor es mejor)
  RMSE, error maximo, error por articulacion, error final (ultima accion)
  Dentro de tolerancia: fraccion de fotogramas con TODAS las articulaciones a menos de N grados
  Acierto de pinza: fraccion de fotogramas con el mismo estado abierta/cerrada que el operador
  Suavidad: jerk de la accion predicha vs. la del operador (cociente; 1 = igual de suave)
  Latencia de inferencia media / p95 / maxima y frecuencia alcanzable (Hz); memoria GPU pico
  Sensibilidad al lenguaje: cuanto cambia la accion al cambiar solo la redaccion (grados) y
  cuanto cambia el error.
LAZO CERRADO (el modelo mueve el robot; juez de exito):
  Tasa de exito (IC 95 % Wilson)                                                 PRINCIPAL
  Etapa alcanzada (exito / agarro sin colocar / no agarro), exito por escena y por redaccion,
  terminacion por estabilidad vs. tiempo agotado, duracion, pasos, ciclos de pinza (reagarres),
  recorrido articular, jerk, latencia.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict

import numpy as np

from ..comun import estadistica as est
from ..comun.almacen import leer_jsonl
from ..comun.tablas import Tabla, num, pct, valor_p


def offline(dir_corrida: str) -> Dict[str, Any]:
    eps = leer_jsonl(f"{dir_corrida}/episodios.jsonl")
    if not eps:
        return {"secciones": [], "resumen": {}}
    pm = defaultdict(list)
    for e in eps:
        pm[e["modelo"]].append(e)
    puntajes = {m: {e["episodio"]: e["mae_articular"] for e in rs} for m, rs in pm.items()}
    lat = {m: float(np.mean([e["latencia_media_s"] for e in rs])) for m, rs in pm.items()}
    rk = est.ranking(puntajes, mayor_es_mejor=False, desempate=lat)
    n = len(next(iter(puntajes.values())))
    t = Tabla(f"Ranking — fuera de línea ({n} episodios de prueba, "
              f"{sum(e['fotogramas'] for e in pm[rk[0]['modelo']])} fotogramas por modelo)",
              ["#", "Grupo", "Modelo", "MAE articular °", "RMSE °", "Dentro de tolerancia", "Acierto pinza",
               "Error final °", "Jerk pred/real", "Latencia media ms", "p95 ms", "Hz máx.", "GPU GB", "p vs. mejor"],
              "MAE con IC bootstrap por episodio. Grupo: misma letra = sin diferencia significativa en MAE por "
              "episodio (prueba de signos pareada, Holm). Hz máx. = 1 / latencia media.")
    for f in rk:
        rs = pm[f["modelo"]]
        jr = [e["jerk_pred"] / e["jerk_real"] for e in rs if e.get("jerk_real")]
        lm = float(np.mean([e["latencia_media_s"] for e in rs]))
        gpu = [e["memoria_gpu_gb"] for e in rs if e.get("memoria_gpu_gb")]
        t.agregar(f["lugar"], f["grupo"], f["modelo"], num(est.bootstrap([e["mae_articular"] for e in rs])),
                  num(est.bootstrap([e["rmse_articular"] for e in rs])),
                  num(est.bootstrap([100 * e["dentro_tolerancia"] for e in rs]), 1, unidad=" %"),
                  num(est.bootstrap([100 * e["acierto_pinza"] for e in rs if e.get("acierto_pinza") is not None]), 1, unidad=" %"),
                  num(est.bootstrap([e["error_final"] for e in rs])), num(float(np.mean(jr)) if jr else None),
                  num(1000 * lm, 1), num(1000 * float(np.mean([e["latencia_p95_s"] for e in rs])), 1),
                  num(1 / lm if lm else None, 1), num(max(gpu) if gpu else None), valor_p(f["p_vs_mejor"]))
    tablas = [t]
    nombres = rs[0].get("nombres") or []
    if nombres:
        ta = Tabla("Error medio por articulación (°)", ["Articulación"] + [f["modelo"] for f in rk])
        for j, nom in enumerate(nombres):
            ta.agregar(nom, *[num(float(np.mean([e["mae_por_articulacion"][j] for e in pm[f["modelo"]]]))) for f in rk])
        tablas.append(ta)
    leng = leer_jsonl(f"{dir_corrida}/lenguaje.jsonl")
    if leng:
        tl = Tabla("Sensibilidad al lenguaje (mismas observaciones, otra redacción)",
                   ["Modelo", "Cambio de acción °", "MAE original °", "MAE con otra redacción °"],
                   "Un cambio grande indica que el modelo depende de la redacción exacta del conjunto de entrenamiento.")
        pl = defaultdict(list)
        for x in leng:
            pl[x["modelo"]].append(x)
        for f in rk:
            xs = pl.get(f["modelo"], [])
            if xs:
                tl.agregar(f["modelo"], num(est.bootstrap([x["cambio_accion"] for x in xs])),
                           num(float(np.mean([x["mae_original"] for x in xs]))),
                           num(float(np.mean([x["mae_variante"] for x in xs]))))
        tablas.append(tl)
    return {"secciones": [{"titulo": "Fuera de línea (contra demostraciones)", "tablas": tablas}],
            "resumen": {"offline": rk}}


def lazo(dir_corrida: str) -> Dict[str, Any]:
    ens = [e for e in leer_jsonl(f"{dir_corrida}/ensayos.jsonl") if e.get("etapa") != "invalido"]
    if not ens:
        return {"secciones": [], "resumen": {}}
    pm = defaultdict(list)
    for e in ens:
        pm[e["modelo"]].append(e)
    puntajes = {m: {e["clave"]: 1.0 if e["exito"] else 0.0 for e in rs} for m, rs in pm.items()}
    dur = {m: float(np.median([e["duracion_s"] for e in rs])) for m, rs in pm.items()}
    rk = est.ranking(puntajes, desempate=dur)
    jueces = sorted({e.get("juez", "?") for e in ens})
    t = Tabla(f"Ranking — lazo cerrado (juez: {', '.join(jueces)})",
              ["#", "Grupo", "Modelo", "Éxito", "Agarró sin colocar", "No agarró / no colocó", "Terminó por estabilidad",
               "Duración s", "Pasos", "Ciclos de pinza", "Recorrido °", "Latencia ms", "p vs. mejor"],
              "Éxito con IC 95 % de Wilson. Ensayos pareados por escena inicial (misma escena k para todos los "
              "modelos). Los ensayos marcados como inválidos se excluyen."
              + (" ATENCIÓN: juez 'detector' o 'vlm' no son verdad de referencia." if set(jueces) & {"detector", "vlm"} else ""))
    for f in rk:
        rs = pm[f["modelo"]]
        t.agregar(f["lugar"], f["grupo"], f["modelo"], pct(est.proporcion([e["exito"] for e in rs])),
                  pct(est.proporcion([e["etapa"] == "agarro_sin_colocar" for e in rs]), ic=False),
                  pct(est.proporcion([e["etapa"] in ("no_agarro", "no_colocado", "tiempo_agotado") for e in rs]), ic=False),
                  pct(est.proporcion([e["terminado"] for e in rs]), ic=False),
                  num(est.bootstrap([e["duracion_s"] for e in rs]), 1),
                  num(float(np.mean([e["pasos"] for e in rs])), 0),
                  num(float(np.mean([e["ciclos_pinza"] for e in rs])), 1),
                  num(float(np.mean([e["recorrido_articular"] for e in rs])), 0),
                  num(1000 * float(np.mean([e["latencia_media_s"] or 0 for e in rs])), 1), valor_p(f["p_vs_mejor"]))
    te = Tabla("Éxito por escena inicial", ["Escena"] + [f["modelo"] for f in rk])
    for esc in list(dict.fromkeys(e["escena"] for e in ens)):
        te.agregar(esc, *[pct(est.proporcion([e["exito"] for e in pm[f["modelo"]] if e["escena"] == esc]), dec=0)
                          for f in rk])
    tv = Tabla("Éxito por redacción de la instrucción (0 = la del conjunto de datos)",
               ["Instrucción"] + [f["modelo"] for f in rk])
    for ins in list(dict.fromkeys(e["instruccion"] for e in ens)):
        tv.agregar(ins, *[pct(est.proporcion([e["exito"] for e in pm[f["modelo"]] if e["instruccion"] == ins]), dec=0)
                          for f in rk])
    return {"secciones": [{"titulo": "Lazo cerrado", "tablas": [t, te, tv]}], "resumen": {"lazo": rk}}
