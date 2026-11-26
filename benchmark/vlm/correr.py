"""Ejecucion del benchmark VLM: cada modelo responde cada caso etiquetado, con cada
variante de pregunta y cada perturbacion de imagen. Usa la capa de percepcion del agente
(misma pregunta, mismo esquema JSON, misma validacion). Reanudable."""
from __future__ import annotations

import os
import time
import zlib
from typing import Any, Dict, List

from agente.capas.percepcion import CapaPercepcion
from agente.tareas.perfiles import load_profile
from agente.tipos import Observacion
from agente.util.imagen import leer_rgb
from agente.util.texto import norm, rellenar
from ..comun import config as bcfg
from ..comun.almacen import Corrida, recursos_ollama
from . import datos
from .perturbaciones import aplicar


def preguntas_de(cfg) -> Dict[str, List[Dict[str, Any]]]:
    """Plantillas de pregunta por tipo, con el esquema del perfil del agente."""
    p = load_profile(cfg.get("perfil_tarea", "empaquetado_medicamentos"))
    verif = p.skills[0].verify
    ini = p.initial_observation
    out = {"verificacion": [], "inventario": []}
    for tipo, base, esquema in (("verificacion", verif.question, verif.schema), ("inventario", ini.question, ini.schema)):
        for j, q in enumerate((cfg.get("preguntas") or {}).get(tipo, ["perfil"])):
            out[tipo].append({"id": f"q{j}", "plantilla": base if q == "perfil" else q, "esquema": esquema,
                              "es_perfil": q == "perfil"})
    return out


def texto_pregunta(plantilla: str, caso: Dict[str, Any]) -> str:
    if caso["tipo"] == "verificacion":
        return rellenar(plantilla, {"destination": caso["destino"]}, {"object": caso["objeto"],
                                                                        "destination": caso["destino"]})
    return rellenar(plantilla, {"items": dict(caso["conteos"])})


def calificar(caso: Dict[str, Any], datos_resp) -> Dict[str, Any]:
    if caso["tipo"] == "verificacion":
        pred = datos_resp.get("success") if isinstance(datos_resp, dict) else None
        pred = pred if isinstance(pred, bool) else None
        # una respuesta invalida se trata como "no" (asi actua el agente: no da la subtarea por buena)
        return {"prediccion": pred, "correcto": (pred is True) == bool(caso["esperado"])}
    items = datos_resp.get("items") if isinstance(datos_resp, dict) else None
    if not isinstance(items, list):
        return {"prediccion": None, "correcto": False, "conteos_pred": None}
    pred: Dict[str, int] = {}
    extra = []
    for it in items:
        if not isinstance(it, dict):
            continue
        nombre = str(it.get("name", ""))
        try:
            n = int(it.get("count", 0))
        except (TypeError, ValueError):
            n = 0
        clave = next((k for k in caso["conteos"] if norm(k) == norm(nombre)), None)
        if clave is None:
            extra.append(nombre)
        else:
            pred[clave] = pred.get(clave, 0) + n
    conteos = {k: pred.get(k, 0) for k in caso["conteos"]}
    return {"prediccion": conteos, "conteos_pred": conteos, "no_pedidos": extra,
            "correcto": all(conteos[k] == v for k, v in caso["conteos"].items())}


def correr(cfg: Dict[str, Any], corrida: Corrida, modelos: List[str], rapido: bool = False) -> None:
    casos = []
    for nombre in bcfg.lista(cfg["conjuntos"]):
        _, cs = datos.leer(nombre)
        for c in cs:
            c["conjunto"] = nombre
            c["clave"] = f"{nombre}/{c['id']}"
        casos += cs
    rr = cfg.get("rapido", {})
    perts = bcfg.lista(rr.get("perturbaciones", ["original"]) if rapido else cfg.get("perturbaciones", ["original"]))
    preg = preguntas_de(cfg)
    if rapido:
        casos = casos[: int(rr.get("max_casos", 12))]
        preg = {t: v[: int(rr.get("max_preguntas", 2))] for t, v in preg.items()}
    reps = 1 if rapido else int(cfg.get("repeticiones", 1))
    total = sum(len(preg[c["tipo"]]) for c in casos) * len(perts) * reps
    print(f"{len(casos)} casos etiquetados x variantes de pregunta x {len(perts)} perturbaciones x {reps} rep. "
          f"= {total} consultas por modelo")
    hechos = corrida.hechos("respuestas", ["modelo", "caso", "pregunta", "perturbacion", "rep"])
    imagenes = {}
    for modelo in modelos:
        vlm = bcfg.modelo("vlm", modelo, cfg.get("parametros_comunes", {}))
        capa = CapaPercepcion(vlm, None, cfg.get("salida_estructurada", "json"))
        trabajo = [(c, q, p, r) for c in casos for q in preg[c["tipo"]] for p in perts for r in range(reps)
                   if (modelo, c["clave"], q["id"], p, r) not in hechos]
        if not trabajo:
            continue
        print(f"\n=== {modelo} | {len(trabajo)} consultas")
        if cfg.get("calentamiento", True) and trabajo:
            c0 = trabajo[0][0]
            capa.preguntar("calentamiento", "Describe the image in one sentence.",
                           Observacion(imagenes={"front": leer_rgb(c0["_ruta"])}))
        if hasattr(vlm, "url"):
            rec = recursos_ollama(vlm.url.replace("/api/generate", ""), vlm.nombre)
            if rec:
                corrida.escribir("recursos", {"modelo": modelo, **rec})
        t0 = time.time()
        for i, (c, q, pert, rep) in enumerate(trabajo, 1):
            if c["_ruta"] not in imagenes:
                imagenes = {c["_ruta"]: leer_rgb(c["_ruta"])}       # cache de una imagen
            img = aplicar(imagenes[c["_ruta"]], pert, semilla=zlib.crc32(c["clave"].encode()) % 10000)
            pregunta = texto_pregunta(q["plantilla"], c)
            r = capa.preguntar(c["tipo"], pregunta, Observacion(imagenes={"front": img}), q["esquema"])
            cal = calificar(c, r.datos)
            corrida.escribir("respuestas", {
                "modelo": modelo, "caso": c["clave"], "conjunto": c["conjunto"], "tipo": c["tipo"],
                "grupo": c.get("grupo", c["id"]), "pregunta": q["id"], "pregunta_perfil": q["es_perfil"],
                "perturbacion": pert, "rep": rep, "esperado": c.get("esperado", c.get("conteos")),
                "formato_ok": r.formato_ok, "latencia_s": r.latencia_s, "error": r.error,
                "respuesta": (r.texto or "")[:2000], **cal})
            if i == 1 or i % 25 == 0 or i == len(trabajo):
                dt = time.time() - t0
                print(f"  {modelo}: {i}/{len(trabajo)}  ({dt / 60:.1f} min, faltan ~{dt / i * (len(trabajo) - i) / 60:.1f} min)",
                      flush=True)
