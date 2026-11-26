#!/usr/bin/env python3
"""Calcula indicadores a partir de las corridas guardadas en resultados/.

  agente evaluar                                   # todas las corridas
  agente evaluar resultados/llm_real__qwen3.8_*    # algunas
  agente evaluar resultados/ --por-tarea -o resumen.csv

Un indicador es una funcion en evaluacion/indicadores/*.py que recibe una
Corrida y devuelve filas con fila(...). Como cada corrida guarda TODAS las
entradas y salidas de cada capa (eventos.jsonl), un indicador nuevo se calcula
sobre corridas viejas sin repetir los ensayos ni tocar el agente.
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import statistics
import sys
from collections import defaultdict

import yaml

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.dirname(AQUI)


def fila(capa, modelo, indicador, valores, unidad="", definicion=""):
    """Fila estandar: media, desviacion y n de una lista de valores (bool o numeros)."""
    vals = [float(v) for v in valores if v is not None]
    if not vals:
        return None
    return {"capa": capa, "modelo": modelo, "indicador": indicador, "valor": statistics.fmean(vals),
            "desv": statistics.pstdev(vals) if len(vals) > 1 else 0.0, "n": len(vals),
            "unidad": unidad, "definicion": definicion}


def _id_tarea(e):
    t = e.get("tarea", "")
    return t.get("id", "") if isinstance(t, dict) else t     # corridas viejas guardaban la tarea completa


def _jsonl(p):
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f if x.strip()]


class Corrida:
    """Una carpeta de resultados: configuracion, eventos y resultados por tarea."""

    def __init__(self, ruta):
        self.ruta = ruta
        p = os.path.join(ruta, "corrida.yaml")
        self.config = yaml.safe_load(open(p, encoding="utf-8")) if os.path.exists(p) else {}
        self.etiqueta = re.sub(r"_\d{8}_\d{6}$", "", os.path.basename(ruta.rstrip("/")))
        self.todos = _jsonl(os.path.join(ruta, "eventos.jsonl"))
        self.resultados = _jsonl(os.path.join(ruta, "tareas.jsonl"))

    def eventos(self, tipo=None):
        return [e for e in self.todos if tipo is None or e["tipo"] == tipo]

    def por_tarea(self):
        g = defaultdict(list)
        for e in self.todos:
            g[_id_tarea(e)].append(e)
        return g

    def solo(self, tarea_id):
        c = Corrida.__new__(Corrida)
        c.ruta, c.config, c.etiqueta = self.ruta, self.config, self.etiqueta
        c.todos = [e for e in self.todos if _id_tarea(e) == tarea_id]
        c.resultados = [r for r in self.resultados if r.get("tarea_id") == tarea_id]
        return c


def cargar_indicadores():
    funcs = []
    sys.path.insert(0, AQUI)
    for p in sorted(glob.glob(os.path.join(AQUI, "indicadores", "*.py"))):
        if os.path.basename(p).startswith("_"):
            continue
        spec = importlib.util.spec_from_file_location(f"indicadores.{os.path.basename(p)[:-3]}", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        funcs += getattr(mod, "INDICADORES", [])
    return funcs


def buscar(rutas):
    out = []
    for r in rutas:
        r = os.path.expanduser(r)
        for p in glob.glob(r) or [r]:
            if os.path.isfile(os.path.join(p, "eventos.jsonl")):
                out.append(p)
            elif os.path.isdir(p):
                out += [os.path.dirname(x) for x in glob.glob(os.path.join(p, "**", "eventos.jsonl"), recursive=True)]
    return sorted(set(out))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("rutas", nargs="*")
    ap.add_argument("--por-tarea", action="store_true", help="indicadores por tarea, ademas de por corrida")
    ap.add_argument("-o", "--salida", default="", help="CSV de salida")
    a = ap.parse_args(argv)
    rutas = a.rutas or [os.path.join(RAIZ, "resultados")]
    funcs = cargar_indicadores()
    filas = []
    for ruta in buscar(rutas):
        c = Corrida(ruta)
        unidades = [(c, "")] + ([(c.solo(t), t) for t in c.por_tarea() if t] if a.por_tarea else [])
        for cc, tid in unidades:
            for fn in funcs:
                try:
                    filas += [{"corrida": c.etiqueta, "tarea": tid, **r} for r in (fn(cc) or []) if r]
                except Exception as exc:
                    print(f"[aviso] el indicador {fn.__name__} fallo en {ruta}: {exc}", file=sys.stderr)
    if not filas:
        print("No hay corridas en", rutas)
        return 1
    enc = f"{'corrida':<34}{'capa':<13}{'modelo':<18}{'indicador':<30}{'valor':>10}{'desv':>9}{'n':>5}"
    print(enc + "\n" + "-" * len(enc))
    for r in filas:
        nombre = (r["corrida"] + (f" [{r['tarea']}]" if r["tarea"] else ""))[:33]
        print(f"{nombre:<34}{r['capa'][:12]:<13}{str(r['modelo'])[:17]:<18}{r['indicador'][:29]:<30}"
              f"{r['valor']:>10.4g}{r['desv']:>9.3g}{r['n']:>5}")
    if a.salida:
        claves = ["corrida", "tarea", "capa", "modelo", "indicador", "valor", "desv", "n", "unidad", "definicion"]
        with open(a.salida, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=claves, extrasaction="ignore")
            w.writeheader()
            w.writerows(filas)
        print(f"\nGuardado en {a.salida}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
