"""Benchmark de modelos por capa del agente. Ver BENCHMARK.md para el procedimiento.

  python3 -m benchmark llm-casos      [-c llm] [--modelos a,b] [--rapido] [--corrida DIR]
  python3 -m benchmark llm-episodios  [-c llm] [--modelos a,b] [--rapido] [--corrida DIR]
  python3 -m benchmark vlm-datos      desde-lerobot | desde-carpeta | capturar | perturbar | revisar ...
  python3 -m benchmark vlm            [-c vlm] [--modelos a,b] [--rapido] [--corrida DIR]
  python3 -m benchmark vla-offline    [-c vla] [--modelos a,b] [--rapido] [--corrida DIR]
  python3 -m benchmark vla-lazo       [-c vla] [--modelos a,b] [--corrida DIR]
  python3 -m benchmark reporte        [carpetas de corridas ...]   (sin argumentos: la ultima de cada tipo)
  python3 -m benchmark comandas       comanda.xlsx                (regenera config/benchmark/comandas.yaml)
  (tambien:  agente benchmark <subcomando> ...)
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from agente.util.config import CONFIG, RAIZ, cargar_rutas


def _corrida(capa, a, cfg, modelos, tipo):
    from .comun.almacen import Corrida, huella
    nombre = time.strftime("%Y%m%d_%H%M%S") + ("_rapido" if getattr(a, "rapido", False) else "")
    c = {"tipo": tipo, "modelos": modelos, "rapido": getattr(a, "rapido", False), **cfg}
    c["huella"] = huella({k: v for k, v in c.items() if k not in ("modelos",)})
    return Corrida(capa, nombre, a.corrida, c)


def _comunes(p, config):
    p.add_argument("-c", "--config", default=config, help=f"config/benchmark/<nombre>.yaml (por omision {config})")
    p.add_argument("--modelos", help="lista separada por comas; por omision la del archivo de configuracion")
    p.add_argument("--rapido", action="store_true", help="poblacion pequena para comprobar que todo funciona")
    p.add_argument("--corrida", default="", help="carpeta de una corrida previa para REANUDARLA")
    p.add_argument("--set", action="append", default=[], help="clave.subclave=valor (repetible)")


def main(argv=None):
    cargar_rutas()
    os.environ.setdefault("NO_PROXY", "localhost,127.0.0.1")
    ap = argparse.ArgumentParser(prog="benchmark", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _comunes(sub.add_parser("llm-casos", help="LLM: decisiones aisladas con respuesta conocida"), "llm")
    _comunes(sub.add_parser("llm-episodios", help="LLM: tareas completas con el agente"), "llm")
    _comunes(sub.add_parser("vlm", help="VLM: verificacion e inventario contra etiquetas"), "vlm")
    _comunes(sub.add_parser("vla-offline", help="VLA: acciones contra el conjunto de datos"), "vla")
    _comunes(sub.add_parser("vla-lazo", help="VLA: ensayos en lazo cerrado (simulacion o robot)"), "vla")
    p = sub.add_parser("vlm-datos", help="construir el conjunto etiquetado del VLM", add_help=False)
    p.add_argument("resto", nargs=argparse.REMAINDER)
    p = sub.add_parser("reporte", help="REPORTE.md con el ranking de cada capa")
    p.add_argument("carpetas", nargs="*")
    p.add_argument("-o", "--salida", default="")
    p = sub.add_parser("comandas", help="convierte comanda.xlsx a config/benchmark/comandas.yaml")
    p.add_argument("xlsx", nargs="?", default=os.path.join(RAIZ, "comanda.xlsx"))
    p.add_argument("--destino", default="brown box")
    a = ap.parse_args(argv)

    if a.cmd in ("llm-casos", "llm-episodios"):
        from .comun import config as bcfg
        from .llm import correr
        cfg = bcfg.cargar(a.config, a.set)
        modelos = bcfg.lista(a.modelos) or bcfg.lista(cfg["modelos"])
        tipo = "llm_casos" if a.cmd == "llm-casos" else "llm_episodios"
        cor = _corrida(tipo, a, cfg, modelos, tipo)
        try:
            (correr.correr_casos if tipo == "llm_casos" else correr.correr_episodios)(cfg, cor, modelos, a.rapido)
        finally:
            cor.cerrar()
        print(f"\nListo. Resultados en {cor.dir}\nReporte:  python3 -m benchmark reporte {cor.dir}")
    elif a.cmd == "vlm":
        from .comun import config as bcfg
        from .vlm import correr
        cfg = bcfg.cargar(a.config, a.set)
        modelos = bcfg.lista(a.modelos) or bcfg.lista(cfg["modelos"])
        cor = _corrida("vlm", a, cfg, modelos, "vlm")
        try:
            correr.correr(cfg, cor, modelos, a.rapido)
        finally:
            cor.cerrar()
        print(f"\nListo. Resultados en {cor.dir}\nReporte:  python3 -m benchmark reporte {cor.dir}")
    elif a.cmd in ("vla-offline", "vla-lazo"):
        from .comun import config as bcfg
        cfg = bcfg.cargar(a.config, a.set)
        modelos = bcfg.lista(a.modelos) or bcfg.lista(cfg["modelos"])
        tipo = "vla_offline" if a.cmd == "vla-offline" else "vla_lazo"
        cor = _corrida(tipo, a, cfg, modelos, tipo)
        try:
            if tipo == "vla_offline":
                from .vla import offline
                offline.correr(cfg, cor, modelos, a.rapido)
            else:
                from .vla import lazo
                lazo.correr(cfg, cor, modelos, a.rapido)
        finally:
            cor.cerrar()
        print(f"\nListo. Resultados en {cor.dir}\nReporte:  python3 -m benchmark reporte {cor.dir}")
    elif a.cmd == "vlm-datos":
        from .vlm import datos
        datos.main(a.resto)
    elif a.cmd == "reporte":
        from . import reporte
        reporte.generar(a.carpetas, a.salida)
    elif a.cmd == "comandas":
        from .comun.comandas import convertir
        convertir(a.xlsx, os.path.join(CONFIG, "benchmark", "comandas.yaml"), a.destino)


if __name__ == "__main__":
    main(sys.argv[1:])
