"""Linea de comandos del banco de pruebas.

  agente ejecutar  -x <experimento> -e <ensayo> [-r N] [--llm ..] [--vlm ..] [--vla ..]
  agente servir    -x <experimento> --fuente terminal|ros2
  agente comparar  -x <experimento> -e <ensayo> -r N --llm a,b,c [--vlm ..] [--vla ..]
  agente modelos                       lista el catalogo de config/modelos.yaml
  agente verificar [-x <experimento>]  revisa Ollama, rutas, librerias y topicos
  agente evaluar   [resultados/...]    calcula indicadores de las corridas
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import sys

from .util.config import CONFIG, RAIZ, aplicar, cargar_rutas, cargar_yaml


def _config(a) -> dict:
    cargar_rutas()
    cfg = cargar_yaml(a.experimento, "experimentos")
    cfg.setdefault("modelos", {})
    cfg.setdefault("entorno", {})
    for capa in ("llm", "vlm", "vla"):
        v = getattr(a, capa, None)
        if v and "," not in v:
            cfg["modelos"][capa] = v
    if getattr(a, "entorno", None):
        cfg["entorno"]["tipo"] = a.entorno
    if getattr(a, "robot", None):
        cfg["entorno"]["robot"] = a.robot
    if getattr(a, "modo", None):
        cfg["modo"] = a.modo
    if getattr(a, "nombre", None):
        cfg["nombre"] = a.nombre
    return aplicar(cfg, a.set or [])


def _registro(cfg, sufijo=""):
    from .registro import Registro
    r = cfg.get("registro", {})
    nombre = cfg.get("nombre", "corrida") + (f"__{sufijo}" if sufijo else "")
    return Registro(nombre, r.get("carpeta", ""), cfg, bool(r.get("guardar_imagenes", False)))


def _imprimir(e):
    extra = f" ({e['progreso']:.0%})" if isinstance(e.get("progreso"), float) and e["progreso"] >= 0 else ""
    print(f"  [{e.get('tarea') or '-'}] {e.get('estado')}: {str(e.get('detalle', ''))[:150]}{extra}", flush=True)


def _resumen(resultados, reg):
    if not resultados:
        return
    ok = sum(r.exito for r in resultados)
    t = sum(r.duracion_s for r in resultados) / len(resultados)
    print(f"\n>>> {ok}/{len(resultados)} tareas exitosas; tiempo medio {t:.1f} s; registro en {reg.dir}\n")


# ---------------------------------------------------------------------- comandos
def cmd_ejecutar(a):
    from .agente import Agente
    from .fuentes import crear_fuente
    cfg = _config(a)
    reg = _registro(cfg)
    agente = Agente(cfg, reg, _imprimir)
    print(f"Agente listo: modo={agente.modo} modelos={agente.modelos()} entorno={cfg['entorno']}")
    resultados = []
    try:
        for t in crear_fuente("archivo", ensayo=a.ensayo, repeticiones=a.repeticiones):
            resultados.append(agente.resolver(t))
    finally:
        _resumen(resultados, reg)
        reg.cerrar()
        agente.cerrar()


def cmd_servir(a):
    from .agente import Agente
    from .fuentes import crear_fuente
    cfg = _config(a)
    reg = _registro(cfg)
    fuente = crear_fuente(a.fuente, perfil=a.perfil)

    def reportar(e):
        _imprimir(e)
        fuente.reportar(e)
    agente = Agente(cfg, reg, reportar)
    print(f"Agente listo: modo={agente.modo} modelos={agente.modelos()} entorno={cfg['entorno']}")
    resultados = []
    try:
        for t in fuente:
            if isinstance(t, dict) and "cambiar" in t:
                for capa, nombre in t["cambiar"].items():
                    try:
                        agente.cambiar_modelo(capa, nombre)
                        reportar({"estado": "modelo_cambiado", "detalle": f"{capa} -> {nombre}"})
                    except Exception as exc:
                        reportar({"estado": "error", "detalle": f"no se pudo cambiar {capa}: {exc}"})
                continue
            resultados.append(agente.resolver(t))
    except KeyboardInterrupt:
        pass
    finally:
        _resumen(resultados, reg)
        reg.cerrar()
        agente.cerrar()


def cmd_comparar(a):
    """Misma tarea, N repeticiones, para cada combinacion de modelos. El entorno y
    los modelos se cargan una vez; solo se intercambia lo que cambia."""
    from .agente import Agente
    from .fuentes import crear_fuente
    cfg = _config(a)
    def lista(capa):
        v = getattr(a, capa) or cfg["modelos"].get(capa) or ""
        return [v] if isinstance(v, dict) else [x.strip() for x in str(v).split(",")]
    listas = {capa: lista(capa) for capa in ("llm", "vlm", "vla")}
    from .modelos.catalogo import etiqueta_de
    if cfg.get("modo", "modular") != "modular":
        listas["llm"] = [""]
    combos = list(itertools.product(listas["llm"], listas["vlm"], listas["vla"]))
    base = copy.deepcopy(cfg)
    for capa in ("llm", "vlm", "vla"):
        if listas[capa][0]:
            base["modelos"][capa] = listas[capa][0]
    agente = Agente(base, None, _imprimir)
    carpetas = []
    try:
        for llm, vlm, vla in combos:
            for capa, nombre in (("llm", llm), ("vlm", vlm), ("vla", vla)):
                if nombre and agente.cfg["modelos"].get(capa) != nombre:
                    agente.cambiar_modelo(capa, nombre)
            etiqueta = "+".join(etiqueta_de(x) for x in (llm, vlm, vla) if x).replace(":", "_").replace("/", "_")
            for m in (agente.percepcion.vlm, agente.ejecucion.vla):   # mismo ruido para cada combinacion
                if hasattr(m, "reiniciar_semilla"):
                    m.reiniciar_semilla()
            c = copy.deepcopy(agente.cfg)
            reg = _registro(c, etiqueta)
            agente.usar_registro(reg)
            print(f"\n=== {etiqueta} ({a.repeticiones or 'ensayo'} rep.)")
            res = [agente.resolver(t) for t in crear_fuente("archivo", ensayo=a.ensayo, repeticiones=a.repeticiones)]
            _resumen(res, reg)
            agente.usar_registro(None)
            reg.cerrar()
            carpetas.append(reg.dir)
    finally:
        agente.cerrar()
    sys.path.insert(0, RAIZ)
    from evaluacion.evaluar import main as evaluar
    evaluar(carpetas + ["-o", os.path.join(os.path.dirname(carpetas[0]), f"{cfg.get('nombre', 'comparacion')}_resumen.csv")])


def cmd_benchmark(a):
    """Delegado: agente benchmark <subcomando> ... == python3 -m benchmark <subcomando> ..."""
    sys.path.insert(0, RAIZ)
    from benchmark.__main__ import main as bench
    bench(a.resto)


def cmd_modelos(a):
    from .modelos.catalogo import leer_catalogo
    for capa, modelos in leer_catalogo().items():
        print(f"\n{capa.upper()}")
        for nombre, d in (modelos or {}).items():
            det = d.get("model") or d.get("checkpoint") or ""
            print(f"  {nombre:<18} {d.get('backend', ''):<13} {det}")
    print("\nTambien: <backend>:<modelo>, p. ej. --llm ollama:llama3.2:3b")


def cmd_verificar(a):
    import importlib
    import requests
    rutas = cargar_rutas()
    print("Rutas (config/rutas.env):")
    for k, v in rutas.items():
        existe = "" if not v.startswith("/") else ("  [ok]" if os.path.exists(v) else "  [NO EXISTE]")
        print(f"  {k} = {v}{existe}")
    url = os.environ.get("OLLAMA_URL", "http://localhost:11434")
    try:
        tags = [m["name"] for m in requests.get(url + "/api/tags", timeout=3).json().get("models", [])]
        print(f"\nOllama en {url}: {len(tags)} modelos -> {', '.join(tags)}")
    except Exception as exc:
        tags = None
        print(f"\nOllama en {url}: NO RESPONDE ({exc}). Inicia 'ollama serve'.")
    print("\nLibrerias en este Python (" + sys.executable + "):")
    for mod in ("numpy", "yaml", "requests", "PIL", "rclpy", "control_msgs", "torch", "transformers", "lerobot"):
        try:
            m = importlib.import_module(mod)
            print(f"  {mod:<13} ok {getattr(m, '__version__', '')}")
        except Exception as exc:
            print(f"  {mod:<13} --  ({type(exc).__name__})")
    if a.experimento:
        cfg = _config(a)
        print(f"\nExperimento '{a.experimento}': modelos {cfg['modelos']}")
        from .modelos.catalogo import resolver
        for capa, nombre in cfg["modelos"].items():
            try:
                clase, kw = resolver(capa, nombre)
                ref = kw.get("model") or kw.get("checkpoint") or ""
                nota = ""
                if clase.endswith(("OllamaLLM", "OllamaVLM")) and tags is not None:
                    nota = "  [ok]" if ref in tags or f"{ref}:latest" in tags else "  [falta: ollama pull " + ref + "]"
                elif ref.startswith("/"):
                    nota = "  [ok]" if os.path.exists(ref) else "  [NO EXISTE]"
                print(f"  {capa}: {nombre} -> {clase.split(':')[1]} {ref}{nota}")
            except Exception as exc:
                print(f"  {capa}: {nombre} -> ERROR {exc}")
        if cfg["entorno"].get("tipo") == "ros2":
            from .entornos.robot import cargar_robot
            r = cargar_robot(cfg["entorno"].get("robot", "omx"))
            print(f"\nTopicos que espera el robot '{r.nombre}': camaras {list(r.camaras.values())}, "
                  f"{r.estado_articular}, " + ", ".join(b.trayectoria for b in r.brazos))
            print("Compruebalos con:  ros2 topic list   y   ros2 topic hz <topico>")


def cmd_evaluar(a):
    sys.path.insert(0, RAIZ)
    from evaluacion.evaluar import main as evaluar
    evaluar(a.resto)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "benchmark":          # todo lo demas va tal cual al benchmark
        return cmd_benchmark(argparse.Namespace(resto=argv[1:]))
    ap = argparse.ArgumentParser(prog="agente", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def comunes(p, ensayo=True):
        p.add_argument("-x", "--experimento", default="sim_completo", help="config/experimentos/<nombre>.yaml")
        if ensayo:
            p.add_argument("-e", "--ensayo", default="medicamentos_basico", help="config/ensayos/<nombre>.yaml")
            p.add_argument("-r", "--repeticiones", type=int, default=0)
        p.add_argument("--llm"), p.add_argument("--vlm"), p.add_argument("--vla")
        p.add_argument("--entorno", help="simulado | ros2")
        p.add_argument("--robot", help="omx | so101_bimanual | ruta.yaml")
        p.add_argument("--modo", help="modular | extremo_a_extremo")
        p.add_argument("--nombre", help="nombre de la corrida en resultados/")
        p.add_argument("--set", action="append", help="clave.subclave=valor (repetible)")

    comunes(sub.add_parser("ejecutar", help="corre un ensayo"))
    p = sub.add_parser("servir", help="espera tareas (terminal o ROS 2)")
    comunes(p, ensayo=False)
    p.add_argument("--fuente", default="terminal", help="terminal | ros2")
    p.add_argument("--perfil", default="manipulacion_general", help="perfil para instrucciones en texto libre")
    comunes(sub.add_parser("comparar", help="compara modelos con el mismo ensayo"))
    sub.add_parser("modelos", help="lista el catalogo")
    p = sub.add_parser("verificar", help="revisa la instalacion")
    comunes(p, ensayo=False)
    p.set_defaults(experimento=None)
    p = sub.add_parser("benchmark", help="benchmark por capa: llm | vlm | vla | reporte (ver BENCHMARK.md)",
                       add_help=False)
    p.add_argument("resto", nargs=argparse.REMAINDER)
    p = sub.add_parser("evaluar", help="indicadores de las corridas")
    p.add_argument("resto", nargs=argparse.REMAINDER)
    a = ap.parse_args(argv)
    {"ejecutar": cmd_ejecutar, "servir": cmd_servir, "comparar": cmd_comparar, "modelos": cmd_modelos,
     "verificar": cmd_verificar, "evaluar": cmd_evaluar,
     "benchmark": cmd_benchmark}[a.cmd](a)


if __name__ == "__main__":
    main()
