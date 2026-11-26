"""Construccion del CONJUNTO ETIQUETADO para el benchmark VLM.

Un conjunto es una carpeta  datos/vlm/<nombre>/  con  imagenes/  y  etiquetas.jsonl.
Cada linea de etiquetas.jsonl es un caso con su respuesta correcta:
  verificacion  {"id", "imagen", "tipo": "verificacion", "objeto", "destino", "esperado": true|false,
                 "instruccion", "grupo", "fuente"}
  inventario    {"id", "imagen", "tipo": "inventario", "conteos": {"Hidroferol": 2, "Aspirina": 0, ...},
                 "ubicaciones": {"Hidroferol": "left"} (opcional), "grupo", "fuente"}

Subcomandos (python3 -m benchmark vlm-datos <subcomando> ...):
  desde-lerobot  episodios grabados -> fotograma inicial (objeto AUN NO en el destino = false)
                 y final (objeto YA en el destino = true); opcional: negativos cruzados (fotograma
                 final preguntando por OTRO objeto = false). Sin etiquetar a mano.
  desde-carpeta  tus fotos + un CSV de etiquetas (imagen,tipo,objeto,destino,esperado,conteos)
  capturar       toma fotos de la camara ROS 2 (MuJoCo o robot) y pide la etiqueta de cada una
  revisar        mosaico PNG con cada imagen y su etiqueta, para comprobar el conjunto a ojo
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import random
import re
import sys

import numpy as np

from agente.util.config import RAIZ
from agente.util.imagen import guardar_png, leer_rgb

BASE = os.path.join(RAIZ, "datos", "vlm")
PATRONES_OBJETO = [r"pick up the (.+?) and (?:place|put)", r"pick (?:up )?the (.+?)(?: from| and|$)",
                   r"grab the (.+?)(?: and|$)", r"the (.+?) (?:in|into|inside) the"]
PATRON_DESTINO = r"(?:in|into|inside|on) the (.+?)\.?$"


def _carpeta(nombre):
    d = nombre if os.path.isabs(nombre) else os.path.join(BASE, nombre)
    os.makedirs(os.path.join(d, "imagenes"), exist_ok=True)
    return d


def _escribir(d, casos, modo="w"):
    with open(os.path.join(d, "etiquetas.jsonl"), modo, encoding="utf-8") as f:
        for c in casos:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"{len(casos)} casos -> {os.path.join(d, 'etiquetas.jsonl')}")


def leer(nombre: str):
    d = nombre if os.path.isabs(nombre) else os.path.join(BASE, nombre)
    p = os.path.join(d, "etiquetas.jsonl")
    if not os.path.exists(p):
        raise FileNotFoundError(f"No existe {p}. Construye el conjunto con: python3 -m benchmark vlm-datos ...")
    with open(p, encoding="utf-8") as f:
        casos = [json.loads(x) for x in f if x.strip()]
    for c in casos:
        c["_ruta"] = os.path.join(d, c["imagen"])
    return d, casos


def objeto_de(tarea: str) -> str:
    """Objeto de la tarea del conjunto ("Pick up the Hidroferol and place it ..." -> Hidroferol)."""
    t = tarea.strip()
    for p in PATRONES_OBJETO:
        m = re.search(p, t, re.IGNORECASE)
        if m:
            return t[m.start(1):m.end(1)].strip()
    return t


def destino_de(tarea: str, defecto: str) -> str:
    t = tarea.strip()
    m = re.search(PATRON_DESTINO, t, re.IGNORECASE)
    return t[m.start(1):m.end(1)].strip() if m else defecto


# ----------------------------------------------------------------------------- desde episodios
def desde_episodios(a):
    from ..comun.episodios import abrir, seleccionar
    fuente = abrir(a.fuente, **({"camara": a.camara} if a.fuente.startswith("lerobot") else {}))
    eps = seleccionar(fuente.episodios(), a.episodios)
    d = _carpeta(a.salida)
    objetos_extra = [o.strip() for o in (a.objetos or "").split(",") if o.strip()]
    rng = random.Random(0)
    casos = []
    for ep in eps:
        ini, fin = fuente.fotograma(ep, a.fotograma_inicial), fuente.fotograma(ep, a.fotograma_final)
        tarea = ini["tarea"] or a.tarea
        obj = a.objeto or objeto_de(tarea)
        dest = a.destino or destino_de(tarea, "box")
        for nombre, fg, esperado in (("inicio", ini, False), ("final", fin, True)):
            img = f"imagenes/ep{ep:04d}_{nombre}.png"
            guardar_png(fg["imagen"], os.path.join(d, img))
            casos.append({"id": f"ep{ep:04d}_{nombre}", "imagen": img, "tipo": "verificacion", "objeto": obj,
                          "destino": dest, "esperado": esperado, "instruccion": tarea, "grupo": f"ep{ep:04d}",
                          "fuente": f"{a.fuente} episodio {ep} fotograma {fg['indice']}"})
        otros = [o for o in objetos_extra if o.lower() != obj.lower()]
        for k in range(min(a.negativos_cruzados, len(otros))):
            o = rng.choice(otros)
            otros.remove(o)
            casos.append({"id": f"ep{ep:04d}_cruzado_{k}", "imagen": f"imagenes/ep{ep:04d}_final.png",
                          "tipo": "verificacion", "objeto": o, "destino": dest, "esperado": False,
                          "instruccion": tarea, "grupo": f"ep{ep:04d}",
                          "fuente": f"{a.fuente} episodio {ep} final, pregunta por un objeto que no se movio"})
    _escribir(d, casos)
    print("Revisa las etiquetas con:  python3 -m benchmark vlm-datos revisar", a.salida)


# ----------------------------------------------------------------------------- desde carpeta + CSV
def desde_carpeta(a):
    d = _carpeta(a.salida)
    casos = []
    with open(a.etiquetas, newline="", encoding="utf-8") as f:
        for i, fila in enumerate(csv.DictReader(f)):
            src = fila["imagen"] if os.path.isabs(fila["imagen"]) else os.path.join(a.carpeta, fila["imagen"])
            img = f"imagenes/{i:05d}_{os.path.basename(src).rsplit('.', 1)[0]}.png"
            guardar_png(leer_rgb(src), os.path.join(d, img))
            tipo = fila.get("tipo", "verificacion").strip()
            c = {"id": f"img{i:05d}", "imagen": img, "tipo": tipo, "grupo": fila.get("grupo") or f"img{i:05d}",
                 "fuente": src}
            if tipo == "verificacion":
                c.update(objeto=fila["objeto"], destino=fila.get("destino") or a.destino,
                         esperado=str(fila["esperado"]).strip().lower() in ("1", "true", "si", "sí", "yes"),
                         instruccion=fila.get("instruccion", ""))
            else:
                c.update(conteos={k: int(v) for k, v in json.loads(fila["conteos"]).items()})
                if fila.get("ubicaciones"):
                    c["ubicaciones"] = json.loads(fila["ubicaciones"])
            casos.append(c)
    _escribir(d, casos)


# ----------------------------------------------------------------------------- captura ROS 2
def capturar(a):
    """Fotos de la camara del robot (MuJoCo o fisico) con etiqueta tecleada."""
    from agente.entornos.ros2 import EntornoROS2
    ent = EntornoROS2(a.robot, espera_datos_s=20)
    ent.conectar()
    d = _carpeta(a.salida)
    nombres = [n.strip() for n in a.objetos.split(",") if n.strip()]
    previos = len(open(os.path.join(d, "etiquetas.jsonl")).readlines()) if os.path.exists(os.path.join(d, "etiquetas.jsonl")) else 0
    print(f"Captura de casos '{a.tipo}'. Acomoda la escena y presiona Enter (q para terminar).")
    casos = []
    try:
        while True:
            if input("\nEnter = capturar, q = salir > ").strip().lower() == "q":
                break
            obs = ent.observar()
            img = obs.imagen(a.camara)
            k = previos + len(casos)
            ruta = f"imagenes/cap{k:05d}.png"
            guardar_png(img, os.path.join(d, ruta))
            c = {"id": f"cap{k:05d}", "imagen": ruta, "tipo": a.tipo, "grupo": f"cap{k:05d}", "fuente": "captura ROS 2"}
            if a.tipo == "verificacion":
                c["objeto"] = input(f"  objeto [{nombres[0] if nombres else ''}]: ").strip() or (nombres[0] if nombres else "")
                c["destino"] = input(f"  destino [{a.destino}]: ").strip() or a.destino
                c["esperado"] = input("  ¿esta el objeto dentro del destino? (s/n): ").strip().lower().startswith("s")
                c["instruccion"] = f"Pick up the {c['objeto']} and place it inside the {c['destino']}"
            else:
                c["conteos"] = {}
                for n in nombres:
                    v = input(f"  cajas de {n} visibles (vacio = 0): ").strip()
                    c["conteos"][n] = int(v) if v else 0
            casos.append(c)
            print(f"  guardado {ruta}")
    finally:
        _escribir(d, casos, modo="a")
        ent.cerrar()


# ----------------------------------------------------------------------------- revision visual
def revisar(a):
    from PIL import Image, ImageDraw
    from .perturbaciones import DESCRIPCION, aplicar
    d, casos = leer(a.conjunto)
    tam, cols = 220, 6
    filas = (len(casos) + cols - 1) // cols
    lienzo = Image.new("RGB", (cols * tam, filas * (tam + 34)), "white")
    dib = ImageDraw.Draw(lienzo)
    for i, c in enumerate(casos):
        im = Image.fromarray(leer_rgb(c["_ruta"])).resize((tam, tam))
        x, y = (i % cols) * tam, (i // cols) * (tam + 34)
        lienzo.paste(im, (x, y))
        if c["tipo"] == "verificacion":
            txt = f"{c['id']}\n{c['objeto']} en {c['destino']}: {'SI' if c['esperado'] else 'NO'}"
        else:
            txt = f"{c['id']}\n" + ", ".join(f"{k}:{v}" for k, v in c["conteos"].items())
        dib.text((x + 3, y + tam + 2), txt[:80], fill="black")
    salida = os.path.join(d, "revision.png")
    lienzo.save(salida)
    muestra = leer_rgb(casos[0]["_ruta"])
    tira = Image.new("RGB", (len(DESCRIPCION) * tam, tam + 18), "white")
    for j, nombre in enumerate(DESCRIPCION):
        tira.paste(Image.fromarray(aplicar(muestra, nombre)).resize((tam, tam)), (j * tam, 0))
        ImageDraw.Draw(tira).text((j * tam + 3, tam + 2), nombre, fill="black")
    tira.save(os.path.join(d, "perturbaciones.png"))
    ver = sum(1 for c in casos if c["tipo"] == "verificacion")
    pos = sum(1 for c in casos if c["tipo"] == "verificacion" and c["esperado"])
    print(f"{len(casos)} casos: {ver} de verificacion ({pos} positivos, {ver - pos} negativos), "
          f"{len(casos) - ver} de inventario")
    print(f"Mosaico: {salida}\nPerturbaciones: {os.path.join(d, 'perturbaciones.png')}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="benchmark vlm-datos", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("desde-lerobot", help="inicio/fin de episodios grabados")
    p.add_argument("--fuente", required=True, help="lerobot:<repo>[@carpeta] o npz:<carpeta>")
    p.add_argument("--salida", required=True, help="nombre del conjunto (datos/vlm/<nombre>)")
    p.add_argument("--episodios", default="todos", help='"todos", "0-49" o "0,3,7"')
    p.add_argument("--camara", default="observation.images.front")
    p.add_argument("--objeto", default="", help="nombre del objeto si la tarea no lo dice (p. ej. Hidroferol)")
    p.add_argument("--destino", default="", help="nombre del destino (p. ej. 'brown box')")
    p.add_argument("--tarea", default="", help="tarea si el conjunto no la trae")
    p.add_argument("--objetos", default="Aspirina,Blaxitec,Diotul,Hidroferol,Laritol,Tylenol",
                   help="objetos para los negativos cruzados")
    p.add_argument("--negativos-cruzados", type=int, default=1)
    p.add_argument("--fotograma-inicial", type=int, default=0)
    p.add_argument("--fotograma-final", type=int, default=-1)
    p = sub.add_parser("desde-carpeta", help="fotos propias + CSV de etiquetas")
    p.add_argument("--carpeta", required=True)
    p.add_argument("--etiquetas", required=True)
    p.add_argument("--salida", required=True)
    p.add_argument("--destino", default="brown box")
    p = sub.add_parser("capturar", help="fotos de la camara ROS 2 con etiqueta")
    p.add_argument("--robot", default="omx")
    p.add_argument("--camara", default="front")
    p.add_argument("--salida", required=True)
    p.add_argument("--tipo", choices=["verificacion", "inventario"], default="inventario")
    p.add_argument("--objetos", default="Aspirina,Blaxitec,Diotul,Hidroferol,Laritol,Tylenol")
    p.add_argument("--destino", default="brown box")
    p = sub.add_parser("revisar", help="mosaico de imagenes con su etiqueta")
    p.add_argument("conjunto")
    a = ap.parse_args(argv)
    {"desde-lerobot": desde_episodios, "desde-carpeta": desde_carpeta, "capturar": capturar,
     "revisar": revisar}[a.cmd](a)


if __name__ == "__main__":
    main(sys.argv[1:])
