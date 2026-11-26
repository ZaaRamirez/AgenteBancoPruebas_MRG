"""Tablas del reporte: formato de celdas, Markdown y CSV."""
from __future__ import annotations

import csv
import math
from typing import Any, Dict, List, Optional


def pct(d: Optional[Dict], ic: bool = True, dec: int = 1) -> str:
    if not d or d.get("n", 0) == 0 or d["valor"] != d["valor"]:
        return "–"
    s = f"{100 * d['valor']:.{dec}f} %"
    return s + (f" [{100 * d['ic_bajo']:.{dec}f}–{100 * d['ic_alto']:.{dec}f}]" if ic else "")


def num(d, dec: int = 2, ic: bool = True, unidad: str = "") -> str:
    if d is None:
        return "–"
    if isinstance(d, dict):
        if d.get("n", 0) == 0 or d["valor"] != d["valor"]:
            return "–"
        s = f"{d['valor']:.{dec}f}{unidad}"
        return s + (f" [{d['ic_bajo']:.{dec}f}–{d['ic_alto']:.{dec}f}]" if ic and d.get("n", 0) > 1 else "")
    if isinstance(d, float) and (math.isnan(d) or math.isinf(d)):
        return "–"
    if isinstance(d, (int, float)):
        return f"{d:.{dec}f}{unidad}" if isinstance(d, float) else f"{d}{unidad}"
    return str(d)


def valor_p(p) -> str:
    if p is None:
        return "–"
    return "<0.001" if p < 0.001 else f"{p:.3f}"


class Tabla:
    def __init__(self, titulo: str, columnas: List[str], nota: str = "", clave: str = ""):
        self.titulo, self.columnas, self.nota, self.clave = titulo, columnas, nota, clave
        self.filas: List[List[Any]] = []

    def agregar(self, *celdas):
        self.filas.append([("–" if c is None else c) for c in celdas])

    def markdown(self) -> str:
        if not self.filas:
            return f"**{self.titulo}**\n\n_(sin datos)_\n"
        out = [f"**{self.titulo}**", "", "| " + " | ".join(self.columnas) + " |",
               "|" + "|".join("---" for _ in self.columnas) + "|"]
        out += ["| " + " | ".join(str(c).replace("|", "/") for c in f) + " |" for f in self.filas]
        if self.nota:
            out += ["", f"_{self.nota}_"]
        return "\n".join(out) + "\n"

    def texto(self) -> str:
        if not self.filas:
            return f"{self.titulo}\n  (sin datos)\n"
        filas = [[str(c) for c in f] for f in self.filas]
        anchos = [max(len(h), *(len(f[i]) for f in filas)) for i, h in enumerate(self.columnas)]
        out = [self.titulo, "  ".join(h.ljust(w) for h, w in zip(self.columnas, anchos)),
               "  ".join("-" * w for w in anchos)]
        out += ["  ".join(c.ljust(w) for c, w in zip(f, anchos)) for f in filas]
        return "\n".join(out) + "\n"

    def csv(self, ruta: str):
        with open(ruta, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(self.columnas)
            w.writerows(self.filas)
