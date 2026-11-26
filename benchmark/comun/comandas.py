"""Convierte comanda.xlsx (Order_ID, Doctor_Name, Medicine_Name, Quantity) a
config/benchmark/comandas.yaml. Acepta tambien .csv con las mismas columnas."""
from __future__ import annotations

import csv
from collections import OrderedDict

import yaml

ENCABEZADO = """# =============================================================================
#  PEDIDOS REALES (convertidos de comanda.xlsx). Si cambias comanda.xlsx, vuelve a
#  generar este archivo con:  python3 -m benchmark comandas comanda.xlsx
#  Los usan el benchmark LLM (casos y episodios).
# =============================================================================
"""


def _filas(ruta):
    if ruta.lower().endswith(".csv"):
        with open(ruta, newline="", encoding="utf-8") as f:
            r = csv.reader(f)
            next(r, None)
            yield from r
        return
    try:
        import openpyxl
    except ImportError:
        raise SystemExit("Falta openpyxl (pip install openpyxl) o guarda comanda.xlsx como CSV.")
    ws = openpyxl.load_workbook(ruta, read_only=True).active
    for i, fila in enumerate(ws.iter_rows(values_only=True)):
        if i:
            yield fila


def convertir(ruta: str, salida: str, destino: str = "brown box"):
    pedidos = OrderedDict()
    for f in _filas(ruta):
        if not f or not f[0] or not f[2]:
            continue
        p = pedidos.setdefault(str(f[0]), {"id": str(f[0]), "solicitante": str(f[1] or ""), "items": {}})
        p["items"][str(f[2]).strip()] = p["items"].get(str(f[2]).strip(), 0) + int(f[3])
    with open(salida, "w", encoding="utf-8") as fh:
        fh.write(ENCABEZADO + yaml.safe_dump({"destino": destino, "pedidos": list(pedidos.values())},
                                             allow_unicode=True, sort_keys=False))
    print(f"{len(pedidos)} pedidos -> {salida}")
