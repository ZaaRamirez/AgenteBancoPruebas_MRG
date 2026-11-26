"""Tareas escritas en la terminal.

  Put the banana in the bowl                      -> instruccion libre (perfil por omision)
  {"perfil": "empaquetado_medicamentos", "parametros": {"items": {"Hidroferol": 1}, "destination": "brown box"}}
  @medicamentos_basico                            -> corre un ensayo de config/ensayos/
  :llm deepseek-r1   :vlm gemma4   :vla molmoact2_omx   -> cambia el modelo de una capa
  (linea vacia o 'salir' para terminar)
"""
from __future__ import annotations

from ..interfaces import FuenteTareas
from .archivo import FuenteArchivo
from .comun import a_tarea


class FuenteTerminal(FuenteTareas):
    def __init__(self, perfil: str = "manipulacion_general", **_):
        self.perfil = perfil

    def __iter__(self):
        print(__doc__)
        while True:
            try:
                linea = input("tarea> ").strip()
            except EOFError:
                return
            if not linea or linea.lower() in ("salir", "exit", "q"):
                return
            if linea.startswith(":"):
                partes = linea[1:].split(maxsplit=1)
                if len(partes) == 2:
                    yield {"cambiar": {partes[0]: partes[1]}}
                continue
            if linea.startswith("@"):
                yield from FuenteArchivo(linea[1:].strip())
                continue
            try:
                yield a_tarea(linea, self.perfil)
            except Exception as exc:
                print(f"No entendi la tarea: {exc}")

