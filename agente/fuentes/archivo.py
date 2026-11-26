"""Tareas desde un ensayo (config/ensayos/<ensayo>.yaml)."""
from __future__ import annotations

from ..interfaces import FuenteTareas
from ..util.config import cargar_yaml
from .comun import a_tarea


class FuenteArchivo(FuenteTareas):
    def __init__(self, ensayo: str, repeticiones: int = 0, **_):
        d = cargar_yaml(ensayo, "ensayos")
        self.tareas = d.get("tareas") or [d]
        self.repeticiones = int(repeticiones or d.get("repeticiones", 1))

    def __iter__(self):
        for r in range(self.repeticiones):
            for i, t in enumerate(self.tareas):
                tarea = a_tarea(t)
                base = t.get("id") or f"T{i + 1}"
                tarea.id = f"{base}-r{r + 1:02d}"
                yield tarea
