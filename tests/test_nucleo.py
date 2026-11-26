"""Pruebas del nucleo (sin ROS, sin GPU):  python3 -m pytest -q tests   o   python3 tests/test_nucleo.py"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agente.agente import Agente  # noqa: E402
from agente.capas.razonamiento import ContextoTarea, interpretar  # noqa: E402
from agente.entornos.robot import cargar_robot  # noqa: E402
from agente.fuentes.archivo import FuenteArchivo  # noqa: E402
from agente.modelos import catalogo  # noqa: E402
from agente.tareas.perfiles import load_profile  # noqa: E402
from agente.util.config import aplicar, cargar_yaml  # noqa: E402
from agente.util.texto import extraer_json, rellenar, validar_esquema  # noqa: E402


def test_json_y_esquema():
    assert extraer_json('<think>{"x":1}</think> ok ```json\n{"action": "done"}\n```') == {"action": "done"}
    assert validar_esquema({"a": 1}, {"type": "object", "required": ["b"]}) == ["falta 'b'"]
    assert rellenar("{items_text} -> {destination} {{lit}}", {"items": {"A": 2}, "destination": "box"}) == "2 A -> box {lit}"


def test_validacion_del_razonamiento():
    p = load_profile("empaquetado_medicamentos")
    c = ContextoTarea("T", "x", {"items": {"Hidroferol": 1}, "destination": "box"}, p)
    d = interpretar('{"action": "pick_and_place", "arguments": {"object": "hidroferol"}}', c)
    assert d.valida and d.argumentos == {"object": "Hidroferol", "destination": "box"}
    assert interpretar('{"action": "done"}', c).motivo == "fin_prematuro"
    assert interpretar('{"action": "fly"}', c).motivo.startswith("accion_desconocida")
    c.seguimiento.register("pick_and_place", {"object": "Hidroferol"})
    assert interpretar('{"action": "pick_and_place", "arguments": {"object": "Hidroferol"}}', c).motivo.startswith("already_complete")


def test_robots():
    omx = cargar_robot("omx")
    e = omx.especificacion()
    assert e.dim == 6 and e.indices_articulaciones == [0, 1, 2, 3, 4]
    s = omx.vector_estado({"Shoulder_Pitch": np.pi / 2, "Gripper": 1.0})
    assert abs(s[1] - 90) < 1e-3 and abs(s[5] - 40) < 1e-3
    assert cargar_robot("so101_bimanual").especificacion().dim == 12


def test_catalogo():
    assert catalogo.resolver("llm", "qwen3.8")[1]["model"] == "qwen3.8:latest"
    clase, kw = catalogo.resolver("llm", "ollama:llama3.2:3b")
    assert clase == "agente.modelos.llm:OllamaLLM" and kw["model"] == "llama3.2:3b"


def _cfg(**kw):
    cfg = aplicar(cargar_yaml("sim_completo", "experimentos"), [f"{k}={v}" for k, v in kw.items()])
    cfg["registro"] = {}
    return cfg


def test_agente_modular_y_extremo():
    a = Agente(_cfg())
    for t in FuenteArchivo("medicamentos_basico"):
        r = a.resolver(t)
        assert r.exito and r.subtareas == 3 and r.progreso == 1.0
    a2 = Agente(_cfg(**{"modo": "extremo_a_extremo"}))
    assert a2.resolver(next(iter(FuenteArchivo("omx_hidroferol")))).exito


def test_no_realizable_y_cambio_de_modelo():
    a = Agente(_cfg(**{"modelos.vlm": "simulado_faltante"}))
    r = a.resolver(next(iter(FuenteArchivo("medicamentos_basico"))))
    assert r.estado == "no_realizable" and not r.exito
    a.cambiar_modelo("vlm", "simulado")
    assert a.resolver(next(iter(FuenteArchivo("medicamentos_basico")))).exito


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
