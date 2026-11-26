"""Pruebas del benchmark (sin modelos reales):  python3 -m pytest -q tests  o  python3 tests/test_benchmark.py"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agente.agente import Agente  # noqa: E402
from agente.tipos import Tarea  # noqa: E402
from benchmark.comun import estadistica as est  # noqa: E402
from benchmark.comun.config import vocabulario  # noqa: E402
from benchmark.llm.calificar import calificar  # noqa: E402
from benchmark.llm.casos import CATEGORIAS, generar  # noqa: E402
from benchmark.vlm.correr import calificar as calificar_vlm  # noqa: E402


def test_estadistica():
    p, lo, hi = est.wilson(90, 100)
    assert abs(p - 0.9) < 1e-9 and 0.82 < lo < 0.83 and 0.94 < hi < 0.95
    assert est.prueba_signos([1] * 6 + [0], [0] * 7)["p"] < 0.05
    r = est.ranking({"a": {i: 1.0 for i in range(50)}, "b": {i: float(i % 2) for i in range(50)}})
    assert r[0]["modelo"] == "a" and r[1]["grupo"] == "B"
    assert est.tamano_muestra(0.9, 0.03) == 385


def test_casos_deterministas_y_referencia_perfecta():
    v, c = vocabulario("objetos"), vocabulario("comandas")
    a, b = generar(v, c, 3, 7), generar(v, c, 3, 7)
    assert [x["prompt"] for x in a] == [x["prompt"] for x in b]          # misma semilla, mismos casos
    assert {x["categoria"] for x in a} == set(CATEGORIAS)
    assert all(calificar(x, x["contexto_guion"]["respuesta"])["correcto"] for x in a)


def test_calificacion_de_errores():
    v, c = vocabulario("objetos"), vocabulario("comandas")
    casos = {x["categoria"]: x for x in generar(v, c, 1, 1)}
    k = casos["med_intermedio"]
    hecho = k["completados"][0] if k["completados"] else None
    if hecho:
        assert calificar(k, json.dumps({"action": "pick_and_place", "arguments": {"object": hecho}}))["error"] == "repite_lo_hecho"
    assert calificar(k, "")["error"] == "respuesta_vacia"
    assert calificar(k, "I will pick the next box")["error"] == "formato"
    assert calificar(k, '{"action": "done"}')["error"] == "fin_prematuro"
    assert calificar(k, '{"action": "pick_and_place", "arguments": {"object": "Unicornio"}}')["error"] == "elemento_inventado"
    g = casos["gen_completo"]
    assert calificar(g, '{"action": "done"}')["correcto"]
    a = casos["gen_ausente"]
    assert calificar(a, '{"action": "abort", "arguments": {"reason": "x"}}')["correcto"]


def test_calificacion_vlm():
    caso = {"tipo": "verificacion", "esperado": False}
    assert calificar_vlm(caso, {"success": False})["correcto"]
    assert not calificar_vlm(caso, {"success": True})["correcto"]
    assert calificar_vlm(caso, None)["correcto"]           # sin respuesta = "no" (como el agente)
    inv = {"tipo": "inventario", "conteos": {"Hidroferol": 2, "Aspirina": 0}}
    r = calificar_vlm(inv, {"items": [{"name": "hidroferol", "count": 2}, {"name": "Paracetamol", "count": 1}]})
    assert r["correcto"] and r["no_pedidos"] == ["Paracetamol"]


def test_oraculo_de_escena_y_agente():
    cfg = {"modo": "modular", "modelos": {"llm": "guion", "vlm": {"backend": "escena", "tasa_exito": 0.7, "semilla": 3},
                                          "vla": "simulado"},
           "entorno": {"tipo": "simulado", "robot": "omx"}, "agente": {"max_reintentos": 5},
           "ejecucion": {"ciclos_estables": 3, "min_pasos": 2}}
    a = Agente(cfg)
    r = a.resolver(Tarea(id="t1", perfil="empaquetado_medicamentos",
                         parametros={"items": {"Hidroferol": 2, "Tylenol": 1}, "destination": "brown box"}))
    assert r.exito and r.subtareas_exitosas == 3 and r.unidades_requeridas == 3
    r = a.resolver(Tarea(id="t2", perfil="manipulacion_general", instruccion="Put the banana in the bowl",
                         referencia={"acciones": [{"action": "pick_and_place",
                                                   "arguments": {"object": "banana", "destination": "bowl"}}]}))
    assert r.exito


if __name__ == "__main__":
    for n, f in list(globals().items()):
        if n.startswith("test_"):
            f()
            print("ok", n)
