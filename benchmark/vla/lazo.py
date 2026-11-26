"""Benchmark VLA en LAZO CERRADO: el modelo controla el robot (gemelo MuJoCo o fisico)
a traves del mismo Entorno que usa el agente, y un JUEZ decide si la tarea se cumplio.

Diseno pareado: las escenas iniciales se recorren en el mismo orden para cada modelo
(config lazo.escenas), de modo que el ensayo k de todos los modelos parte del mismo arreglo.

Jueces:
  manual    el operador califica: [e]xito / [a]garro pero no coloco / [n] no agarro / [x] invalido
  pose      el simulador publica la pose del objeto (geometry_msgs/PoseStamped); exito si queda
            dentro de la region del destino  -> verdad de referencia automatica
  vlm       un VLM juzga con la pregunta del agente (util, pero NO es verdad de referencia)
  detector  solo si el robot se detuvo (para probar el banco; NO mide exito real)
Reinicio entre ensayos: manual (Enter), servicio ROS 2 (std_srvs Trigger/Empty) o ninguno.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from agente.capas.ejecucion import DetectorFin
from agente.entornos import crear_entorno
from agente.util.texto import rellenar
from ..comun import config as bcfg
from ..comun.almacen import Corrida

ETAPAS = {"e": "exito", "a": "agarro_sin_colocar", "n": "no_agarro", "x": "invalido"}


class Juez:
    def __init__(self, cfg: Dict[str, Any], entorno):
        self.modo = cfg.get("modo", "manual")
        self.cfg, self.entorno = cfg, entorno
        self._pose = {}
        if self.modo == "vlm":
            from agente.capas.percepcion import CapaPercepcion
            self.vlm = CapaPercepcion(bcfg.modelo("vlm", cfg.get("modelo_vlm", "gemma4"), {"temperature": 0.0}), None)

    def _suscribir_pose(self, objeto):
        if objeto in self._pose:
            return
        from geometry_msgs.msg import PoseStamped
        topico = self.cfg["topico_pose"].format(object=objeto)
        self._pose[objeto] = None
        self.entorno.nodo.create_subscription(PoseStamped, topico, lambda m, o=objeto: self._pose.__setitem__(o, m), 10)

    def calificar(self, tarea: Dict[str, Any], terminado: bool) -> Dict[str, Any]:
        if self.modo == "detector":
            return {"exito": bool(terminado), "etapa": "exito" if terminado else "tiempo_agotado", "juez": "detector"}
        if self.modo == "manual":
            while True:
                r = input("  Resultado [e]xito / [a]garro sin colocar / [n]o agarro / [x] ensayo invalido: ").strip().lower()[:1]
                if r in ETAPAS:
                    return {"exito": r == "e", "etapa": ETAPAS[r], "juez": "manual"}
        if self.modo == "pose":
            self._suscribir_pose(tarea["objeto"])
            fin = time.time() + 3
            while self._pose.get(tarea["objeto"]) is None and time.time() < fin:
                time.sleep(0.1)
            m = self._pose.get(tarea["objeto"])
            if m is None:
                return {"exito": False, "etapa": "invalido", "juez": "pose", "detalle": "sin pose del objeto"}
            p = m.pose.position
            reg = self.cfg["region_destino"]
            dentro = all(reg[k][0] <= getattr(p, k) <= reg[k][1] for k in ("x", "y", "z"))
            return {"exito": dentro, "etapa": "exito" if dentro else "no_colocado", "juez": "pose",
                    "posicion_final": [p.x, p.y, p.z]}
        if self.modo == "vlm":
            q = (f"The robot was asked to: \"{tarea['instruccion']}\". Look at the image: is the {tarea['objeto']} "
                 f"now inside the {tarea['destino']}?")
            r = self.vlm.preguntar("juez", q, self.entorno.observar(), {"type": "object", "properties": {
                "success": {"type": "boolean"}, "reason": {"type": "string"}}, "required": ["success"]})
            ok = isinstance(r.datos, dict) and r.datos.get("success") is True
            return {"exito": ok, "etapa": "exito" if ok else "no_colocado", "juez": "vlm", "detalle": r.texto[:200]}
        raise ValueError(f"Juez desconocido: {self.modo}")


def _reiniciar(cfg, entorno, escena: str):
    modo = cfg.get("modo", "manual")
    if modo == "manual":
        input(f"  Prepara la escena: [{escena}] y presiona Enter ")
    elif modo == "servicio":
        from agente.entornos import ros_comun
        if cfg.get("tipo", "Trigger") == "Empty":
            from std_srvs.srv import Empty as Srv
        else:
            from std_srvs.srv import Trigger as Srv
        nodo = getattr(entorno, "nodo", None) or ros_comun.crear_nodo("benchmark_reinicio")
        cli = nodo.create_client(Srv, cfg["servicio"])
        if not cli.wait_for_service(timeout_sec=10):
            raise RuntimeError(f"El servicio {cfg['servicio']} no responde")
        fut = cli.call_async(Srv.Request())
        fin = time.time() + 30
        while not fut.done() and time.time() < fin:
            time.sleep(0.05)
    time.sleep(float(cfg.get("espera_s", 0.0)))


def ejecutar(politica, entorno, instruccion, det_cfg, max_duracion_s):
    """Mismo bucle que la capa de ejecucion del agente, guardando la trayectoria."""
    esp = entorno.especificacion
    det = DetectorFin(esp.indices_articulaciones, det_cfg.get("umbral_estable", 3.0), det_cfg.get("ciclos_estables", 20))
    politica.reiniciar()
    estados, acciones, lats = [], [], []
    terminado, motivo, t0 = False, "tiempo_agotado", time.time()
    for paso in range(int(max_duracion_s * esp.hz)):
        obs = entorno.observar()
        ti = time.perf_counter()
        try:
            a = np.asarray(politica.predecir(obs, instruccion), dtype=np.float32)
        except Exception as exc:
            motivo = f"error_vla:{exc}"
            break
        lats.append(time.perf_counter() - ti)
        estados.append(obs.estado.copy())
        acciones.append(a)
        if det.actualizar(obs.estado, a) and paso + 1 >= det_cfg.get("min_pasos", 10):
            terminado, motivo = True, "estable"
            break
        entorno.actuar(a)
        entorno.esperar_ciclo()
    return {"terminado": terminado, "motivo": motivo, "duracion_s": time.time() - t0, "pasos": len(acciones),
            "estados": np.array(estados), "acciones": np.array(acciones), "latencias": lats}


def correr(cfg: Dict[str, Any], corrida: Corrida, modelos: List[str], rapido: bool = False) -> None:
    cl = cfg["lazo"]
    rr = cfg.get("rapido", {})
    ent_cfg = dict(cl["entorno"])
    entorno = crear_entorno(ent_cfg.pop("tipo", "ros2"), **ent_cfg)
    entorno.conectar()
    juez = Juez(cl.get("juez", {}), entorno)
    escenas = cl.get("escenas") or ["escena por defecto"]
    n_rep = int(rr.get("ensayos_por_escena", 1)) if rapido else int(cl.get("ensayos_por_escena", 4))
    if rapido:
        escenas = escenas[: int(rr.get("max_escenas", 2))]
    tareas = []
    for t in cl["instrucciones"]:
        variantes = [t["instruccion"]] + [rellenar(p, {}, {"object": t["objeto"], "destination": t["destino"]})
                                          for p in (cl.get("parafrasis") or [])]
        for v, texto in enumerate(variantes):
            tareas.append({**t, "instruccion": texto, "variante": v, "base": t["instruccion"]})
    ensayos = [(t, e, k) for t in tareas for e in range(len(escenas)) for k in range(n_rep)]
    print(f"{len(ensayos)} ensayos por modelo ({len(tareas)} instrucciones/redacciones x {len(escenas)} escenas x {n_rep})"
          f" | juez: {juez.modo} | reinicio: {cl.get('reinicio', {}).get('modo', 'manual')}")
    hechos = corrida.hechos("ensayos", ["modelo", "clave"])
    pinzas = [b.pinza.indice_accion for b in entorno.robot.brazos if b.pinza]
    art = entorno.especificacion.indices_articulaciones
    try:
        for modelo in modelos:
            faltan = [(t, e, k) for t, e, k in ensayos
                      if (modelo, f"{t['base']}|{t['variante']}|{e}|{k}") not in hechos]
            if not faltan:
                continue
            politica = bcfg.modelo("vla", modelo)
            print(f"\n=== {modelo} | {len(faltan)} ensayos")
            for n, (t, e, k) in enumerate(faltan, 1):
                print(f"\n[{modelo}] ensayo {n}/{len(faltan)}: \"{t['instruccion']}\" | escena: {escenas[e]} | rep {k + 1}")
                _reiniciar(cl.get("reinicio", {}), entorno, escenas[e])
                r = ejecutar(politica, entorno, t["instruccion"], cl.get("detector", {}), float(cl.get("max_duracion_s", 90)))
                veredicto = juez.calificar(t, r["terminado"])
                A, S = r["acciones"], r["estados"]
                ciclos = 0
                for j in pinzas:
                    if len(A) and j < A.shape[1]:
                        cerrada = A[:, j] > float(cfg.get("offline", {}).get("umbral_pinza", 40.0))
                        ciclos += int(np.sum(cerrada[1:] & ~cerrada[:-1]))
                ja = [i for i in art if len(S) and i < S.shape[1]]
                corrida.escribir("ensayos", {
                    "modelo": modelo, "clave": f"{t['base']}|{t['variante']}|{e}|{k}", "instruccion": t["instruccion"],
                    "base": t["base"], "variante": t["variante"], "escena": escenas[e], "escena_id": e, "rep": k,
                    **veredicto, "terminado": r["terminado"], "motivo": r["motivo"], "duracion_s": r["duracion_s"],
                    "pasos": r["pasos"], "latencia_media_s": float(np.mean(r["latencias"])) if r["latencias"] else None,
                    "latencia_p95_s": float(np.percentile(r["latencias"], 95)) if r["latencias"] else None,
                    "ciclos_pinza": ciclos,
                    "recorrido_articular": float(np.abs(np.diff(S[:, ja], axis=0)).sum()) if len(S) > 1 else 0.0,
                    "jerk": float(np.abs(np.diff(S[:, ja], n=3, axis=0)).mean()) if len(S) > 3 else None})
                print(f"  -> {veredicto['etapa']} ({r['motivo']}, {r['duracion_s']:.1f} s, {r['pasos']} pasos)")
            del politica
    finally:
        entorno.cerrar()
