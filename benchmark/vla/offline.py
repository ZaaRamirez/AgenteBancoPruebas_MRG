"""Benchmark VLA FUERA DE LINEA: el modelo recibe, fotograma por fotograma, las
observaciones grabadas (imagen + estado + instruccion) y su accion se compara con la que
ejecuto el operador en la demostracion. No necesita robot ni simulador.

Se guarda por fotograma (fotogramas.jsonl): error por articulacion, pinza predicha y real,
latencia. Por episodio (episodios.jsonl): resumen. Por redaccion alternativa (lenguaje.jsonl):
cuanto cambia la accion al cambiar solo la redaccion de la instruccion."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from agente.entornos.robot import cargar_robot
from agente.tipos import Observacion
from agente.util.texto import rellenar
from ..comun import config as bcfg
from ..comun.almacen import Corrida
from ..comun.episodios import abrir, seleccionar
from ..vlm.datos import destino_de, objeto_de


def _memoria_gpu():
    try:
        import torch
        if torch.cuda.is_available():
            return round(torch.cuda.max_memory_allocated() / 1e9, 2)
    except Exception:
        pass
    return None


def _reiniciar_memoria():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass


def _recorrer(politica, fuente, ep, instr, cam, paso, max_f):
    politica.reiniciar()
    preds, reales, lats, estados = [], [], [], []
    n = fuente.longitud(ep)
    for k, i in enumerate(range(0, n, paso)):
        if max_f and k >= max_f:
            break
        fg = fuente.fotograma(ep, i)
        obs = Observacion(imagenes={cam: fg["imagen"]}, estado=fg["estado"])
        t0 = time.perf_counter()
        a = np.asarray(politica.predecir(obs, instr), dtype=np.float32).reshape(-1)
        lats.append(time.perf_counter() - t0)
        preds.append(a)
        reales.append(fg["accion"])
        estados.append(fg["estado"])
    return preds, reales, lats


def _jerk(serie: np.ndarray) -> float:
    """Media de |tercera diferencia| (suavidad; menor = mas suave)."""
    return float(np.abs(np.diff(serie, n=3, axis=0)).mean()) if len(serie) > 3 else float("nan")


def correr(cfg: Dict[str, Any], corrida: Corrida, modelos: List[str], rapido: bool = False) -> None:
    co = cfg["offline"]
    robot = cargar_robot(cfg.get("robot", "omx"))
    esp = robot.especificacion()
    art = esp.indices_articulaciones
    pinzas = [b.pinza.indice_accion for b in robot.brazos if b.pinza]
    kw = {"camara": co.get("camara_conjunto", "observation.images.front")} if co["fuente"].startswith("lerobot") else {}
    fuente = abrir(co["fuente"], **kw)
    rr = cfg.get("rapido", {})
    eps = seleccionar(fuente.episodios(), rr.get("episodios", "0-1") if rapido else co.get("episodios", "todos"))
    max_f = int(rr.get("max_fotogramas", 40)) if rapido else 0
    paso, cam = int(co.get("paso", 1)), co.get("camara_robot", "front")
    tol, umbral = float(co.get("tolerancia_grados", 5.0)), float(co.get("umbral_pinza", 40.0))
    print(f"{len(eps)} episodios de prueba: {eps[:20]}{' ...' if len(eps) > 20 else ''}")
    hechos = corrida.hechos("episodios", ["modelo", "episodio"])
    hechos_l = corrida.hechos("lenguaje", ["modelo", "episodio", "variante"])
    for modelo in modelos:
        faltan = [e for e in eps if (modelo, e) not in hechos]
        n_leng = 0 if rapido else int(co.get("episodios_lenguaje", 5))
        faltan_l = [(e, j) for e in eps[:n_leng] for j in range(len(co.get("instrucciones_alternativas") or []))
                    if (modelo, e, j) not in hechos_l]
        if not faltan and not faltan_l:
            continue
        _reiniciar_memoria()
        politica = bcfg.modelo("vla", modelo)
        print(f"\n=== {modelo} | {len(faltan)} episodios")
        t_ini = time.time()
        for k, ep in enumerate(faltan, 1):
            f0 = fuente.fotograma(ep, 0)
            instr = f0["tarea"] or co.get("tarea", "")
            preds, reales, lats = _recorrer(politica, fuente, ep, instr, cam, paso, max_f)
            d = min(len(preds[0]), len(reales[0]))
            P, R = np.stack([p[:d] for p in preds]), np.stack([r[:d] for r in reales])
            E = np.abs(P - R)
            ja = [i for i in art if i < d]
            for i, (p, r, lat) in enumerate(zip(P, R, lats)):
                corrida.escribir("fotogramas", {"modelo": modelo, "episodio": ep, "i": i * paso,
                                                "error": E[i].round(4).tolist(), "latencia_s": lat,
                                                "pinza_pred": [float(p[j]) for j in pinzas if j < d],
                                                "pinza_real": [float(r[j]) for j in pinzas if j < d]})
            pz_ok = [((P[:, j] > umbral) == (R[:, j] > umbral)).mean() for j in pinzas if j < d]
            corrida.escribir("episodios", {
                "modelo": modelo, "episodio": ep, "instruccion": instr, "fotogramas": len(P),
                "mae_articular": float(E[:, ja].mean()), "rmse_articular": float(np.sqrt((E[:, ja] ** 2).mean())),
                "mae_por_articulacion": E[:, ja].mean(axis=0).round(3).tolist(),
                "error_max": float(E[:, ja].max()), "dentro_tolerancia": float((E[:, ja] < tol).all(axis=1).mean()),
                "error_final": float(E[-1, ja].mean()),
                "acierto_pinza": float(np.mean(pz_ok)) if pz_ok else None,
                "mae_pinza": float(E[:, [j for j in pinzas if j < d]].mean()) if pinzas else None,
                "jerk_pred": _jerk(P[:, ja]), "jerk_real": _jerk(R[:, ja]),
                "latencia_media_s": float(np.mean(lats)), "latencia_p95_s": float(np.percentile(lats, 95)),
                "latencia_max_s": float(np.max(lats)), "memoria_gpu_gb": _memoria_gpu(),
                "nombres": [esp.nombres[i] for i in ja]})
            dt = time.time() - t_ini
            print(f"  {modelo}: episodio {ep} ({k}/{len(faltan)})  MAE {E[:, ja].mean():.2f}  "
                  f"({dt / 60:.1f} min, faltan ~{dt / k * (len(faltan) - k) / 60:.1f} min)", flush=True)
        # ---- sensibilidad al lenguaje
        for ep, j in faltan_l:
            f0 = fuente.fotograma(ep, 0)
            instr = f0["tarea"] or co.get("tarea", "")
            alt = rellenar(co["instrucciones_alternativas"][j], {}, {"object": objeto_de(instr),
                                                                      "destination": destino_de(instr, "box")})
            base, reales, _ = _recorrer(politica, fuente, ep, instr, cam, paso, max_f)
            var, _, _ = _recorrer(politica, fuente, ep, alt, cam, paso, max_f)
            d = min(len(base[0]), len(reales[0]))
            B, V, R = (np.stack([x[:d] for x in s]) for s in (base, var, reales))
            ja = [i for i in art if i < d]
            corrida.escribir("lenguaje", {"modelo": modelo, "episodio": ep, "variante": j, "instruccion": alt,
                                          "cambio_accion": float(np.abs(B - V)[:, ja].mean()),
                                          "mae_original": float(np.abs(B - R)[:, ja].mean()),
                                          "mae_variante": float(np.abs(V - R)[:, ja].mean())})
            print(f"  {modelo}: lenguaje ep {ep} variante {j}: cambio {np.abs(B - V)[:, ja].mean():.2f}")
        del politica
