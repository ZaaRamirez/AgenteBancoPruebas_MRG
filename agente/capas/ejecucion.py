"""Capa de EJECUCION (VLA): convierte una instruccion en acciones sobre el entorno.

Bucle de control: observar -> predecir accion -> actuar, a la frecuencia del
entorno, hasta que el robot se estabiliza en la accion predicha (fin de la
subtarea) o se agota el tiempo. El exito real lo decide despues la percepcion.
"""
from __future__ import annotations

import time

import numpy as np

from ..tipos import ResultadoEjecucion


class DetectorFin:
    """Fin de subtarea: |estado - accion| medio < umbral durante N ciclos seguidos."""

    def __init__(self, indices, umbral: float = 3.0, ciclos: int = 20):
        self.indices, self.umbral, self.ciclos = list(indices), float(umbral), int(ciclos)
        self.cuenta, self.delta = 0, float("nan")

    def reiniciar(self):
        self.cuenta = 0

    def actualizar(self, estado: np.ndarray, accion: np.ndarray) -> bool:
        idx = [i for i in self.indices if i < len(estado) and i < len(accion)]
        if not idx:
            return False
        self.delta = float(np.abs(estado[idx] - accion[idx]).mean())
        self.cuenta = self.cuenta + 1 if self.delta < self.umbral else 0
        return self.cuenta >= self.ciclos


class CapaEjecucion:
    def __init__(self, vla, entorno, registro=None, umbral_estable: float = 3.0, ciclos_estables: int = 20,
                 min_pasos: int = 10, max_duracion_s: float = 90.0, registrar_cada: int = 0):
        self.vla, self.entorno, self.registro = vla, entorno, registro
        self.umbral, self.ciclos = umbral_estable, ciclos_estables
        self.min_pasos, self.max_duracion_s = int(min_pasos), float(max_duracion_s)
        self.registrar_cada = int(registrar_cada)   # >0: guarda cada N pasos estado/accion

    @property
    def modelo(self) -> str:
        return getattr(self.vla, "etiqueta", getattr(self.vla, "nombre", "?"))

    def ejecutar(self, instruccion: str, max_duracion_s: float = 0.0) -> ResultadoEjecucion:
        esp = self.entorno.especificacion
        max_pasos = int((max_duracion_s or self.max_duracion_s) * esp.hz)   # limite en pasos de control
        det = DetectorFin(esp.indices_articulaciones, self.umbral, self.ciclos)
        self.vla.reiniciar()
        t0, latencias, motivo, terminado, paso = time.time(), [], "tiempo_agotado", False, 0
        while paso < max_pasos:
            obs = self.entorno.observar()
            ti = time.perf_counter()
            try:
                accion = np.asarray(self.vla.predecir(obs, instruccion), dtype=np.float32)
            except Exception as exc:
                motivo = f"error_vla:{exc}"
                break
            latencias.append(time.perf_counter() - ti)
            paso += 1
            if self.registrar_cada and paso % self.registrar_cada == 0 and self.registro:
                self.registro.evento("vla_paso", capa="ejecucion", modelo=self.modelo, paso=paso,
                                     estado=obs.estado, accion=accion, delta=det.delta)
            if obs.estado is not None and det.actualizar(obs.estado, accion) and paso >= self.min_pasos:
                terminado, motivo = True, f"estable (delta={det.delta:.2f})"
                break
            self.entorno.actuar(accion)
            self.entorno.esperar_ciclo()
        r = ResultadoEjecucion(terminado=terminado, motivo=motivo, pasos=paso, duracion_s=time.time() - t0,
                               latencia_media_s=float(np.mean(latencias)) if latencias else 0.0)
        if self.registro:
            self.registro.evento("vla", capa="ejecucion", modelo=self.modelo, instruccion=instruccion,
                                 terminado=r.terminado, motivo=r.motivo, pasos=r.pasos, duracion_s=r.duracion_s,
                                 latencia_media_s=r.latencia_media_s,
                                 latencia_max_s=float(np.max(latencias)) if latencias else 0.0)
        return r
