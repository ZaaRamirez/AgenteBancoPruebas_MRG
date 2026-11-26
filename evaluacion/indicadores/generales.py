"""Indicadores iniciales GENERALES: no dependen de la tarea ni de una referencia.

Son un punto de partida; la lista definitiva se decide despues. Cada funcion
recibe una Corrida y devuelve filas. Tipos de evento disponibles en eventos.jsonl:
  llm        prompt, respuesta, latencia_s, formato_ok, valida, motivo, accion, argumentos
  vlm        proposito, pregunta, respuesta, datos, latencia_s, formato_ok, imagen
  vla        instruccion, terminado, motivo, pasos, duracion_s, latencia_media_s
  subtarea   habilidad, argumentos, instruccion, exito, verificado, detalle
  estado     estado, detalle            (aceptada, subtarea, completada, ...)
y en corrida.resultados (tareas.jsonl): estado, exito, duracion_s, subtareas,
reintentos, progreso, llamadas_llm, modelos, modo, perfil.
"""
from collections import defaultdict

from evaluar import fila

ERRORES_DE_ESTADO = ("already_complete", "fin_prematuro", "unknown_")


def _por_modelo(eventos):
    g = defaultdict(list)
    for e in eventos:
        g[e.get("modelo", "?")].append(e)
    return g


def razonamiento(c):
    out = []
    for m, ev in _por_modelo(c.eventos("llm")).items():
        out += [
            fila("razonamiento", m, "formato_valido", [e["formato_ok"] for e in ev], "fraccion",
                 "respuestas con JSON y campo action"),
            fila("razonamiento", m, "decision_valida", [e["valida"] for e in ev], "fraccion",
                 "acciones aplicables: habilidad existente, argumentos completos, coherente con el estado"),
            fila("razonamiento", m, "errores_de_estado",
                 [str(e["motivo"]).startswith(ERRORES_DE_ESTADO) for e in ev if e["formato_ok"]], "fraccion",
                 "repetir lo ya hecho, terminar antes de tiempo o inventar elementos"),
            fila("razonamiento", m, "latencia_s", [e["latencia_s"] for e in ev], "s", "tiempo por consulta"),
        ]
    return out


def percepcion(c):
    out = []
    for m, ev in _por_modelo(c.eventos("vlm")).items():
        out += [fila("percepcion", m, "formato_valido", [e["formato_ok"] for e in ev], "fraccion",
                     "respuestas que cumplen el esquema pedido"),
                fila("percepcion", m, "latencia_s", [e["latencia_s"] for e in ev], "s", "tiempo por consulta")]
        for p in sorted({e["proposito"] for e in ev}):
            sub = [e for e in ev if e["proposito"] == p]
            out.append(fila("percepcion", m, f"latencia_{p}_s", [e["latencia_s"] for e in sub], "s"))
            if p in ("verificacion", "revision_final"):
                out.append(fila("percepcion", m, f"positivos_{p}",
                                [isinstance(e.get("datos"), dict) and e["datos"].get("success") is True for e in sub],
                                "fraccion", "veredictos de exito (sin referencia NO es exactitud)"))
    return out


def ejecucion(c):
    out = []
    for m, ev in _por_modelo(c.eventos("vla")).items():
        out += [fila("ejecucion", m, "fin_detectado", [e["terminado"] for e in ev], "fraccion",
                     "ejecuciones que terminaron por estabilidad (no por tiempo o error)"),
                fila("ejecucion", m, "duracion_s", [e["duracion_s"] for e in ev], "s", "tiempo por ejecucion"),
                fila("ejecucion", m, "pasos", [e["pasos"] for e in ev], "pasos"),
                fila("ejecucion", m, "latencia_inferencia_s", [e["latencia_media_s"] for e in ev], "s",
                     "tiempo medio por paso de control")]
    return out


def agente(c):
    rs = c.resultados
    if not rs:
        return []
    combo = " + ".join(f"{k}:{v}" for k, v in sorted((rs[0].get("modelos") or {}).items()))
    out = [fila("agente", combo, "exito_tarea", [r["exito"] for r in rs], "fraccion"),
           fila("agente", combo, "no_realizable", [r["estado"] == "no_realizable" for r in rs], "fraccion",
                "precondicion no cumplida (p. ej. faltante)"),
           fila("agente", combo, "tiempo_tarea_s", [r["duracion_s"] for r in rs], "s"),
           fila("agente", combo, "reintentos", [r["reintentos"] for r in rs], "por tarea"),
           fila("agente", combo, "subtareas", [r["subtareas"] for r in rs], "por tarea"),
           fila("agente", combo, "llamadas_llm", [r["llamadas_llm"] for r in rs], "por tarea"),
           fila("agente", combo, "progreso", [r["progreso"] for r in rs if r.get("progreso", -1) >= 0], "fraccion",
                "elementos completados segun el seguimiento del agente"),
           fila("agente", combo, "subtareas_exitosas", [e["exito"] for e in c.eventos("subtarea")], "fraccion",
                "subtareas confirmadas entre intentadas")]
    return out


INDICADORES = [razonamiento, percepcion, ejecucion, agente]
