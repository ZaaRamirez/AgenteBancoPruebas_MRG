"""El agente fundacional: une las capas en el ciclo ReAct de la Seccion III.

    tarea --> [interaccion] --> [razonamiento LLM] <--> [percepcion VLM]
                                        |                    ^
                                        v                    |
                                  [ejecucion VLA] --> entorno (simulado | ROS 2)

modo "modular":           el LLM descompone la tarea en subtareas; el VLA ejecuta cada
                          una; el VLM observa y verifica; el agente lleva el estado.
modo "extremo_a_extremo": el VLA recibe la tarea completa (referencia MolmoAct2); el
                          mismo VLM verifica el resultado final.
Nada de este archivo depende de la tarea, del modelo ni de si el robot es simulado
o fisico: eso lo deciden el perfil de tarea, el catalogo de modelos y el entorno.
"""
from __future__ import annotations

import time
from typing import Any, Callable, Dict, Optional

from .capas.ejecucion import CapaEjecucion
from .capas.percepcion import CapaPercepcion
from .capas.razonamiento import CapaRazonamiento, ContextoTarea
from .entornos import crear_entorno
from .modelos import catalogo
from .registro import Registro
from .tareas.perfiles import load_profile
from .tipos import ResultadoTarea, Tarea
from .util.texto import rellenar

PREGUNTA_FINAL = ('The robot was asked to: "{instruction}". Look at the image: '
                  'is the whole instruction satisfied?')
ESQUEMA_EXITO = {"type": "object", "properties": {"success": {"type": "boolean"}, "reason": {"type": "string"}},
                 "required": ["success"]}


class Agente:
    def __init__(self, cfg: Dict[str, Any], registro: Optional[Registro] = None,
                 reportar: Optional[Callable[[Dict[str, Any]], None]] = None):
        self.cfg = cfg
        self.modo = cfg.get("modo", "modular")
        self.registro = registro
        self.reportar = reportar or (lambda e: None)
        ent = dict(cfg.get("entorno", {}))
        self.entorno = crear_entorno(ent.pop("tipo", "simulado"), **ent)
        self.entorno.conectar()
        m = cfg.get("modelos", {})
        cr, cp, ce = cfg.get("razonamiento", {}), cfg.get("percepcion", {}), cfg.get("ejecucion", {})
        self.razonamiento = None
        if self.modo == "modular":
            self.razonamiento = CapaRazonamiento(catalogo.cargar("llm", m["llm"]), registro,
                                                 cr.get("salida_estructurada", "json"),
                                                 cr.get("max_reintentos_formato", 2))
        self.percepcion = CapaPercepcion(catalogo.cargar("vlm", m["vlm"]), registro,
                                         cp.get("salida_estructurada", "json"), cp.get("camaras"))
        self.verificar_subtareas = bool(cp.get("verificar_subtareas", True))
        self.ejecucion = CapaEjecucion(catalogo.cargar("vla", m["vla"]), self.entorno, registro,
                                       ce.get("umbral_estable", 3.0), ce.get("ciclos_estables", 20),
                                       ce.get("min_pasos", 10), ce.get("max_duracion_s", 90.0),
                                       ce.get("registrar_cada", 0))
        self.max_duracion_e2e = float(ce.get("max_duracion_extremo_s", 300.0))
        self.limites = cfg.get("agente", {})

    # ------------------------------------------------------------------ utilidades
    def modelos(self) -> Dict[str, str]:
        d = {"vlm": self.percepcion.modelo, "vla": self.ejecucion.modelo}
        if self.razonamiento:
            d["llm"] = self.razonamiento.modelo
        return d

    def usar_registro(self, registro: Registro) -> None:
        """Cambia el registro de todas las capas (una carpeta de resultados por configuracion)."""
        self.registro = registro
        for capa in (self.razonamiento, self.percepcion, self.ejecucion):
            if capa is not None:
                capa.registro = registro

    def cambiar_modelo(self, capa: str, nombre: str) -> None:
        """Sustituye el modelo de una capa sin reiniciar el agente."""
        modelo = catalogo.cargar(capa, nombre)
        if capa == "llm" and self.razonamiento:
            self.razonamiento.llm = modelo
        elif capa == "vlm":
            self.percepcion.vlm = modelo
        elif capa == "vla":
            self.ejecucion.vla = modelo
        else:
            raise ValueError(f"Capa '{capa}' no valida (llm | vlm | vla)")
        self.cfg.setdefault("modelos", {})[capa] = nombre
        if self.registro:
            self.registro.evento("cambio_modelo", capa=capa, modelo=nombre)

    def _estado(self, tarea_id, estado, detalle="", **extra):
        e = {"tarea": tarea_id, "estado": estado, "detalle": detalle, "t": time.time(), **extra}
        if self.registro:
            self.registro.evento("estado", estado=estado, detalle=detalle, **extra)
        self.reportar(e)

    def _fin(self, c_o_tarea, estado, exito, detalle, inicio, **kw) -> ResultadoTarea:
        c = c_o_tarea if isinstance(c_o_tarea, ContextoTarea) else None
        tid = c.tarea_id if c else c_o_tarea
        r = ResultadoTarea(tarea_id=tid, estado=estado, exito=exito, detalle=detalle,
                           duracion_s=time.time() - inicio,
                           subtareas=c.indice_subtarea if c else kw.get("subtareas", 1),
                           reintentos=c.reintentos if c else 0,
                           progreso=c.seguimiento.progress() if c else (1.0 if exito else 0.0),
                           llamadas_llm=c.llamadas_llm if c else 0,
                           subtareas_exitosas=c.subtareas_exitosas if c else (1 if exito else 0),
                           unidades_requeridas=sum(getattr(c.seguimiento, "required", {}).values())
                           if (c and hasattr(c.seguimiento, "required")) else -1)
        self._estado(tid, estado, detalle, exito=exito, progreso=r.progreso)
        if self.registro:
            self.registro.tarea(r, modo=self.modo, modelos=self.modelos(),
                                perfil=kw.get("perfil", c.perfil.name if c else ""))
        return r

    # ------------------------------------------------------------------ punto de entrada
    def resolver(self, tarea: Tarea) -> ResultadoTarea:
        if self.registro:
            self.registro.iniciar_episodio(tarea.id)
            self.registro.evento("tarea", datos_tarea=tarea.como_dict(), modo=self.modo, modelos=self.modelos())
        try:
            if self.modo == "extremo_a_extremo":
                return self._extremo_a_extremo(tarea)
            return self._modular(tarea)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            return self._fin(tarea.id, "fallida", False, f"error: {exc}", time.time(), perfil=tarea.perfil)

    # ------------------------------------------------------------------ modo modular
    def _modular(self, tarea: Tarea) -> ResultadoTarea:
        inicio = time.time()
        p = load_profile(tarea.perfil)
        params = dict(tarea.parametros)
        instr = p.instruction(params, tarea.instruccion)
        c = ContextoTarea(tarea.id, instr, params, p, referencia=dict(tarea.referencia))
        ctx = {"tarea": tarea.id, "parametros": params, "referencia": dict(tarea.referencia)}
        max_reint = int(self.limites.get("max_reintentos", p.limits.get("max_retries", 2)))
        max_llm = int(self.limites.get("max_llamadas_llm", p.limits.get("max_llm_calls", 40)))
        self._estado(tarea.id, "aceptada", instr)

        q = p.initial_observation
        if q:
            r = self.percepcion.preguntar("observacion_inicial", rellenar(q.question, params, {"instruction": instr}),
                                          self.entorno.observar(), q.schema, rellenar(q.simulated_answer, params),
                                          contexto_extra=ctx)
            c.observaciones.append({"proposito": "inicial", "pregunta": q.question, "respuesta": r.texto[:600]})
            if q.check_with_tracker and isinstance(r.datos, dict):
                ok, por_que = c.seguimiento.check_observation(r.datos)
                if not ok:
                    return self._fin(c, "no_realizable", False, por_que, inicio)
            if q.abort_if_false and isinstance(r.datos, dict) and r.datos.get(q.abort_if_false) is False:
                return self._fin(c, "no_realizable", False, f"{q.abort_if_false}=false", inicio)

        reint_sub = 0
        while True:
            if p.auto_complete and c.seguimiento.is_complete():
                return self._cerrar(c, inicio, "seguimiento completo")
            if c.llamadas_llm >= max_llm:
                return self._fin(c, "fallida", False, "limite de llamadas al LLM", inicio)
            d = self.razonamiento.decidir(c)
            if not d.valida:
                return self._fin(c, "fallida", False, f"el LLM no dio una accion valida ({d.motivo})", inicio)
            if d.accion == "done":
                return self._cerrar(c, inicio, "el LLM declaro la tarea completa")
            if d.accion == "abort":
                return self._fin(c, "abortada", False, str(d.argumentos.get("reason", "")), inicio)
            if d.accion == "observe":
                pregunta = str(d.argumentos["question"])
                r = self.percepcion.preguntar("pregunta_llm", pregunta, self.entorno.observar(), contexto_extra=ctx)
                c.observaciones.append({"proposito": "llm", "pregunta": pregunta, "respuesta": r.texto[:600]})
                continue

            # ---- habilidad: subtarea para el VLA
            s = p.skill(d.accion)
            instr_sub = CapaRazonamiento.instruccion_subtarea(c, d.accion, d.argumentos)
            self._estado(tarea.id, "subtarea", instr_sub, indice=c.indice_subtarea, habilidad=d.accion,
                         argumentos=d.argumentos, pensamiento=d.pensamiento)
            if s.executor == "vla":
                ej = self.ejecucion.ejecutar(instr_sub)
                exito, detalle = ej.terminado, ej.motivo
            else:
                exito, detalle = True, "sin ejecucion fisica"
            verificado = None
            if exito and s.verify and self.verificar_subtareas:
                v = self.percepcion.preguntar("verificacion", rellenar(s.verify.question, params, dict(d.argumentos)),
                                              self.entorno.observar(), s.verify.schema,
                                              rellenar(s.verify.simulated_answer, params, dict(d.argumentos)),
                                              contexto_extra={**ctx, "habilidad": d.accion, "argumentos": dict(d.argumentos),
                                                              "intento": c.indice_subtarea})
                verificado = bool(isinstance(v.datos, dict) and v.datos.get("success") is True)
                exito = verificado
                detalle = (v.datos or {}).get("reason", v.texto[:200]) if isinstance(v.datos, dict) else "verificacion ilegible"
            c.historial.append({"indice": c.indice_subtarea, "habilidad": d.accion, "argumentos": d.argumentos,
                                "exito": exito, "detalle": str(detalle)})
            if self.registro:
                self.registro.evento("subtarea", indice=c.indice_subtarea, habilidad=d.accion,
                                     argumentos=d.argumentos, instruccion=instr_sub, exito=exito,
                                     verificado=verificado, detalle=str(detalle))
            c.indice_subtarea += 1
            if exito:
                c.seguimiento.register(d.accion, d.argumentos)
                reint_sub = 0
                self._estado(tarea.id, "subtarea_exitosa", instr_sub, progreso=c.seguimiento.progress())
            else:
                c.reintentos += 1
                reint_sub += 1
                self._estado(tarea.id, "subtarea_fallida", f"{instr_sub}: {detalle}")
                if reint_sub > max_reint:
                    return self._fin(c, "fallida", False, f"subtarea fallida tras {max_reint} reintentos", inicio)

    def _cerrar(self, c: ContextoTarea, inicio, motivo) -> ResultadoTarea:
        q = c.perfil.final_check
        if q:
            r = self.percepcion.preguntar("revision_final", rellenar(q.question, c.parametros, {"instruction": c.instruccion}),
                                          self.entorno.observar(), q.schema or ESQUEMA_EXITO,
                                          rellenar(q.simulated_answer, c.parametros),
                                          contexto_extra={"tarea": c.tarea_id, "parametros": c.parametros,
                                                          "referencia": c.referencia})
            ok = bool(isinstance(r.datos, dict) and r.datos.get("success") is True)
            return self._fin(c, "completada" if ok else "fallida", ok,
                             f"{motivo}; revision final: {(r.datos or {}).get('reason', '') if isinstance(r.datos, dict) else r.texto[:100]}",
                             inicio)
        return self._fin(c, "completada", True, motivo, inicio)

    # ------------------------------------------------------------------ extremo a extremo
    def _extremo_a_extremo(self, tarea: Tarea) -> ResultadoTarea:
        inicio = time.time()
        p = load_profile(tarea.perfil)
        params = dict(tarea.parametros)
        instr = tarea.instruccion or rellenar(p.e2e_instruction_template or p.instruction_template, params)
        self._estado(tarea.id, "aceptada", f"extremo a extremo: {instr}")
        ej = self.ejecucion.ejecutar(instr, self.max_duracion_e2e)
        if not ej.terminado:
            return self._fin(tarea.id, "fallida", False, ej.motivo, inicio, perfil=p.name)
        q = p.final_check
        pregunta = rellenar(q.question if q else PREGUNTA_FINAL, params, {"instruction": instr})
        r = self.percepcion.preguntar("revision_final", pregunta, self.entorno.observar(),
                                      (q.schema if q and q.schema else ESQUEMA_EXITO),
                                      contexto_extra={"tarea": tarea.id, "parametros": params, "e2e": True,
                                                      "referencia": dict(tarea.referencia)})
        ok = bool(isinstance(r.datos, dict) and r.datos.get("success") is True)
        return self._fin(tarea.id, "completada" if ok else "fallida", ok,
                         f"verificado por percepcion: {(r.datos or {}).get('reason', '') if isinstance(r.datos, dict) else ''}",
                         inicio, perfil=p.name)

    def cerrar(self):
        self.entorno.cerrar()
