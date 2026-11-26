"""REPORTE del benchmark: junta las corridas de las tres capas en un documento ordenado.

  python3 -m benchmark reporte                      # la ultima corrida de cada tipo
  python3 -m benchmark reporte <carpeta> [<carpeta> ...]

Escribe resultados/benchmark/reporte_<fecha>/REPORTE.md, un CSV por tabla y, si hay
matplotlib, una grafica por ranking (metrica principal con su IC 95 %).
Estructura: resumen (modelo recomendado por capa) -> por capa: que se mide, bajo que
parametros, resultados (ranking, desgloses, errores, robustez, costo).
"""
from __future__ import annotations

import glob
import os
import re
import time
from typing import Any, Dict, List

import yaml

from .comun.almacen import BASE
from .comun.tablas import Tabla

TIPOS = ["llm_casos", "llm_episodios", "vlm", "vla_offline", "vla_lazo"]
# modelos de referencia del banco (no se recomiendan; sirven para validar y como piso)
REFERENCIAS = {"guion", "postura_prueba", "simulado", "simulado_70", "simulado_faltante", "escena"}

QUE_SE_MIDE = {
    "llm_casos": """Cada caso congela un momento de una tarea (qué se pidió, qué ya se hizo, qué falló, qué ve la
percepción) y pide al LLM la **siguiente acción**; la respuesta correcta se conoce de antemano. El prompt es
el mismo que construye el agente. Se mide:

- **Exactitud de decisión** (principal): fracción de casos con una acción correcta.
- **Formato válido**: respuestas con JSON y campo `action`.
- **Errores de estado**: decisiones que dejarían la tarea mal (repetir lo hecho, tomar algo no pedido o
  inventado, terminar antes, ignorar a la percepción, orden o destino equivocado).
- **Errores por tipo**, **exactitud por categoría**, **estabilidad** entre repeticiones (con muestreo),
  **latencia** (media, p95), **tokens**, **respuestas truncadas** y **memoria**.""",
    "llm_episodios": """El LLM resuelve **tareas completas** dentro del agente real. La percepción es un oráculo
que conoce la escena (responde la verdad, salvo los fallos inyectados) y la ejecución es simulada: si la tarea
sale mal, fue el razonamiento. Se mide:

- **Plan correcto** (principal): hizo exactamente lo pedido, sin pasos de más, de menos ni fuera de orden.
- **Éxito**, **progreso**, **llamadas por subtarea exitosa** (eficiencia), **decisiones inválidas por tarea**,
  **recuperación tras fallo** (éxito cuando se inyectaron fallos) y **tiempo de razonamiento por tarea**.""",
    "vlm": """El VLM responde las **mismas preguntas que hace el agente** sobre imágenes con respuesta conocida,
con varias redacciones de la pregunta y varias perturbaciones de la imagen. Se mide:

- **Verificación de subtareas** (¿quedó el objeto en el destino?): **exactitud balanceada** (principal),
  sensibilidad, especificidad, **falsos positivos** (el error más costoso: el agente avanza sin reintentar),
  F1, MCC, sin respuesta.
- **Inventario**: **conteo exacto por imagen** (principal), conteo por elemento, MAE, detección de presencia,
  **alucinaciones** y omisiones.
- **Robustez** por perturbación, **invariancia** frente a la imagen original, **consistencia** entre
  redacciones, latencia y memoria.""",
    "vla_offline": """El VLA recibe las **observaciones grabadas** de demostraciones (episodios de prueba) y su
acción se compara con la del operador, fotograma por fotograma. Se mide:

- **MAE articular** por episodio (principal, grados; menor es mejor), RMSE, error máximo, por articulación y
  final.
- **Dentro de tolerancia** (todas las articulaciones a menos de N °), **acierto de pinza**, **suavidad**
  (jerk relativo al operador).
- **Latencia** de inferencia, **frecuencia alcanzable**, **memoria GPU** y **sensibilidad al lenguaje**.""",
    "vla_lazo": """El VLA **mueve el robot** (gemelo MuJoCo o físico) con el mismo entorno que usa el agente; un
juez decide si cumplió la tarea. Los ensayos están pareados por escena inicial. Se mide:

- **Tasa de éxito** (principal), etapa alcanzada (agarró sin colocar, no agarró), éxito por escena y por
  redacción.
- Terminación por estabilidad o por tiempo, duración, pasos, ciclos de pinza (reagarres), recorrido, latencia.""",
}
TITULOS = {"llm_casos": "Razonamiento (LLM) — casos de decisión", "llm_episodios": "Razonamiento (LLM) — tareas completas",
           "vlm": "Percepción (VLM)", "vla_offline": "Ejecución (VLA) — fuera de línea",
           "vla_lazo": "Ejecución (VLA) — lazo cerrado"}
CAPA = {"llm_casos": "LLM", "llm_episodios": "LLM", "vlm": "VLM", "vla_offline": "VLA", "vla_lazo": "VLA"}


def _ultimas() -> List[str]:
    out = []
    for t in TIPOS:
        cs = sorted(d for d in glob.glob(os.path.join(BASE, t, "*")) if os.path.isfile(os.path.join(d, "config.yaml")))
        if cs:
            out.append(cs[-1])
    return out


def _tipo(carpeta: str) -> str:
    cfg = yaml.safe_load(open(os.path.join(carpeta, "config.yaml"), encoding="utf-8")) or {}
    return cfg.get("tipo") or os.path.basename(os.path.dirname(carpeta.rstrip("/")))


def _metricas(tipo: str, carpeta: str) -> Dict[str, Any]:
    if tipo == "llm_casos":
        from .llm.metricas import casos
        return casos(carpeta)
    if tipo == "llm_episodios":
        from .llm.metricas import episodios
        return episodios(carpeta)
    if tipo == "vlm":
        from .vlm.metricas import calcular
        return calcular(carpeta)
    if tipo == "vla_offline":
        from .vla.metricas import offline
        return offline(carpeta)
    if tipo == "vla_lazo":
        from .vla.metricas import lazo
        return lazo(carpeta)
    return {"secciones": [], "resumen": {}}


def _parametros(tipo: str, cfg: Dict[str, Any], carpeta: str) -> Tabla:
    t = Tabla("Parámetros de la corrida", ["Parámetro", "Valor"])
    t.agregar("Modelos", ", ".join(map(str, cfg.get("modelos", []))))
    t.agregar("Perfil rápido (prueba)", "sí" if cfg.get("rapido") else "no")
    if cfg.get("parametros_comunes"):
        t.agregar("Parámetros comunes de inferencia",
                  ", ".join(f"{k}={v}" for k, v in cfg["parametros_comunes"].items()))
    from .comun.almacen import leer_jsonl
    if tipo == "llm_casos":
        casos = leer_jsonl(os.path.join(carpeta, "casos.jsonl"))
        cats = sorted({c["categoria"] for c in casos})
        t.agregar("Casos", f"{len(casos)} ({len(cats)} categorías × {len(casos) // max(1, len(cats))}); "
                           f"semilla {cfg.get('casos', {}).get('semilla')}")
        for c in cfg.get("condiciones", []):
            t.agregar(f"Condición {c['nombre']}", f"salida={c.get('salida_estructurada')}, temperature="
                      f"{c.get('temperature')}, repeticiones={c.get('repeticiones', 1)}")
    elif tipo == "llm_episodios":
        e = cfg.get("episodios", {})
        eps = leer_jsonl(os.path.join(carpeta, "episodios.jsonl"))
        tareas = sorted({x["tarea"] for x in eps})
        t.agregar("Tareas", f"{len(tareas)} (pedidos reales de comanda.xlsx, pedidos generados e instrucciones libres)")
        t.agregar("Repeticiones", e.get("repeticiones"))
        for c in e.get("condiciones", []):
            t.agregar(f"Condición {c['nombre']}", f"probabilidad de éxito de cada subtarea = {c['tasa_exito']}")
        t.agregar("Inferencia", f"temperature={e.get('temperature')}, salida={e.get('salida_estructurada')}")
    elif tipo == "vlm":
        resp = leer_jsonl(os.path.join(carpeta, "respuestas.jsonl"))
        t.agregar("Conjuntos etiquetados", ", ".join(map(str, cfg.get("conjuntos", []))))
        t.agregar("Imágenes etiquetadas", len({r["caso"] for r in resp}))
        t.agregar("Variantes de pregunta", ", ".join(sorted({r["pregunta"] for r in resp})) + " (q0 = la del agente)")
        t.agregar("Perturbaciones", ", ".join(dict.fromkeys(r["perturbacion"] for r in resp)))
        t.agregar("Salida estructurada", cfg.get("salida_estructurada"))
        t.agregar("Consultas por modelo", len(resp) // max(1, len({r['modelo'] for r in resp})))
    elif tipo == "vla_offline":
        o = cfg.get("offline", {})
        t.agregar("Fuente de demostraciones", o.get("fuente"))
        t.agregar("Episodios de prueba", o.get("episodios"))
        t.agregar("Paso entre fotogramas", o.get("paso"))
        t.agregar("Tolerancia / umbral de pinza", f"{o.get('tolerancia_grados')} ° / {o.get('umbral_pinza')}")
        t.agregar("Robot (vector de acción)", cfg.get("robot"))
    elif tipo == "vla_lazo":
        l = cfg.get("lazo", {})
        t.agregar("Entorno", l.get("entorno"))
        t.agregar("Instrucciones", "; ".join(i["instruccion"] for i in l.get("instrucciones", [])))
        t.agregar("Redacciones extra", "; ".join(l.get("parafrasis") or []) or "–")
        t.agregar("Escenas × ensayos", f"{len(l.get('escenas') or [])} × {l.get('ensayos_por_escena')}")
        t.agregar("Juez / reinicio", f"{l.get('juez', {}).get('modo')} / {l.get('reinicio', {}).get('modo')}")
        t.agregar("Tiempo máximo por ensayo", f"{l.get('max_duracion_s')} s")
    return t


def _grafica(rk: List[Dict], titulo: str, unidad: str, ruta: str, porcentaje: bool) -> str:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    f = 100 if porcentaje else 1
    nombres = [r["modelo"] for r in rk][::-1]
    medias = [f * r["media"] for r in rk][::-1]
    err = [[f * (r["media"] - r["ic_bajo"]) for r in rk][::-1], [f * (r["ic_alto"] - r["media"]) for r in rk][::-1]]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(rk) + 1.4))
    colores = ["#2b6cb0" if r["grupo"] == "A" else "#a0aec0" for r in rk][::-1]
    ax.barh(nombres, medias, xerr=err, color=colores, capsize=4)
    tope = max([f * r["ic_alto"] for r in rk if r["ic_alto"] == r["ic_alto"]] + medias + [1e-9])
    for y, r in enumerate(rk[::-1]):
        x = f * r["ic_alto"] if r["ic_alto"] == r["ic_alto"] else f * r["media"]
        ax.text(x + 0.02 * tope, y, f"{f * r['media']:.1f}{unidad} · grupo {r['grupo']}", va="center", fontsize=8)
    if porcentaje:
        ax.set_xlim(0, 128)
        ax.set_xticks([0, 20, 40, 60, 80, 100])
    else:
        ax.set_xlim(0, tope * 1.35)
    ax.set_title(titulo, fontsize=10)
    ax.set_xlabel(unidad.strip() or "valor")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(ruta, dpi=150)
    plt.close(fig)
    return os.path.basename(ruta)


def generar(carpetas: List[str], salida: str = "") -> str:
    carpetas = [os.path.abspath(c) for c in carpetas] or _ultimas()
    if not carpetas:
        print("No hay corridas en", BASE)
        return ""
    dir_rep = salida or os.path.join(BASE, f"reporte_{time.strftime('%Y%m%d_%H%M%S')}")
    os.makedirs(dir_rep, exist_ok=True)
    partes, resumen = [], Tabla("Modelo recomendado por capa (mejor modelo real y empatados con él)",
                                ["Capa", "Prueba", "Métrica principal", "Mejor modelo", "Valor [IC 95 %]",
                                 "Sin diferencia significativa con él", "Corrida"],
                                "Se excluyen los modelos de referencia del banco (guion, postura_prueba, simulado).")
    equipo = {}
    n_tabla = 0
    for k, carpeta in enumerate(sorted(carpetas, key=lambda c: TIPOS.index(_tipo(c)) if _tipo(c) in TIPOS else 99), 1):
        tipo = _tipo(carpeta)
        cfg = yaml.safe_load(open(os.path.join(carpeta, "config.yaml"), encoding="utf-8")) or {}
        if os.path.exists(os.path.join(carpeta, "entorno.yaml")):
            equipo = yaml.safe_load(open(os.path.join(carpeta, "entorno.yaml"), encoding="utf-8")) or equipo
        m = _metricas(tipo, carpeta)
        partes.append(f"\n## {k}. {TITULOS.get(tipo, tipo)}\n\n`{os.path.relpath(carpeta, BASE)}`\n")
        partes.append(f"### {k}.1 Qué se mide\n\n{QUE_SE_MIDE.get(tipo, '')}\n")
        partes.append(f"### {k}.2 Bajo qué parámetros\n\n{_parametros(tipo, cfg, carpeta).markdown()}")
        partes.append(f"### {k}.3 Resultados\n")
        for sec in m["secciones"]:
            partes.append(f"#### {sec['titulo']}\n")
            for t in sec["tablas"]:
                n_tabla += 1
                partes.append(t.markdown())
                t.csv(os.path.join(dir_rep, f"tabla_{n_tabla:02d}_{re.sub(r'[^a-z0-9]+', '_', t.titulo.lower())[:50]}.csv"))
        for clave, rk in (m.get("resumen") or {}).items():
            if not rk:
                continue
            principal = {"llm_casos": ("exactitud de decisión", " %", True),
                         "llm_episodios": ("plan correcto", " %", True),
                         "vlm": ({"verificacion": "exactitud balanceada (verificación)",
                                  "inventario": "conteo exacto por imagen"}.get(clave, clave), " %", True),
                         "vla_offline": ("MAE articular", " °", False),
                         "vla_lazo": ("tasa de éxito", " %", True)}[tipo]
            refs = REFERENCIAS | set(map(str, cfg.get("referencias", [])))
            reales = [r for r in rk if r["modelo"] not in refs] or rk
            lider = reales[0]
            f = 100 if principal[2] else 1
            grupo_a = [r["modelo"] for r in reales if r["grupo"] == lider["grupo"] and r is not lider]
            prueba = {"llm_casos": "casos de decisión", "llm_episodios": "tareas completas", "vlm": "",
                      "vla_offline": "fuera de línea", "vla_lazo": "lazo cerrado"}[tipo]
            prueba = f"{prueba} · {clave}" if tipo.startswith("llm") else (
                {"verificacion": "verificación", "inventario": "inventario"}.get(clave, prueba) if tipo == "vlm" else prueba)
            resumen.agregar(CAPA[tipo], prueba, principal[0], f"**{lider['modelo']}**",
                            f"{f * lider['media']:.1f}{principal[1]} [{f * lider['ic_bajo']:.1f}–{f * lider['ic_alto']:.1f}]",
                            ", ".join(grupo_a) or "ninguno", os.path.basename(carpeta))
            g = _grafica(rk, f"{TITULOS[tipo]} · {clave}: {principal[0]}", principal[1],
                         os.path.join(dir_rep, f"grafica_{tipo}_{clave}.png"), principal[2])
            if g:
                partes.append(f"\n![{tipo} {clave}]({g})\n")
    enc = ["# Benchmark de modelos para el agente fundacional", "",
           f"Generado: {time.strftime('%Y-%m-%d %H:%M')}  ·  Procedimiento y definiciones: `BENCHMARK.md`", ""]
    if equipo:
        te = Tabla("Equipo donde se midió", ["Dato", "Valor"])
        for kk in ("fecha", "sistema", "cpu", "ram_gb", "gpu", "python", "version_torch", "version_transformers",
                   "version_lerobot"):
            if kk in equipo:
                te.agregar(kk, equipo[kk] if not isinstance(equipo[kk], list) else "; ".join(equipo[kk]))
        enc.append(te.markdown())
    enc += ["## Resumen", "", resumen.markdown(),
            "_Cómo leer: los intervalos son al 95 %. El **grupo** agrupa modelos cuya diferencia con el líder no es "
            "estadísticamente significativa (prueba de signos pareada sobre los mismos casos, corrección de Holm, "
            "α = 0.05). Si varios modelos comparten el grupo A, elige entre ellos por costo (latencia, memoria) o "
            "aumenta la población para separarlos._", ""]
    texto = "\n".join(enc) + "\n".join(partes)
    ruta = os.path.join(dir_rep, "REPORTE.md")
    with open(ruta, "w", encoding="utf-8") as fh:
        fh.write(texto)
    resumen.csv(os.path.join(dir_rep, "resumen.csv"))
    print(resumen.texto())
    print(f"Reporte: {ruta}\nTablas CSV y gráficas en: {dir_rep}")
    return ruta
