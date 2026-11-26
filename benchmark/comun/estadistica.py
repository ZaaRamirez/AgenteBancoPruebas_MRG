"""Estadistica del benchmark (sin dependencias fuera de numpy).

- Proporciones (exactitud, exito, ...):   intervalo de Wilson al 95 %.
- Medias de magnitudes continuas:          intervalo bootstrap percentil al 95 %.
- Comparacion PAREADA de dos modelos sobre los mismos casos: prueba de signos exacta
  (equivale a McNemar exacta cuando el resultado es acierto/fallo).
- Varias comparaciones a la vez: correccion de Holm.
- Ranking con "grupos de empate": modelos cuya diferencia con el mejor no es
  significativa comparten letra (A, B, ...).
"""
from __future__ import annotations

import math
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np

Z95 = 1.959963984540054


def wilson(k: float, n: int, z: float = Z95):
    """(proporcion, limite inferior, limite superior)."""
    if n <= 0:
        return (float("nan"),) * 3
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / d
    return p, max(0.0, c - h), min(1.0, c + h)


def proporcion(valores: Sequence) -> Dict[str, float]:
    v = [bool(x) for x in valores if x is not None]
    p, lo, hi = wilson(sum(v), len(v))
    return {"valor": p, "ic_bajo": lo, "ic_alto": hi, "n": len(v), "tipo": "proporcion"}


def bootstrap(valores: Sequence, estadistico: Callable = np.mean, b: int = 2000, semilla: int = 0) -> Dict[str, float]:
    v = np.asarray([float(x) for x in valores if x is not None and not (isinstance(x, float) and math.isnan(x))])
    if v.size == 0:
        return {"valor": float("nan"), "ic_bajo": float("nan"), "ic_alto": float("nan"), "n": 0, "tipo": "media"}
    if v.size == 1:
        x = float(estadistico(v))
        return {"valor": x, "ic_bajo": x, "ic_alto": x, "n": 1, "tipo": "media"}
    rng = np.random.default_rng(semilla)
    idx = rng.integers(0, v.size, size=(b, v.size))
    est = np.apply_along_axis(estadistico, 1, v[idx]) if estadistico is not np.mean else v[idx].mean(axis=1)
    return {"valor": float(estadistico(v)), "ic_bajo": float(np.percentile(est, 2.5)),
            "ic_alto": float(np.percentile(est, 97.5)), "n": int(v.size), "tipo": "media"}


def percentil(valores: Sequence, q: float) -> float:
    v = [float(x) for x in valores if x is not None]
    return float(np.percentile(v, q)) if v else float("nan")


def _binom_dos_colas(k: int, n: int) -> float:
    """P de dos colas de una binomial(n, 0.5) exacta (en log para n grande)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    cola = math.exp(m) * sum(math.exp(x - m) for x in logs)
    return min(1.0, 2 * cola)


def prueba_signos(a: Sequence[float], b: Sequence[float]) -> Dict[str, float]:
    """Comparacion pareada caso a caso (mismo caso para ambos modelos).
    a_gana: casos donde a > b; b_gana: casos donde b > a; empates se descartan.
    Con aciertos 0/1 es la prueba exacta de McNemar."""
    a_gana = sum(1 for x, y in zip(a, b) if x is not None and y is not None and x > y)
    b_gana = sum(1 for x, y in zip(a, b) if x is not None and y is not None and y > x)
    return {"a_gana": a_gana, "b_gana": b_gana, "p": _binom_dos_colas(a_gana, a_gana + b_gana)}


def holm(pvalores: List[float]) -> List[float]:
    m = len(pvalores)
    orden = sorted(range(m), key=lambda i: pvalores[i])
    ajust, maximo = [0.0] * m, 0.0
    for r, i in enumerate(orden):
        maximo = max(maximo, min(1.0, (m - r) * pvalores[i]))
        ajust[i] = maximo
    return ajust


def tamano_muestra(p: float = 0.8, semiancho: float = 0.05) -> int:
    """Casos necesarios para que el IC 95 % de una proporcion p tenga +/- semiancho."""
    return int(math.ceil(Z95 ** 2 * p * (1 - p) / semiancho ** 2))


def _p_pareado(x: Dict[str, float], y: Dict[str, float], signo: int):
    comunes = sorted(set(x) & set(y))
    r = prueba_signos([signo * x[c] for c in comunes], [signo * y[c] for c in comunes])
    return r["p"], len(comunes)


def ranking(puntajes: Dict[str, Dict[str, float]], mayor_es_mejor: bool = True, alfa: float = 0.05,
            desempate: Optional[Dict[str, float]] = None) -> List[Dict]:
    """Ordena modelos por la media de su metrica principal y forma grupos de empate.

    puntajes:  {modelo: {caso_id: valor}}  valor 0/1 (acierto) o continuo; los casos con la
               misma clave se comparan entre modelos (diseno pareado)
    desempate: {modelo: valor}, menor es mejor (p. ej. latencia); solo ordena empates exactos
    Grupos: el mejor modelo abre el grupo A; todo modelo cuya diferencia con el no es
    significativa (prueba de signos, Holm, alfa) entra al grupo A; con los demas se repite
    (grupo B, ...). Dentro de un grupo la diferencia NO esta demostrada.
    """
    signo = 1 if mayor_es_mejor else -1
    medias = {m: float(np.mean(list(v.values()))) if v else float("nan") for m, v in puntajes.items()}
    nan_al_final = lambda x: (math.isnan(x), -signo * (0.0 if math.isnan(x) else x))
    orden = sorted(puntajes, key=lambda m: (*nan_al_final(medias[m]), (desempate or {}).get(m, 0.0)))
    mejor = orden[0] if orden else None
    p_mejor = {}
    if mejor:
        otros = orden[1:]
        crudos = [_p_pareado(puntajes[mejor], puntajes[m], signo) for m in otros]
        for m, pa, (_, n) in zip(otros, holm([p for p, _ in crudos]), crudos):
            p_mejor[m] = (pa, n)
    filas, restantes, letra = [], list(orden), 0
    while restantes:
        lider, resto = restantes[0], restantes[1:]
        crudos = [_p_pareado(puntajes[lider], puntajes[m], signo) for m in resto]
        ajust = holm([p for p, _ in crudos]) if crudos else []
        grupo = chr(ord("A") + min(letra, 25))
        miembros = [lider] + [m for m, pa in zip(resto, ajust) if pa >= alfa]
        for m in miembros:
            pa, n = p_mejor.get(m, (None, len(puntajes[m])))
            filas.append({"modelo": m, "media": medias[m], "grupo": grupo, "p_vs_mejor": pa, "n_pareados": n})
        restantes = [m for m in resto if m not in miembros]
        letra += 1
    for f in filas:                       # IC 95 % de la media por caso (bootstrap)
        b = bootstrap(list(puntajes[f["modelo"]].values()), b=1000)
        f["ic_bajo"], f["ic_alto"] = b["ic_bajo"], b["ic_alto"]
    filas.sort(key=lambda f: orden.index(f["modelo"]))
    for i, f in enumerate(filas):
        f["lugar"] = i + 1
    return filas
