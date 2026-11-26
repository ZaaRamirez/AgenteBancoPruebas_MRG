"""Lectura de episodios grabados (conjunto de datos de LeRobot o carpeta .npz).

Lo usan el benchmark VLM (fotogramas inicial y final de cada episodio) y el VLA fuera
de linea (todos los fotogramas, comparando la accion predicha con la grabada).

Fuentes:
  lerobot:<repo_id>[@<carpeta_local>]    p. ej. lerobot:arrg-unam/omx_pick_hidroferol
  npz:<carpeta>                          ep_0000.npz, ... con arrays
                                         imagenes (T,H,W,3) uint8, estado (T,D), accion (T,D)
                                         y la cadena tarea  (formato para pruebas o exportaciones)
"""
from __future__ import annotations

import glob
import os
from typing import Dict, Iterator, List, Optional

import numpy as np


class Fotograma(dict):
    """{"imagen": HxWx3 uint8, "estado": (D,), "accion": (D,), "tarea": str, "episodio": int, "indice": int}"""


class Fuente:
    def episodios(self) -> List[int]:
        raise NotImplementedError

    def longitud(self, ep: int) -> int:
        raise NotImplementedError

    def fotograma(self, ep: int, i: int) -> Fotograma:
        raise NotImplementedError

    def recorrer(self, ep: int, paso: int = 1) -> Iterator[Fotograma]:
        for i in range(0, self.longitud(ep), max(1, paso)):
            yield self.fotograma(ep, i)


class FuenteNPZ(Fuente):
    def __init__(self, carpeta: str):
        self.archivos = sorted(glob.glob(os.path.join(os.path.expanduser(carpeta), "ep_*.npz")))
        if not self.archivos:
            raise FileNotFoundError(f"No hay ep_*.npz en {carpeta}")
        self._cache: Dict[int, Dict] = {}

    def _ep(self, ep):
        if ep not in self._cache:
            self._cache = {ep: dict(np.load(self.archivos[ep], allow_pickle=False))}
        return self._cache[ep]

    def episodios(self):
        return list(range(len(self.archivos)))

    def longitud(self, ep):
        return int(self._ep(ep)["imagenes"].shape[0])

    def fotograma(self, ep, i):
        d = self._ep(ep)
        i = i if i >= 0 else self.longitud(ep) + i
        return Fotograma(imagen=d["imagenes"][i], estado=d["estado"][i].astype(np.float32),
                         accion=d["accion"][i].astype(np.float32), tarea=str(d["tarea"]), episodio=ep, indice=i)


class FuenteLeRobot(Fuente):
    def __init__(self, repo_id: str, raiz: Optional[str] = None, camara: str = "observation.images.front",
                 clave_estado: str = "observation.state", clave_accion: str = "action"):
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        self.ds = LeRobotDataset(repo_id, root=os.path.expanduser(raiz) if raiz else None)
        self.camara, self.k_estado, self.k_accion = camara, clave_estado, clave_accion
        self.rangos: Dict[int, tuple] = {}
        try:      # formato v3: indices de cada episodio en los metadatos
            m = self.ds.meta.episodes
            for ep, a, b in zip(m["episode_index"], m["dataset_from_index"], m["dataset_to_index"]):
                self.rangos[int(ep)] = (int(a), int(b))
        except Exception:
            col = self.ds.hf_dataset["episode_index"]
            eps = np.asarray([int(x) for x in col])
            for ep in np.unique(eps):
                idx = np.where(eps == ep)[0]
                self.rangos[int(ep)] = (int(idx.min()), int(idx.max()) + 1)

    def episodios(self):
        return sorted(self.rangos)

    def longitud(self, ep):
        a, b = self.rangos[ep]
        return b - a

    def fotograma(self, ep, i):
        a, b = self.rangos[ep]
        i = i if i >= 0 else (b - a) + i
        s = self.ds[a + i]
        img = s[self.camara]
        img = img.permute(1, 2, 0).numpy() if hasattr(img, "permute") else np.asarray(img)
        img = (img * 255).clip(0, 255).astype(np.uint8) if img.dtype != np.uint8 else img
        t = lambda k: np.asarray(s[k].numpy() if hasattr(s[k], "numpy") else s[k], dtype=np.float32)
        return Fotograma(imagen=img, estado=t(self.k_estado), accion=t(self.k_accion),
                         tarea=str(s.get("task", "")), episodio=ep, indice=i)


def abrir(especificacion: str, **kw) -> Fuente:
    tipo, _, resto = especificacion.partition(":")
    if tipo == "npz":
        return FuenteNPZ(resto)
    if tipo == "lerobot":
        repo, _, raiz = resto.partition("@")
        return FuenteLeRobot(repo, raiz or None, **kw)
    raise ValueError(f"Fuente desconocida '{especificacion}': usa lerobot:<repo>[@carpeta] o npz:<carpeta>")


def seleccionar(eps: List[int], seleccion) -> List[int]:
    """seleccion: None/todos | "0-49" | [0, 3, 7] | {"ultimos": 0.2} (episodios de prueba)."""
    if seleccion in (None, "", "todos"):
        return list(eps)
    if isinstance(seleccion, dict) and "ultimos" in seleccion:
        k = max(1, int(round(len(eps) * float(seleccion["ultimos"]))))
        return list(eps)[-k:]
    if isinstance(seleccion, str):
        out = []
        for parte in seleccion.split(","):
            a, _, b = parte.partition("-")
            out += list(range(int(a), int(b) + 1)) if b else [int(a)]
        return [e for e in eps if e in set(out)]
    return [e for e in eps if e in set(seleccion)]
