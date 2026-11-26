"""Modelos de la capa de EJECUCION (VLA)."""
from __future__ import annotations

import importlib
import json
import os
import random

import numpy as np

from ..interfaces import PoliticaVLA
from ..tipos import Observacion


class LeRobotVLA(PoliticaVLA):
    """Politica de LeRobot cargada desde un checkpoint (por omision MolmoAct2).
    Reproduce molmo_ros_bridge.py: from_pretrained + make_pre_post_processors +
    select_action(..., inference_action_mode="continuous").
    Requiere el entorno de Python donde ya estan torch y su fork de LeRobot."""

    def __init__(self, checkpoint: str,
                 policy_class: str = "lerobot.policies.molmoact2.modeling_molmoact2:MolmoAct2Policy",
                 device: str = "auto", camaras: dict = None, state_key: str = "observation.state",
                 select_action_kwargs=None, **_):
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        if not os.path.isdir(os.path.expanduser(checkpoint)):
            raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint} (revisa config/rutas.env)")
        checkpoint = os.path.expanduser(checkpoint)
        mod, cls = policy_class.split(":")
        Politica = getattr(importlib.import_module(mod), cls)
        self.torch, self.nombre = torch, checkpoint
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device == "auto" else torch.device(device)
        self.policy = Politica.from_pretrained(checkpoint)
        if hasattr(self.policy, "init_rtc_processor"):
            self.policy.init_rtc_processor()
        self.policy.eval().to(self.device)
        self.policy.reset()
        self.pre, self.post = make_pre_post_processors(policy_cfg=self.policy.config, pretrained_path=checkpoint)
        # clave de LeRobot -> nombre de camara del entorno
        self.camaras = camaras or {"observation.images.front": "front"}
        self.state_key = state_key
        k = select_action_kwargs or {"inference_action_mode": "continuous"}
        self.kwargs = json.loads(k) if isinstance(k, str) else dict(k)

    def reiniciar(self):
        self.policy.reset()

    def predecir(self, obs: Observacion, instruccion: str) -> np.ndarray:
        t = self.torch
        lote = {"task": [instruccion],
                self.state_key: t.tensor(obs.estado, dtype=t.float32).unsqueeze(0)}
        for clave, cam in self.camaras.items():
            img = obs.imagen(cam)
            if img is None:
                raise RuntimeError(f"La observacion no trae la camara '{cam}'")
            lote[clave] = (t.from_numpy(img).permute(2, 0, 1).float() / 255.0).unsqueeze(0)
        x = self.pre(lote)
        x = {k: v.to(self.device) if isinstance(v, t.Tensor) else v for k, v in x.items()}
        with t.inference_mode():
            a = self.policy.select_action(x, **self.kwargs)
        try:
            r = self.post(a)
            a = (r.actions if hasattr(r, "actions") else r)
        except Exception:
            pass
        a = a.detach().cpu().numpy()
        return a[0] if a.ndim > 1 else a


class PosturaVLA(PoliticaVLA):
    """Diagnostico: siempre predice la misma postura (unidades del conjunto de datos).
    Comprueba topicos, nombres de articulacion, unidades, gripper y detector de fin
    con el gemelo MuJoCo o el robot antes de cargar un VLA real."""

    def __init__(self, checkpoint: str = "postura", objetivo=None, **_):
        self.nombre = checkpoint or "postura"
        o = objetivo if objetivo is not None else [0, 0, 0, 0, 0, 40]
        self.objetivo = np.asarray(json.loads(o) if isinstance(o, str) else o, dtype=np.float32)

    def predecir(self, obs, instruccion):
        return self.objetivo.copy()


class SimuladoVLA(PoliticaVLA):
    """Ejecucion simulada para probar la arquitectura sin GPU: en cada subtarea se
    mueve a una postura aleatoria (el entorno la alcanza y el detector declara fin).
    Con probabilidad 1 - tasa_exito la politica "titubea" y nunca se estabiliza, de
    modo que la subtarea termina por tiempo (fallo de ejecucion)."""

    def __init__(self, checkpoint: str = "simulado", tasa_exito: float = 1.0, semilla: int = 0, **_):
        self.nombre = checkpoint or "simulado"
        self.tasa = float(tasa_exito)
        self.semilla = semilla
        self.reiniciar_semilla()
        self.objetivo, self.falla = None, False

    def reiniciar_semilla(self):
        self.rng = random.Random(self.semilla)
        self.np_rng = np.random.default_rng(self.semilla)

    def reiniciar(self):
        self.objetivo, self.falla = None, self.rng.random() >= self.tasa

    def predecir(self, obs, instruccion):
        dim = len(obs.estado) if obs.estado is not None else 6
        if self.objetivo is None:
            self.objetivo = self.np_rng.uniform(-30, 30, dim).astype(np.float32)
        if self.falla:
            return self.objetivo + self.np_rng.uniform(-20, 20, dim).astype(np.float32)
        return self.objetivo.copy()
