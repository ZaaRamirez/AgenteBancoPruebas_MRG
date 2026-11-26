"""Conversion de imagenes (sin cv_bridge)."""
from __future__ import annotations

import base64
import io

import numpy as np


def ros_a_rgb(msg) -> np.ndarray:
    """sensor_msgs/Image -> arreglo RGB (H, W, 3) uint8."""
    enc = msg.encoding.lower()
    raw = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    h, w = msg.height, msg.width
    if enc in ("rgb8", "rgb"):
        return raw.reshape(h, w, 3).copy()
    if enc in ("bgr8", "bgr"):
        return raw.reshape(h, w, 3)[:, :, ::-1].copy()
    if enc in ("rgba8", "rgba"):
        return raw.reshape(h, w, 4)[:, :, :3].copy()
    if enc in ("bgra8", "bgra"):
        return raw.reshape(h, w, 4)[:, :, 2::-1].copy()
    if enc in ("mono8", "8uc1"):
        g = raw.reshape(h, w)
        return np.stack([g, g, g], axis=2)
    raise ValueError(f"Codificacion de imagen no soportada: {msg.encoding}")


def png_base64(img: np.ndarray) -> str:
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def guardar_png(img: np.ndarray, archivo: str) -> None:
    from PIL import Image
    Image.fromarray(img).save(archivo)


def leer_rgb(archivo: str) -> np.ndarray:
    from PIL import Image
    return np.array(Image.open(archivo).convert("RGB"))
