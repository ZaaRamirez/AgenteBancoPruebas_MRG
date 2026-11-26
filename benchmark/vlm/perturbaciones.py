"""Perturbaciones controladas de la imagen (robustez del VLM). Se aplican al vuelo y de
forma determinista; la etiqueta del caso no cambia, asi que una respuesta distinta a la
de la imagen original es una falta de robustez."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter

DESCRIPCION = {
    "original": "imagen sin cambios",
    "brillo_bajo": "brillo al 55 % (poca luz)",
    "brillo_alto": "brillo al 150 % (sobreexposicion)",
    "contraste_bajo": "contraste al 50 %",
    "desenfoque": "desenfoque gaussiano radio 2 px (camara fuera de foco o movimiento)",
    "ruido": "ruido gaussiano sigma 12 (sensor)",
    "jpeg_bajo": "compresion JPEG calidad 15 (transmision)",
    "recorte": "recorte central al 85 % y reescalado (encuadre distinto)",
    "media_resolucion": "reduccion a la mitad y regreso al tamano (baja resolucion)",
}


def aplicar(img: np.ndarray, nombre: str, semilla: int = 0) -> np.ndarray:
    if nombre == "original":
        return img
    im = Image.fromarray(img)
    w, h = im.size
    if nombre == "brillo_bajo":
        im = ImageEnhance.Brightness(im).enhance(0.55)
    elif nombre == "brillo_alto":
        im = ImageEnhance.Brightness(im).enhance(1.5)
    elif nombre == "contraste_bajo":
        im = ImageEnhance.Contrast(im).enhance(0.5)
    elif nombre == "desenfoque":
        im = im.filter(ImageFilter.GaussianBlur(2))
    elif nombre == "ruido":
        rng = np.random.default_rng(semilla)
        a = np.asarray(im).astype(np.float32) + rng.normal(0, 12, (h, w, 3))
        return a.clip(0, 255).astype(np.uint8)
    elif nombre == "jpeg_bajo":
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=15)
        im = Image.open(io.BytesIO(buf.getvalue())).convert("RGB")
    elif nombre == "recorte":
        dx, dy = int(w * 0.075), int(h * 0.075)
        im = im.crop((dx, dy, w - dx, h - dy)).resize((w, h))
    elif nombre == "media_resolucion":
        im = im.resize((max(1, w // 2), max(1, h // 2))).resize((w, h))
    else:
        raise ValueError(f"Perturbacion desconocida: {nombre}. Opciones: {list(DESCRIPCION)}")
    return np.asarray(im.convert("RGB"))
