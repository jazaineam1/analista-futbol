"""Utilidades comunes."""
from __future__ import annotations

import math


def json_limpio(o):
    """Datos listos para JSON estándar: tipos de numpy/pandas a Python y NaN/infinito a null
    (un navegador no acepta NaN en un JSON y fallaría la lectura completa)."""
    if isinstance(o, dict):
        return {str(k): json_limpio(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [json_limpio(v) for v in o]
    if hasattr(o, "item") and not isinstance(o, (str, bytes)):
        try:
            o = o.item()
        except (ValueError, AttributeError):
            return str(o)
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if o is None or isinstance(o, (str, int, float, bool)):
        return o
    return str(o)
