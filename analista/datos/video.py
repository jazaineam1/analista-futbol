"""Resultados del análisis de video (carpeta de `analista analizar`) en el esquema común."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def _csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta) if ruta.exists() and ruta.stat().st_size > 5 else pd.DataFrame()


def cargar(carpeta: str | Path) -> dict:
    carpeta = Path(carpeta)
    if not (carpeta / "tracking.csv").exists():
        raise FileNotFoundError(f"{carpeta} no tiene tracking.csv: ¿es una carpeta de resultados de 'analista analizar'?")
    tr = _csv(carpeta / "tracking.csv")
    ba = _csv(carpeta / "balon.csv")
    cam = _csv(carpeta / "camara.csv")
    resumen = json.loads((carpeta / "resumen_proceso.json").read_text(encoding="utf-8")) \
        if (carpeta / "resumen_proceso.json").exists() else {}
    nombres = resumen.get("nombres_equipos", {"A": "Equipo A", "B": "Equipo B"})
    colores = resumen.get("colores_kit") or resumen.get("colores_equipos") or {}
    info = {"fuente": "video", "carpeta": str(carpeta),
            "equipos": {k: {"nombre": nombres.get(k, f"Equipo {k}"), "color": colores.get(k, "#888888")} for k in ("A", "B")},
            "cancha": resumen.get("cancha", {"largo": 105, "ancho": 68}),
            "fps": (resumen.get("fps_video") or 25) / max(1, resumen.get("paso", 1)),
            "camara_fija": resumen.get("camara_fija", False), "proceso": resumen}
    return {"tracking": tr, "balon": ba, "camara": cam, "info": info}
