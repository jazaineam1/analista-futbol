"""Después de procesar un video: métricas, tablas y reporte HTML (lo usan la CLI y el panel)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analisis.tracking import ParametrosMetricas, calcular_todo
from .cancha import Cancha
from .reportes.video import generar_reporte


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta) if ruta.exists() and ruta.stat().st_size > 5 else pd.DataFrame()


def postprocesar(salida: Path, cancha: Cancha, resumen: dict, ataca_derecha: str | None = None,
                 colores_kit: dict | None = None, titulo: str = "Análisis del partido", nota: str = "") -> dict:
    salida = Path(salida)
    df = leer_csv(salida / "tracking.csv")
    balon = leer_csv(salida / "balon.csv")
    cam = leer_csv(salida / "camara.csv")
    p = ParametrosMetricas(largo=cancha.largo, ancho=cancha.ancho, ataca_derecha=ataca_derecha)
    calc = calcular_todo(df, balon, cam, p)
    for nombre in ("forma", "toques", "pases", "eventos", "fisico", "poseedor"):
        tabla = calc[nombre]
        if isinstance(tabla, pd.DataFrame):
            tabla.to_csv(salida / f"{nombre}.csv", index=False)
    (salida / "metricas.json").write_text(json.dumps(calc["resumen"], indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    if colores_kit:
        resumen["colores_kit"] = colores_kit
    resumen["titulo"], resumen["nota"] = titulo, nota          # para rehacer el reporte con `analista reporte`
    (salida / "resumen_proceso.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    ruta = generar_reporte(salida, df, calc, resumen, cancha, titulo, nota)
    return {"reporte": str(ruta), "resumen": calc["resumen"]}
