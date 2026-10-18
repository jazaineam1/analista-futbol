"""Modelo de goles esperados (xG) propio.

Regresión logística entrenada con los tiros de StatsBomb Open Data (ver
`demo/entrenar_xg.py`). Hay dos versiones:

- **basico**: distancia, ángulo, cabeza, tiro libre directo. Sirve con cualquier fuente que
  tenga la posición del tiro (por ejemplo, un remate detectado en el video).
- **contexto**: además, defensores en la línea de tiro, distancia del arquero, primer toque,
  balón parado y presión. Necesita las posiciones de los demás jugadores en el momento del
  tiro (el "freeze frame" de StatsBomb o el tracking del video).

Los penales tienen un xG fijo (la tasa histórica de conversión).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

RUTA_MODELO = Path(__file__).resolve().parents[1] / "modelos" / "xg.json"
XG_PENAL = 0.78
ANCHO_ARCO = 7.32


def geometria(x, y, largo: float = 105.0):
    """Distancia (m) al centro del arco rival y ángulo de apertura (rad) entre los postes.

    x, y en metros, origen en el centro, atacando hacia +x.
    """
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    gx = largo / 2
    dx = np.maximum(gx - x, 0.01)
    dist = np.hypot(dx, y)
    a1 = np.arctan2(y - ANCHO_ARCO / 2, dx)
    a2 = np.arctan2(y + ANCHO_ARCO / 2, dx)
    ang = np.abs(a2 - a1)
    return dist, ang


def _en_triangulo(px, py, ax, ay, bx, by, cx, cy):
    def lado(x1, y1, x2, y2, x3, y3):
        return (x1 - x3) * (y2 - y3) - (x2 - x3) * (y1 - y3)
    d1, d2, d3 = lado(px, py, ax, ay, bx, by), lado(px, py, bx, by, cx, cy), lado(px, py, cx, cy, ax, ay)
    neg = (d1 < 0) | (d2 < 0) | (d3 < 0)
    pos = (d1 > 0) | (d2 > 0) | (d3 > 0)
    return ~(neg & pos)


def contexto_tiro(x: float, y: float, rivales: list[tuple[float, float]], arquero: tuple[float, float] | None,
                  largo: float = 105.0) -> dict:
    """Rasgos de contexto a partir de las posiciones de los rivales en el momento del tiro."""
    gx = largo / 2
    n = 0
    for rx, ry in rivales:
        if _en_triangulo(rx, ry, x, y, gx, ANCHO_ARCO / 2, gx, -ANCHO_ARCO / 2):
            n += 1
    if arquero is not None:
        dist_arq = float(np.hypot(gx - arquero[0], arquero[1]))
    else:
        dist_arq = 1.0      # valor típico
    return {"defensores_en_cono": n, "dist_arquero": dist_arq}


def rasgos(tiros: pd.DataFrame, largo: float = 105.0) -> pd.DataFrame:
    """Rasgos del modelo a partir de tiros en el esquema común (ver `datos.esquema`)."""
    d, a = geometria(tiros["x"].to_numpy(), tiros["y"].to_numpy(), largo)
    cuerpo = tiros.get("cuerpo", pd.Series([None] * len(tiros), index=tiros.index)).fillna("")
    subtipo = tiros.get("subtipo", pd.Series([None] * len(tiros), index=tiros.index)).fillna("")
    jugada = tiros.get("jugada", pd.Series([None] * len(tiros), index=tiros.index)).fillna("")
    r = pd.DataFrame({
        "log_distancia": np.log(np.maximum(d, 0.5)),
        "angulo": a,
        "cabeza": cuerpo.str.contains("Head|cabeza", case=False).astype(float).to_numpy(),
        "tiro_libre": (subtipo == "Free Kick").astype(float).to_numpy(),
        "balon_parado": jugada.isin(["From Corner", "From Free Kick", "From Throw In"]).astype(float).to_numpy(),
        "bajo_presion": tiros.get("bajo_presion", pd.Series(False, index=tiros.index)).fillna(False).astype(float).to_numpy(),
    }, index=tiros.index)
    primer, conos, darq = [], [], []
    for c, (x, y) in zip(tiros.get("contexto", pd.Series([None] * len(tiros), index=tiros.index)),
                         tiros[["x", "y"]].to_numpy()):
        c = c if isinstance(c, dict) else {}
        primer.append(float(bool(c.get("primer_toque", False))))
        cong = c.get("congelado")
        if cong:
            rivales = [(p["x"], p["y"]) for p in cong if not p["companero"] and not p.get("arquero")]
            arq = next(((p["x"], p["y"]) for p in cong if not p["companero"] and p.get("arquero")), None)
            ctx = contexto_tiro(x, y, rivales, arq, largo)
        else:
            ctx = {"defensores_en_cono": c.get("defensores_en_cono", 1), "dist_arquero": c.get("dist_arquero", 1.0)}
        conos.append(ctx["defensores_en_cono"])
        darq.append(ctx["dist_arquero"])
    r["primer_toque"] = primer
    r["defensores_en_cono"] = np.minimum(conos, 4)
    r["log_dist_arquero"] = np.log1p(np.minimum(darq, 30))
    return r


@lru_cache(maxsize=1)
def cargar_modelo(ruta: str | None = None) -> dict:
    return json.loads(Path(ruta or RUTA_MODELO).read_text(encoding="utf-8"))


def predecir(tiros: pd.DataFrame, version: str = "auto", largo: float = 105.0) -> np.ndarray:
    """xG de cada tiro. version: 'basico', 'contexto' o 'auto' (contexto si hay posiciones)."""
    if len(tiros) == 0:
        return np.zeros(0)
    modelo = cargar_modelo()
    r = rasgos(tiros, largo)
    tiene_ctx = tiros.get("contexto", pd.Series([None] * len(tiros), index=tiros.index)).map(
        lambda c: isinstance(c, dict) and bool(c.get("congelado") or "defensores_en_cono" in c)).to_numpy()
    out = np.zeros(len(tiros))
    for v in ("basico", "contexto"):
        m = modelo["modelos"][v]
        X = r[m["rasgos"]].to_numpy(float)
        z = m["intercepto"] + X @ np.asarray(m["coeficientes"], float)
        p = 1 / (1 + np.exp(-z))
        if version == v:
            out = p
        elif version == "auto":
            usar = tiene_ctx if v == "contexto" else ~tiene_ctx
            out = np.where(usar, p, out)
    subtipo = tiros.get("subtipo", pd.Series([None] * len(tiros), index=tiros.index))
    out = np.where(subtipo.to_numpy() == "Penalty", XG_PENAL, out)
    return out


def xg_punto(x: float, y: float, cabeza: bool = False, largo: float = 105.0) -> float:
    """xG básico de un tiro en (x, y) metros."""
    t = pd.DataFrame({"x": [x], "y": [y], "cuerpo": ["Head" if cabeza else "Foot"]})
    return float(predecir(t, "basico", largo)[0])
