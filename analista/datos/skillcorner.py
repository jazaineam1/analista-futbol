"""SkillCorner Open Data: tracking de 22 jugadores y balón (10 Hz) extraído de video de TV.

Datos: https://github.com/skillcorner/opendata (licencia MIT). 20 partidos de la A-League 2024/25.
El tracking de cada partido pesa ~90 MB y se descarga una sola vez (queda en la caché).
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .cache import carpeta_cache, descargar, json_remoto

RAW = "https://raw.githubusercontent.com/skillcorner/opendata/master/data"
LFS = "https://media.githubusercontent.com/media/skillcorner/opendata/master/data"


def _cache(*partes):
    return carpeta_cache().joinpath("skillcorner", *partes)


def partidos() -> pd.DataFrame:
    d = json_remoto(f"{RAW}/matches.json", _cache("matches.json"))
    filas = [{"partido_id": m["id"], "fecha": (m.get("date_time") or "")[:10],
              "local": m["home_team"]["short_name"], "visitante": m["away_team"]["short_name"]} for m in d]
    return pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)


def metadatos(partido_id: int) -> dict:
    return json_remoto(f"{RAW}/matches/{partido_id}/{partido_id}_match.json", _cache(str(partido_id), "match.json"))


def eventos_dinamicos(partido_id: int) -> pd.DataFrame:
    ruta = descargar(f"{RAW}/matches/{partido_id}/{partido_id}_dynamic_events.csv",
                     _cache(str(partido_id), "dynamic_events.csv"))
    return pd.read_csv(ruta, low_memory=False)


def _leer_tracking(ruta) -> tuple[pd.DataFrame, pd.DataFrame]:
    cuadros, periodos, pid, xs, ys, det = [], [], [], [], [], []
    bal = []
    with open(ruta, encoding="utf-8") as f:
        for linea in f:
            d = json.loads(linea)
            per = d.get("period")
            if per is None:
                continue
            fr = d["frame"]
            for p in d["player_data"]:
                cuadros.append(fr)
                periodos.append(per)
                pid.append(p["player_id"])
                xs.append(p["x"])
                ys.append(p["y"])
                det.append(bool(p.get("is_detected")))
            b = d["ball_data"]
            if b.get("x") is not None:
                bal.append((fr, per, b["x"], b["y"], b.get("z") or 0.0, bool(b.get("is_detected"))))
    tr = pd.DataFrame({"cuadro": cuadros, "periodo": periodos, "id": pid, "x": xs, "y": ys, "detectado": det})
    ba = pd.DataFrame(bal, columns=["cuadro", "periodo", "x", "y", "z", "detectado"])
    return tr, ba


def cargar(partido_id: int, progreso: bool = True) -> dict:
    """Tracking, balón y metadatos en el esquema común (equipo A = local, ataca hacia +x)."""
    meta = metadatos(partido_id)
    cache_pk = _cache(str(partido_id), "tracking.pkl")
    if cache_pk.exists():
        tr, ba = pd.read_pickle(cache_pk)
    else:
        ruta = descargar(f"{LFS}/matches/{partido_id}/{partido_id}_tracking_extrapolated.jsonl",
                         _cache(str(partido_id), "tracking.jsonl"), progreso=progreso, timeout=600)
        tr, ba = _leer_tracking(ruta)
        pd.to_pickle((tr, ba), cache_pk)
    tr = tr.copy()
    ba = ba.copy()
    fps = 10.0
    local = meta["home_team"]["id"]
    jug = {p["id"]: p for p in meta["players"]}
    tr["equipo"] = tr["id"].map(lambda i: "A" if jug.get(i, {}).get("team_id") == local else "B")
    tr["rol"] = tr["id"].map(lambda i: "arquero" if (jug.get(i, {}).get("player_role") or {}).get("acronym") == "GK"
                             else "campo")
    tr["nombre"] = tr["id"].map(lambda i: jug.get(i, {}).get("short_name"))
    tr["numero"] = tr["id"].map(lambda i: jug.get(i, {}).get("number"))
    # Normalizar dirección: el local (A) ataca siempre hacia +x. Se decide con la posición del
    # arquero local en cada periodo (más robusto que confiar en la etiqueta de lado).
    for per in tr["periodo"].unique():
        arq = tr[(tr["periodo"] == per) & (tr["equipo"] == "A") & (tr["rol"] == "arquero")]
        voltear = len(arq) and arq["x"].mean() > 0
        if voltear:
            for df in (tr, ba):
                m = df["periodo"] == per
                df.loc[m, "x"] *= -1
                df.loc[m, "y"] *= -1
    tr["t"] = tr["cuadro"] / fps
    ba["t"] = ba["cuadro"] / fps
    tr = tr[["cuadro", "t", "periodo", "id", "nombre", "numero", "equipo", "rol", "x", "y", "detectado"]]
    kit = lambda k: (meta.get(k) or {}).get("jersey_color") or "#888888"  # noqa: E731
    info = {
        "fuente": "SkillCorner Open Data", "partido_id": partido_id,
        "fecha": (meta.get("date_time") or "")[:10],
        "competicion": ((meta.get("competition_edition") or {}).get("name")),
        "estadio": (meta.get("stadium") or {}).get("name"),
        "equipos": {"A": {"nombre": meta["home_team"]["name"], "corto": meta["home_team"].get("short_name"),
                          "color": kit("home_team_kit")},
                    "B": {"nombre": meta["away_team"]["name"], "corto": meta["away_team"].get("short_name"),
                          "color": kit("away_team_kit")}},
        "marcador": {"A": meta.get("home_team_score"), "B": meta.get("away_team_score")},
        "cancha": {"largo": float(meta.get("pitch_length") or 105), "ancho": float(meta.get("pitch_width") or 68)},
        "fps": fps,
        "jugadores": [{"id": p["id"], "nombre": p.get("short_name"), "numero": p.get("number"),
                       "equipo": "A" if p["team_id"] == local else "B",
                       "rol": (p.get("player_role") or {}).get("acronym")} for p in meta["players"]],
    }
    return {"tracking": tr, "balon": ba[["cuadro", "t", "periodo", "x", "y", "z", "detectado"]], "info": info}


def goles(partido_id: int, info: dict | None = None) -> pd.DataFrame:
    """Goles a partir de los eventos dinámicos (cambio de marcador entre posesiones)."""
    try:
        d = eventos_dinamicos(partido_id)
    except Exception:  # noqa: BLE001
        return pd.DataFrame(columns=["cuadro", "t", "equipo"])
    if not {"team_score", "opponent_team_score", "team_id", "frame_start"} <= set(d.columns):
        return pd.DataFrame(columns=["cuadro", "t", "equipo"])
    meta = metadatos(partido_id)
    local = meta["home_team"]["id"]
    d = d.sort_values("frame_start")
    loc = np.where(d["team_id"] == local, d["team_score"], d["opponent_team_score"])
    vis = np.where(d["team_id"] == local, d["opponent_team_score"], d["team_score"])
    remates = d[(d.get("event_type") == "player_possession") & (d.get("end_type") == "shot")] \
        if {"event_type", "end_type"} <= set(d.columns) else d.iloc[0:0]
    filas = []
    prev = (0, 0)
    for fr, a, b in zip(d["frame_start"], loc, vis):
        if (a, b) != prev:
            eq = "A" if a > prev[0] else "B"
            equipo_id = local if eq == "A" else meta["away_team"]["id"]
            # El marcador cambia en el evento siguiente al gol (a veces minutos después, tras la
            # celebración o el VAR): el gol es el último remate de ese equipo antes del cambio
            r = remates[(remates["team_id"] == equipo_id) & (remates["frame_end"] <= fr)]
            cuadro = int(r["frame_end"].iloc[-1]) if len(r) else int(fr)
            jugador = r["player_name"].iloc[-1] if len(r) and "player_name" in r else None
            filas.append({"cuadro": cuadro, "t": cuadro / 10.0, "equipo": eq, "jugador": jugador})
            prev = (a, b)
    return pd.DataFrame(filas, columns=["cuadro", "t", "equipo", "jugador"])
