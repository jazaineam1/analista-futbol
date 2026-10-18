"""StatsBomb Open Data (gratis): competiciones, partidos, alineaciones y eventos.

Datos: https://github.com/statsbomb/open-data — uso para investigación y educación,
citando a Hudl StatsBomb como fuente.

Las coordenadas de StatsBomb (120 x 80, origen arriba a la izquierda, cada equipo atacando
hacia la derecha) se convierten a metros con origen en el centro (ver `esquema`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..cancha import Cancha
from .cache import carpeta_cache, json_remoto
from .esquema import TIPOS

BASE = "https://raw.githubusercontent.com/statsbomb/open-data/master/data"


def _cache(*partes) -> "Path":  # noqa: F821
    return carpeta_cache().joinpath("statsbomb", *partes)


def competiciones() -> pd.DataFrame:
    d = json_remoto(f"{BASE}/competitions.json", _cache("competitions.json"))
    df = pd.DataFrame(d)
    df = df.rename(columns={"competition_id": "competicion_id", "season_id": "temporada_id",
                            "competition_name": "competicion", "season_name": "temporada",
                            "country_name": "pais", "competition_gender": "genero"})
    return df[["competicion_id", "temporada_id", "competicion", "temporada", "pais", "genero"]]


def partidos(competicion_id: int, temporada_id: int) -> pd.DataFrame:
    d = json_remoto(f"{BASE}/matches/{competicion_id}/{temporada_id}.json",
                    _cache("matches", f"{competicion_id}_{temporada_id}.json"))
    filas = []
    for m in d:
        filas.append({"partido_id": m["match_id"], "fecha": m.get("match_date"),
                      "local": m["home_team"]["home_team_name"], "visitante": m["away_team"]["away_team_name"],
                      "goles_local": m.get("home_score"), "goles_visitante": m.get("away_score"),
                      "fase": (m.get("competition_stage") or {}).get("name"),
                      "estadio": (m.get("stadium") or {}).get("name"),
                      "competicion": m["competition"]["competition_name"], "temporada": m["season"]["season_name"]})
    return pd.DataFrame(filas).sort_values("fecha").reset_index(drop=True)


def eventos_crudos(partido_id: int) -> list[dict]:
    return json_remoto(f"{BASE}/events/{partido_id}.json", _cache("events", f"{partido_id}.json"))


def tres_sesenta(partido_id: int) -> dict[str, dict] | None:
    """Capturas 360 (posiciones de los jugadores visibles en cada evento), por id de evento.
    Solo existen para algunas competiciones (Mundial 2022, Euro 2020 y 2024, …): None si no hay."""
    try:
        d = json_remoto(f"{BASE}/three-sixty/{partido_id}.json", _cache("three-sixty", f"{partido_id}.json"))
    except FileNotFoundError:
        return None
    return {f["event_uuid"]: f for f in d}


def alineaciones(partido_id: int) -> list[dict]:
    return json_remoto(f"{BASE}/lineups/{partido_id}.json", _cache("lineups", f"{partido_id}.json"))


def a_metros(xy, cancha: Cancha):
    """Coordenadas StatsBomb (120 x 80) -> metros con origen en el centro, y hacia arriba."""
    if xy is None or (isinstance(xy, float) and np.isnan(xy)):
        return np.nan, np.nan
    x, y = xy[0], xy[1]
    return x / 120.0 * cancha.largo - cancha.L2, cancha.W2 - y / 80.0 * cancha.ancho


def _nombre(d, *claves):
    for c in claves:
        if not isinstance(d, dict):
            return None
        d = d.get(c)
    return d


def apodos(partido_id: int) -> dict[str, str]:
    """Nombre completo -> nombre conocido ('Lionel Andrés Messi Cuccittini' -> 'Lionel Messi')."""
    out = {}
    try:
        for eq in alineaciones(partido_id):
            for j in eq.get("lineup", []):
                out[j["player_name"]] = j.get("player_nickname") or j["player_name"]
    except Exception:  # noqa: BLE001
        pass
    return out


def eventos(partido_id: int | None = None, crudos: list[dict] | None = None, cancha: Cancha | None = None,
            local: str | None = None, nombres_cortos: bool = False) -> pd.DataFrame:
    """Eventos de un partido en el esquema común (ver `analista.datos.esquema`)."""
    cancha = cancha or Cancha()
    crudos = crudos if crudos is not None else eventos_crudos(partido_id)
    if local is None:
        xi = [e for e in crudos if e["type"]["name"] == "Starting XI"]
        local = xi[0]["team"]["name"] if xi else None
    filas = []
    for e in crudos:
        tipo_o = e["type"]["name"]
        ts = e.get("timestamp", "00:00:00.000")
        frac = float(ts.split(":")[-1]) % 1 if ts else 0.0
        x, y = a_metros(e.get("location"), cancha)
        f = {"id": e["id"], "indice": e.get("index"), "periodo": e["period"], "minuto": e["minute"],
             "segundo": e["second"], "t": e["minute"] * 60 + e["second"] + frac,
             "equipo_nombre": _nombre(e, "team", "name"),
             "equipo": "A" if _nombre(e, "team", "name") == local else "B",
             "jugador": _nombre(e, "player", "name"), "jugador_id": _nombre(e, "player", "id"),
             "posicion": _nombre(e, "position", "name"), "tipo": TIPOS.get(tipo_o, tipo_o.lower()),
             "tipo_original": tipo_o, "subtipo": None, "x": x, "y": y, "x_fin": np.nan, "y_fin": np.nan,
             "exito": None, "resultado": None, "xg": np.nan, "cuerpo": None,
             "bajo_presion": bool(e.get("under_pressure", False)), "receptor": None,
             "jugada": _nombre(e, "play_pattern", "name"), "posesion": e.get("possession"),
             "equipo_posesion": _nombre(e, "possession_team", "name"), "duracion": e.get("duration"),
             "contexto": None}
        if tipo_o == "Pass":
            p = e.get("pass", {})
            f["x_fin"], f["y_fin"] = a_metros(p.get("end_location"), cancha)
            f["resultado"] = _nombre(p, "outcome", "name") or "Completo"
            f["exito"] = _nombre(p, "outcome", "name") is None
            f["subtipo"] = _nombre(p, "type", "name")
            f["cuerpo"] = _nombre(p, "body_part", "name")
            f["receptor"] = _nombre(p, "recipient", "name")
            f["contexto"] = {"cross": p.get("cross", False), "alto": _nombre(p, "height", "name"),
                             "asistencia_tiro": p.get("shot_assist", False), "asistencia_gol": p.get("goal_assist", False)}
        elif tipo_o == "Shot":
            s = e.get("shot", {})
            xf, yf = a_metros(s.get("end_location"), cancha)
            f["x_fin"], f["y_fin"] = xf, yf
            f["resultado"] = _nombre(s, "outcome", "name")
            f["exito"] = f["resultado"] == "Goal"
            f["subtipo"] = _nombre(s, "type", "name")
            f["cuerpo"] = _nombre(s, "body_part", "name")
            f["xg"] = s.get("statsbomb_xg", np.nan)
            ff = s.get("freeze_frame") or []
            f["contexto"] = {"primer_toque": s.get("first_time", False), "tecnica": _nombre(s, "technique", "name"),
                             "congelado": [{"x": a_metros(p["location"], cancha)[0], "y": a_metros(p["location"], cancha)[1],
                                            "companero": p["teammate"], "arquero": _nombre(p, "position", "name") == "Goalkeeper"}
                                           for p in ff]}
        elif tipo_o == "Carry":
            f["x_fin"], f["y_fin"] = a_metros(_nombre(e, "carry", "end_location"), cancha)
            f["exito"] = True
        elif tipo_o == "Duel":
            f["subtipo"] = _nombre(e, "duel", "type", "name")
            f["resultado"] = _nombre(e, "duel", "outcome", "name")
            f["exito"] = f["resultado"] in ("Won", "Success", "Success In Play", "Success Out")
        elif tipo_o == "Dribble":
            f["resultado"] = _nombre(e, "dribble", "outcome", "name")
            f["exito"] = f["resultado"] == "Complete"
        elif tipo_o == "Interception":
            f["resultado"] = _nombre(e, "interception", "outcome", "name")
            f["exito"] = f["resultado"] in ("Won", "Success", "Success In Play", "Success Out")
        elif tipo_o == "Ball Recovery":
            f["exito"] = not bool(_nombre(e, "ball_recovery", "recovery_failure"))
        elif tipo_o == "Goal Keeper":
            f["subtipo"] = _nombre(e, "goalkeeper", "type", "name")
            f["resultado"] = _nombre(e, "goalkeeper", "outcome", "name")
        elif tipo_o == "Foul Committed":
            fc = e.get("foul_committed", {}) or {}
            f["subtipo"] = _nombre(fc, "type", "name")
            f["contexto"] = {"penal": bool(fc.get("penalty", False)), "tarjeta": _nombre(fc, "card", "name"),
                             "ventaja": bool(fc.get("advantage", False))}
        elif tipo_o == "Foul Won":
            fw = e.get("foul_won", {}) or {}
            f["contexto"] = {"penal": bool(fw.get("penalty", False))}
        elif tipo_o == "Substitution":
            f["receptor"] = _nombre(e, "substitution", "replacement", "name")
        filas.append(f)
    df = pd.DataFrame(filas)
    if len(df):
        df = df.sort_values(["periodo", "indice"]).reset_index(drop=True)
        if nombres_cortos and partido_id is not None:
            ap = apodos(partido_id)
            if ap:
                df["jugador"] = df["jugador"].map(lambda n: ap.get(n, n) if isinstance(n, str) else n)
                df["receptor"] = df["receptor"].map(lambda n: ap.get(n, n) if isinstance(n, str) else n)
    return df


def info_partido(partido_id: int, competicion_id: int | None = None, temporada_id: int | None = None) -> dict:
    """Metadatos del partido (busca en los partidos de la competición si se indican los ids)."""
    if competicion_id is not None and temporada_id is not None:
        p = partidos(competicion_id, temporada_id)
        fila = p[p["partido_id"] == partido_id]
        if len(fila):
            return fila.iloc[0].to_dict()
    return {"partido_id": partido_id}


def buscar_partido(partido_id: int) -> dict | None:
    """Recorre las competiciones abiertas hasta encontrar el partido (usa la caché)."""
    for r in competiciones().itertuples():
        try:
            p = partidos(r.competicion_id, r.temporada_id)
        except Exception:  # noqa: BLE001
            continue
        fila = p[p["partido_id"] == partido_id]
        if len(fila):
            d = fila.iloc[0].to_dict()
            d.update({"competicion_id": r.competicion_id, "temporada_id": r.temporada_id})
            return d
    return None
