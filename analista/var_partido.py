"""VAR de un `Partido`, según de dónde vengan sus datos.

    from analista import Partido
    p = Partido.skillcorner(1886347)
    rev = p.var()                       # RevisionVar
    rev.resumen(); rev.destacadas()

- Video analizado por la app: pases detectados con los toques, cámara de cada cuadro para saber
  qué zona se ve, errores medidos en la validación (cámara fija ±0,15 m, de TV ±0,55 m).
- SkillCorner: pases, receptor buscado y faltas de los eventos dinámicos; posiciones a 10 Hz.
- StatsBomb: capturas 360 en el instante del pase (solo jugadores visibles) y faltas de los eventos.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analisis import var as av


# ---------------------------------------------------------------------- tracking (video y SkillCorner)
def _goles_tracking(partido) -> list[dict]:
    g = partido.goles()
    if g is None or not len(g):
        return []
    col = "equipo" if "equipo" in g else "equipo_atacante"
    return [{"t": float(r["t"]), "equipo": r[col]} for _, r in g.iterrows() if r.get(col) in ("A", "B")]


def var_video(partido) -> tuple[av.RevisionVar, av.Posiciones]:
    from .analisis.tracking import ParametrosMetricas, poseedor_por_cuadro, secuencia_toques
    proc = partido.info.get("proceso", {})
    fps = float(partido.info.get("fps") or 25)
    p = av.parametros_para("video", fps, partido.cancha.largo, partido.cancha.ancho, bool(proc.get("camara_fija")))
    forma = proc.get("forma_proceso")
    if not forma and proc.get("ancho_proceso"):
        w = int(proc["ancho_proceso"])
        forma = [int(round(w * 9 / 16)), w]
    pm = ParametrosMetricas(largo=p.largo, ancho=p.ancho, radio_control=2.0)
    pose = poseedor_por_cuadro(partido.tracking.dropna(subset=["x", "y"]), partido.balon.dropna(subset=["x", "y"]), pm)
    toques = secuencia_toques(pose, min_dur_s=0.08) if len(pose) else pd.DataFrame()
    afinar = None
    video = proc.get("video")
    if video and Path(video).exists() and partido.camara is not None and "u" in partido.tracking:
        from .vision.var_imagen import afinar_con_cuerpo

        def afinar(revisiones):
            afinar_con_cuerpo(revisiones, partido.tracking, partido.camara, video, int(proc.get("ancho_proceso") or 1280),
                              float(proc.get("grabacion_desfase_s") or 0.0), p.k_dudoso)
    return av.revisar_tracking(partido.tracking, partido.balon, p, partido.direccion(), toques,
                               _goles_tracking(partido), partido.camara, tuple(forma) if forma else None, afinar=afinar)


def var_skillcorner(partido) -> tuple[av.RevisionVar, av.Posiciones]:
    from .datos import skillcorner as sc
    pid = partido.info["partido_id"]
    p = av.parametros_para("skillcorner", 10.0, partido.cancha.largo, partido.cancha.ancho)
    ev = sc.eventos_dinamicos(pid)
    local = sc.metadatos(pid)["home_team"]["id"]
    pp = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "pass") & ev["player_targeted_id"].notna()]
    pases = pd.DataFrame({
        "t_toque": pp["frame_end"].to_numpy(float) / 10.0,
        "t_pase": pp["frame_end"].to_numpy(float) / 10.0,
        "equipo": np.where(pp["team_id"] == local, "A", "B"),
        "pasador": pp["player_id"].to_numpy(), "receptor": pp["player_targeted_id"].astype("int64").to_numpy(),
        "resultado": pp["pass_outcome"].fillna("").to_numpy(), "evento": pp["event_id"].astype(str).to_numpy(),
    })
    sh = ev[(ev["event_type"] == "player_possession") & (ev["end_type"] == "shot")]
    tiros = [{"t": r.frame_end / 10.0, "equipo": "A" if r.team_id == local else "B", "tirador": r.player_id}
             for r in sh.itertuples()]
    rev, pos = av.revisar_tracking(partido.tracking, partido.balon, p, {"A": 1, "B": -1},
                                   goles=_goles_tracking(partido), pases=pases, tiros=tiros)
    # Faltas: posesiones que terminan con falta recibida (el poseedor es la víctima) y las que
    # terminan con el juego reanudado con tiro libre o penal a favor
    tras = ev["game_interruption_after"].fillna("")
    fl = ev[(ev["event_type"] == "player_possession") & ((ev["end_type"] == "foul_suffered") |
            tras.str.startswith("penalty_for") | tras.str.startswith("free_kick_for"))]
    vistas = []
    for r in fl.itertuples():
        t = r.frame_end / 10.0
        if any(abs(t - v) < 3.0 for v in vistas):
            continue
        ins = pos.instante(t)
        jug = ins.jugadores[ins.jugadores["id"] == r.player_id]
        if not len(jug):
            continue
        eq = "A" if r.team_id == local else "B"
        lado = 1 if eq == "A" else -1
        x, y = float(jug["x"].iloc[0]), float(jug["y"].iloc[0])
        sancion = "penal" if str(r.game_interruption_after).startswith("penalty") else "tiro libre"
        dentro, dist = av.en_area(x, y, lado, p)
        if dist < -12.0 and sancion == "penal":
            continue                                    # penal lejos del área: el evento no es la falta misma
        sig = p.error_posicion_m if bool(jug["visible"].iloc[0]) else p.error_estimado_m
        rivales = ins.jugadores[ins.jugadores["equipo"] != eq]
        infractor = None
        if len(rivales):
            infractor = rivales["id"].iloc[int(np.argmin(np.hypot(rivales["x"] - x, rivales["y"] - y)))]
        oc = av.revisar_ocasion(ins, eq, lado, x, y, p, r.player_id, infractor)
        if dist < -4.0 and oc["veredicto"] == "no es ocasión manifiesta":
            continue                                    # lejos del área y sin ocasión: no es jugada de VAR
        vistas.append(t)
        f = av.revisar_falta(x, y, lado, p, sig, sancion)
        f.update(t=round(t, 2), equipo=eq, jugador=r.player_id, cuadro=int(r.frame_end), direccion=lado,
                 receptor=r.player_id, defensor=infractor, ocasion={k: v for k, v in oc.items() if k != "jugadores"})
        _sumar_ocasion(f, oc)
        rev.faltas.append(f)
    rev.faltas.sort(key=lambda f: f["t"])
    return rev, pos


def _sumar_ocasion(f: dict, oc: dict):
    """La revisión de ocasión manifiesta se agrega al texto de la falta."""
    if oc["veredicto"] == "ocasión manifiesta":
        f["explicacion"] += " " + oc["explicacion"]
        f["tarjeta_sugerida"] = "roja" if "roja" in oc["explicacion"] and f["veredicto"] != "dentro del área" else "revisar"
    elif oc["veredicto"] == "dudoso":
        f["explicacion"] += " ¿Ocasión manifiesta? " + oc["explicacion"]


# ---------------------------------------------------------------------- StatsBomb (eventos + 360)
def _sb_xy(loc, largo=105.0, ancho=68.0):
    return (loc[0] / 120.0 - 0.5) * largo, (0.5 - loc[1] / 80.0) * ancho


def instante_360(cuadro_360: dict, equipo: str, balon_loc, largo=105.0, ancho=68.0) -> av.Instante:
    """Captura 360 como `Instante`: el equipo del evento ataca hacia +x (así viene en StatsBomb)."""
    otro = "B" if equipo == "A" else "A"
    filas = []
    for k, f in enumerate(cuadro_360.get("freeze_frame") or []):
        x, y = _sb_xy(f["location"], largo, ancho)
        filas.append({"id": "pasador" if f.get("actor") else f"{'c' if f['teammate'] else 'r'}{k}",
                      "equipo": equipo if f["teammate"] else otro, "rol": "arquero" if f.get("keeper") else "campo",
                      "x": x, "y": y, "vx": 0.0, "vy": 0.0, "visible": True})
    va = cuadro_360.get("visible_area") or []
    zona = np.array([_sb_xy((va[i], va[i + 1]), largo, ancho) for i in range(0, len(va) - 1, 2)]) if len(va) >= 6 else None
    bal = _sb_xy(balon_loc, largo, ancho) if balon_loc else None
    return av.Instante(0.0, pd.DataFrame(filas), bal, None, zona, None)


TARJETA = {"Yellow Card": "amarilla", "Red Card": "roja", "Second Yellow": "segunda amarilla"}


def _tiro_statsbomb(e: dict, crudos: list[dict], eq: str, p: av.ParametrosVar) -> dict | None:
    """Fuera de lugar en un remate con la foto del tiro de StatsBomb (jugadores con nombre): rebote
    (el siguiente del equipo en jugar el balón estaba adelantado) o interferencia con el arquero."""
    otro = "B" if eq == "A" else "A"
    filas = [{"id": (f.get("player") or {}).get("id"), "equipo": eq if f["teammate"] else otro,
              "rol": "arquero" if (f.get("position") or {}).get("name") == "Goalkeeper" else "campo",
              "x": _sb_xy(f["location"], p.largo, p.ancho)[0], "y": _sb_xy(f["location"], p.largo, p.ancho)[1],
              "vx": 0.0, "vy": 0.0, "visible": True} for f in e["shot"]["freeze_frame"]]
    nombres = {(f.get("player") or {}).get("id"): (f.get("player") or {}).get("name") for f in e["shot"]["freeze_frame"]}
    tirador = (e.get("player") or {}).get("id")
    nombres[tirador] = (e.get("player") or {}).get("name")
    filas.append({"id": tirador, "equipo": eq, "rol": "campo", "x": _sb_xy(e["location"], p.largo, p.ancho)[0],
                  "y": _sb_xy(e["location"], p.largo, p.ancho)[1], "vx": 0.0, "vy": 0.0, "visible": True})
    t = e["minute"] * 60 + e["second"]
    ins = av.Instante(float(t), pd.DataFrame(filas), _sb_xy(e["location"], p.largo, p.ancho), None, None, None)
    # Lo que pasó después del tiro: atajadas y desvíos no cuentan; un despeje del rival sí (juego deliberado)
    sig, idx = [], e["index"]
    for f in crudos[idx: idx + 15]:
        if f["index"] <= idx:
            continue
        if f["minute"] * 60 + f["second"] - t > 8:
            break
        tipo, mismo = f["type"]["name"], f["team"]["id"] == e["team"]["id"]
        pid = (f.get("player") or {}).get("id")
        if tipo == "Offside" and mismo:
            sig.append({"id": pid, "equipo": eq, "rol": "campo", "duracion": 1.0})
            break
        if tipo == "Goal Keeper" and not mismo:
            sig.append({"id": pid, "equipo": otro, "rol": "arquero", "duracion": 0.2})
        elif tipo in ("Block", "Deflection") and not mismo:
            sig.append({"id": pid, "equipo": otro, "rol": "campo", "duracion": 0.2})
        elif tipo in ("Ball Receipt*", "Pass", "Carry", "Shot", "Ball Recovery", "Miscontrol", "Dribble",
                      "Interception", "Clearance"):
            sig.append({"id": pid, "equipo": eq if mismo else otro, "rol": "campo", "duracion": 1.0})
        if len(sig) >= 4:
            break
    r = av.revisar_tiro(None, ins, eq, 1, p, tirador, sig)
    if r is None:
        return None
    r.update(periodo=e["period"], minuto=e["minute"], segundo=e["second"], evento=e["id"],
             coordenadas="equipo que remata ataca hacia +x",
             nombres={str(k): v for k, v in nombres.items() if k is not None and v},
             pitado=any(f["type"]["name"] == "Offside" and f["team"]["id"] == e["team"]["id"]
                        for f in crudos[idx + 1: idx + 12] if f["minute"] * 60 + f["second"] - t <= 10))
    return r


def var_statsbomb(partido, crudos: list[dict] | None = None, tres: dict | None = None) -> tuple[av.RevisionVar, None]:
    from .datos import statsbomb as sb
    pid = partido.info.get("partido_id")
    p = av.parametros_para("statsbomb360", 1.0, partido.cancha.largo, partido.cancha.ancho)
    crudos = crudos if crudos is not None else sb.eventos_crudos(pid)
    tres = tres if tres is not None else sb.tres_sesenta(pid)
    ev = partido.eventos
    equipo_de = ev.set_index("id")["equipo"].to_dict()
    rev = av.RevisionVar()
    goles_pos = {}
    for e in crudos:
        if e["type"]["name"] == "Shot" and (e.get("shot", {}).get("outcome", {}) or {}).get("name") == "Goal" \
                and e["period"] < 5:                                   # sin la tanda de penales
            goles_pos[(e["period"], e.get("possession"))] = e
    for e in crudos:
        tipo = e["type"]["name"]
        t = e["minute"] * 60 + e["second"]
        eq = equipo_de.get(e["id"], "A")
        if tipo == "Pass" and tres and e["id"] in tres:
            pa = e.get("pass", {})
            fin = pa.get("end_location")
            if not fin or fin[0] < 50:                    # el destino en campo propio: sin fuera de lugar posible
                continue
            tipo_pase = (pa.get("type") or {}).get("name")
            rean = {"Throw-in": "saque de banda", "Goal Kick": "saque de meta", "Corner": "córner"}.get(tipo_pase)
            ins = instante_360(tres[e["id"]], eq, e.get("location"), p.largo, p.ancho)
            ins.t = float(t)
            rec_xy = _sb_xy(fin, p.largo, p.ancho)
            r = av.revisar_fuera_de_lugar(ins, eq, 1, p, "pasador", None, rean, receptor_xy=rec_xy)
            r.update(t=float(t), periodo=e["period"], minuto=e["minute"], segundo=e["second"],
                     evento=e["id"], pasador_nombre=(e.get("player") or {}).get("name"),
                     receptor_nombre=(pa.get("recipient") or {}).get("name"),
                     resultado_evento=(pa.get("outcome") or {}).get("name") or "Completo",
                     posesion=e.get("possession"), coordenadas="equipo del pase ataca hacia +x")
            av.marcar_intervencion(r, r["resultado_evento"])
            if ins.zona_visible is not None:
                r["zona"] = np.round(ins.zona_visible, 1).tolist()
            r["nombres"] = {"pasador": r["pasador_nombre"]}
            if r.get("receptor") and r.get("receptor_nombre"):
                r["nombres"][r["receptor"]] = r["receptor_nombre"]
            g = goles_pos.get((e["period"], e.get("possession")))
            if g is not None and g["index"] > e["index"]:
                r["en_gol"] = g["minute"] * 60 + g["second"]
            rev.fueras_de_lugar.append(r)
        elif tipo == "Foul Committed" and e.get("location"):
            x, y = _sb_xy(e["location"], p.largo, p.ancho)
            fc = e.get("foul_committed", {}) or {}
            penal = bool(fc.get("penalty", False))
            mano = ((fc.get("type") or {}).get("name") == "Handball")
            tarjeta = (fc.get("card") or {}).get("name")
            dentro, dist = av.en_area(x, y, -1, p)       # el que comete defiende el arco de la izquierda
            oc = None
            eq_at = "B" if eq == "A" else "A"
            if tres and e["id"] in tres:
                ins = instante_360(tres[e["id"]], eq, e.get("location"), p.largo, p.ancho)
                ins.t = float(t)
                oc = av.revisar_ocasion(ins, eq_at, -1, x, y, p, None, "pasador")
            if dist < -4.0 and not penal and not (oc and oc["veredicto"] != "no es ocasión manifiesta"):
                continue
            f = av.revisar_falta(x, y, -1, p, 1.0, "penal" if penal else "tiro libre")
            if mano:
                f["motivo"] = "mano"
                f["explicacion"] = "Mano. " + f["explicacion"]
            f.update(t=float(t), periodo=e["period"], minuto=e["minute"], segundo=e["second"], equipo=eq_at,
                     infractor_equipo=eq, direccion=-1, tarjeta=tarjeta,
                     jugador_nombre=(e.get("player") or {}).get("name"), evento=e["id"],
                     coordenadas="equipo que comete la falta ataca hacia +x")
            if oc is not None:
                f["ocasion"] = {k: v for k, v in oc.items() if k != "jugadores"}
                f["jugadores"] = oc["jugadores"]
                f["defensor"] = "pasador"            # en la captura 360, el que comete la falta
                _sumar_ocasion(f, oc)
            if tarjeta:
                f["explicacion"] += f" Tarjeta mostrada: {TARJETA.get(tarjeta, tarjeta)}."
            rev.faltas.append(f)
        elif tipo == "Shot" and (e.get("shot") or {}).get("freeze_frame") and e.get("location") \
                and ((e.get("shot") or {}).get("type") or {}).get("name") == "Open Play" and e["period"] < 5:
            r = _tiro_statsbomb(e, crudos, eq, p)
            rev.tiros_revisados += 1
            if r is not None:
                rev.tiros.append(r)
    for (per, pos_id), g in goles_pos.items():
        en = [r for r in rev.fueras_de_lugar if r.get("posesion") == pos_id and r.get("periodo") == per]
        malos = [r for r in en if r["veredicto"] == "fuera de lugar"]
        dud = [r for r in en if r["veredicto"] == "dudoso"]
        t = g["minute"] * 60 + g["second"]
        if ((g.get("shot") or {}).get("type") or {}).get("name") == "Penalty":
            rev.goles.append({"tipo": "revisión de gol", "t": float(t), "periodo": per, "minuto": g["minute"],
                              "segundo": g["second"], "equipo": equipo_de.get(g["id"]),
                              "jugador_nombre": (g.get("player") or {}).get("name"), "veredicto": "gol válido",
                              "explicacion": "Gol de penal: no hay jugada previa que revisar por fuera de lugar."})
            continue
        rev.goles.append({
            "tipo": "revisión de gol", "t": float(t), "periodo": per, "minuto": g["minute"], "segundo": g["second"],
            "equipo": equipo_de.get(g["id"]), "jugador_nombre": (g.get("player") or {}).get("name"),
            "veredicto": "revisar" if malos else ("dudoso" if dud else "gol válido"),
            "revision_jugada": {"pases": len(en), "fuera_de_lugar": [r["t"] for r in malos], "dudosos": [r["t"] for r in dud]},
            "explicacion": (f"Jugada previa: {len(en)} pase(s) con captura 360 revisado(s). " +
                            ("Hay un pase con el receptor adelantado." if malos else
                             "Hay un pase dudoso." if dud else "Sin fuera de lugar.")) if en else
                           "Sin capturas 360 de la jugada previa: no se puede revisar el fuera de lugar.",
        })
    return rev, None


def var_partido(partido):
    fuente = partido.info.get("fuente", "")
    if partido.tiene_eventos:
        return var_statsbomb(partido)
    if fuente.startswith("SkillCorner"):
        return var_skillcorner(partido)
    return var_video(partido)
