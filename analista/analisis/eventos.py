"""Análisis de datos de eventos (pases, tiros, presiones…) en el esquema común.

Coordenadas en metros, origen en el centro; cada acción en la perspectiva del equipo que
la hace (atacando hacia +x). Ver `analista.datos.esquema`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import xg as modelo_xg

ACCIONES_DEFENSIVAS = {"duelo", "intercepcion", "falta", "bloqueo"}


def _juego(ev: pd.DataFrame) -> pd.DataFrame:
    """Eventos de juego (sin la tanda de penales)."""
    return ev[ev["periodo"] < 5] if "periodo" in ev else ev


def tiros(ev: pd.DataFrame, largo: float = 105.0) -> pd.DataFrame:
    t = _juego(ev)
    t = t[t["tipo"] == "tiro"].copy()
    if t.empty:
        return t
    t["gol"] = t["resultado"].eq("Goal")
    t["xg_modelo"] = modelo_xg.predecir(t, "auto", largo)
    if "xg" not in t or t["xg"].isna().all():
        t["xg"] = t["xg_modelo"]
    else:
        t["xg"] = t["xg"].fillna(t["xg_modelo"])
    t["distancia"], _ = modelo_xg.geometria(t["x"], t["y"], largo)
    return t


def goles(ev: pd.DataFrame) -> pd.DataFrame:
    """Goles (incluye autogoles a favor del rival)."""
    e = _juego(ev)
    g = e[(e["tipo"] == "tiro") & (e["resultado"] == "Goal")][["t", "minuto", "equipo", "jugador"]].copy()
    ag = e[e["tipo"] == "autogol_en_contra"][["t", "minuto", "equipo", "jugador"]].copy()
    if len(ag):
        ag["equipo"] = ag["equipo"].map({"A": "B", "B": "A"})
        ag["jugador"] = ag["jugador"] + " (en contra)"
    return pd.concat([g, ag]).sort_values("t").reset_index(drop=True)


def flujo_xg(ev: pd.DataFrame, largo: float = 105.0) -> pd.DataFrame:
    """xG acumulado en el tiempo por equipo (minuto a minuto)."""
    t = tiros(ev, largo)
    fin = float(_juego(ev)["minuto"].max() or 90) + 1
    filas = []
    for eq in ("A", "B"):
        te = t[t["equipo"] == eq].sort_values("t") if len(t) else t
        acum = 0.0
        filas.append({"equipo": eq, "minuto": 0.0, "xg": 0.0})
        for r in te.itertuples():
            filas.append({"equipo": eq, "minuto": r.t / 60, "xg": acum})
            acum += r.xg
            filas.append({"equipo": eq, "minuto": r.t / 60, "xg": acum})
        filas.append({"equipo": eq, "minuto": fin, "xg": acum})
    return pd.DataFrame(filas)


def posesion(ev: pd.DataFrame) -> dict[str, float]:
    """% de posesión por tiempo: suma de la duración de los eventos de cada posesión."""
    e = _juego(ev)
    if "equipo_posesion" not in e or e["duracion"].isna().all():
        p = e[e["tipo"] == "pase"]["equipo"].value_counts(normalize=True)
        return {k: round(100 * float(p.get(k, 0)), 1) for k in ("A", "B")}
    nombres = e.drop_duplicates("equipo_nombre").set_index("equipo_nombre")["equipo"].to_dict()
    d = e.dropna(subset=["duracion"]).groupby("equipo_posesion")["duracion"].sum()
    d.index = d.index.map(nombres)
    tot = d.sum()
    return {k: round(100 * float(d.get(k, 0)) / tot, 1) for k in ("A", "B")} if tot else {"A": 50.0, "B": 50.0}


def ppda(ev: pd.DataFrame, largo: float = 105.0) -> dict[str, float]:
    """Pases que permite cada equipo por acción defensiva en los 60% de la cancha más alejados de su arco.

    Bajo = presión alta. Pases del rival en su propio 60% (x < 0.1·L en su perspectiva) dividido
    por duelos, intercepciones, faltas y bloqueos propios en esa zona (x > -0.1·L en la propia).
    """
    e = _juego(ev)
    out = {}
    limite = 0.1 * largo
    for eq, rival in (("A", "B"), ("B", "A")):
        pases_riv = ((e["equipo"] == rival) & (e["tipo"] == "pase") & (e["x"] < limite)).sum()
        acc = ((e["equipo"] == eq) & e["tipo"].isin(ACCIONES_DEFENSIVAS) & (e["x"] > -limite)).sum()
        out[eq] = round(float(pases_riv) / acc, 1) if acc else None
    return out


def field_tilt(ev: pd.DataFrame, largo: float = 105.0) -> dict[str, float]:
    """% de los pases en el último tercio que hace cada equipo (dominio territorial)."""
    e = _juego(ev)
    p = e[(e["tipo"] == "pase") & (e["x"] > largo / 6)]["equipo"].value_counts()
    tot = p.sum()
    return {k: round(100 * float(p.get(k, 0)) / tot, 1) for k in ("A", "B")} if tot else {"A": None, "B": None}


def pases_progresivos(ev: pd.DataFrame, largo: float = 105.0) -> pd.DataFrame:
    """Pases completos (sin balón parado) que acercan el balón al menos un 25% al arco rival
    y terminan en la mitad rival."""
    e = _juego(ev)
    p = e[(e["tipo"] == "pase") & (e["exito"] == True)].copy()  # noqa: E712
    if "subtipo" in p:
        p = p[~p["subtipo"].isin(["Corner", "Free Kick", "Throw-in", "Goal Kick", "Kick Off"])]
    gx = largo / 2
    d0 = np.hypot(gx - p["x"], p["y"])
    d1 = np.hypot(gx - p["x_fin"], p["y_fin"])
    return p[(d1 <= 0.75 * d0) & (p["x_fin"] > 0)]


def red_de_pases(ev: pd.DataFrame, equipo: str, hasta_primer_cambio: bool = True,
                 min_pases: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Red de pases de un equipo: (nodos, aristas).

    nodos: jugador, x, y (posición media de sus pases y recepciones), pases.
    aristas: de, a, n (pases completos entre ambos, en cualquier dirección).
    Por defecto usa el tramo hasta el primer cambio, para que sean los mismos 11.
    """
    e = _juego(ev)
    e = e[e["equipo"] == equipo]
    if hasta_primer_cambio:
        cambios = e[e["tipo"] == "cambio"]
        if len(cambios):
            e = e[e["t"] < cambios["t"].min()]
    p = e[(e["tipo"] == "pase") & (e["exito"] == True) & e["receptor"].notna()]  # noqa: E712
    if p.empty:
        return pd.DataFrame(columns=["jugador", "x", "y", "pases"]), pd.DataFrame(columns=["de", "a", "n"])
    pos = pd.concat([p[["jugador", "x", "y"]],
                     p[["receptor", "x_fin", "y_fin"]].rename(columns={"receptor": "jugador", "x_fin": "x", "y_fin": "y"})])
    nodos = pos.groupby("jugador").agg(x=("x", "mean"), y=("y", "mean"), pases=("x", "size")).reset_index()
    par = p.apply(lambda r: tuple(sorted((r["jugador"], r["receptor"]))), axis=1)
    aristas = par.value_counts().reset_index()
    aristas.columns = ["par", "n"]
    aristas[["de", "a"]] = pd.DataFrame(aristas["par"].tolist(), index=aristas.index)
    aristas = aristas[aristas["n"] >= min_pases][["de", "a", "n"]].reset_index(drop=True)
    return nodos, aristas


def recuperaciones_altas(ev: pd.DataFrame, largo: float = 105.0) -> pd.DataFrame:
    e = _juego(ev)
    return e[e["tipo"].isin(["recuperacion", "intercepcion"]) & (e["x"] > largo / 6)]


def resumen(ev: pd.DataFrame, largo: float = 105.0) -> dict:
    """Indicadores clave del partido por equipo ('A' local, 'B' visitante)."""
    e = _juego(ev)
    t = tiros(ev, largo)
    g = goles(ev)
    pas = e[e["tipo"] == "pase"]
    prog = pases_progresivos(ev, largo)
    rec = recuperaciones_altas(ev, largo)
    pos, pp, ft = posesion(ev), ppda(ev, largo), field_tilt(ev, largo)
    nombres = e.drop_duplicates("equipo").set_index("equipo")["equipo_nombre"].to_dict() if "equipo_nombre" in e else {}
    out = {"equipos": {k: nombres.get(k, f"Equipo {k}") for k in ("A", "B")}, "por_equipo": {}}
    for eq in ("A", "B"):
        te = t[t["equipo"] == eq] if len(t) else t
        pe = pas[pas["equipo"] == eq]
        out["por_equipo"][eq] = {
            "goles": int((g["equipo"] == eq).sum()),
            "xg": round(float(te["xg"].sum()), 2) if len(te) else 0.0,
            "xg_modelo": round(float(te["xg_modelo"].sum()), 2) if len(te) else 0.0,
            "tiros": int(len(te)),
            "tiros_al_arco": int(te["resultado"].isin(["Goal", "Saved", "Saved To Post"]).sum()) if len(te) else 0,
            "xg_por_tiro": round(float(te["xg"].mean()), 3) if len(te) else None,
            "pases": int(len(pe)),
            "precision_pases": round(100 * float(pe["exito"].mean()), 1) if len(pe) else None,
            "pases_progresivos": int((prog["equipo"] == eq).sum()),
            "posesion": pos.get(eq), "ppda": pp.get(eq), "field_tilt": ft.get(eq),
            "recuperaciones_altas": int((rec["equipo"] == eq).sum()),
            "presiones": int(((e["equipo"] == eq) & (e["tipo"] == "presion")).sum()),
        }
    return out


def calor(ev: pd.DataFrame, equipo: str, tipos: set[str] | None = None, bins=(12, 8), largo: float = 105.0,
          ancho: float = 68.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Histograma 2D de acciones de un equipo (por defecto, todas las acciones con balón)."""
    e = _juego(ev)
    e = e[(e["equipo"] == equipo) & e["x"].notna()]
    if tipos:
        e = e[e["tipo"].isin(tipos)]
    H, xe, ye = np.histogram2d(e["x"], e["y"], bins=bins, range=[[-largo / 2, largo / 2], [-ancho / 2, ancho / 2]])
    return H, xe, ye
