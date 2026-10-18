"""Gráficos con matplotlib para notebooks y scripts (`pip install analista-futbol[graficos]`).

    from analista import Partido
    from analista import graficos as gr

    final = Partido.statsbomb(3869685)
    gr.tiros(final)
    gr.red_de_pases(final, "A")

    aleague = Partido.skillcorner(1886347)
    gr.control_de_cancha(aleague, t=5059.9)
"""
from __future__ import annotations

import numpy as np

from .cancha import Cancha
from .reportes.base import SERIE

COLORES = {"A": SERIE["A"][0], "B": SERIE["B"][0]}


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Los gráficos necesitan matplotlib: pip install 'analista-futbol[graficos]'") from e
    return plt


def cancha(cancha: Cancha | None = None, ax=None, color="#a9b8a6", fondo="#eef3ec", margen=3.0):
    """Dibuja la cancha (metros, origen en el centro) y devuelve el eje."""
    plt = _plt()
    c = cancha or Cancha()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 10 * (c.ancho + 2 * margen) / (c.largo + 2 * margen)))
    ax.set_facecolor(fondo)
    for pl in c.lineas(paso=0.5):
        ax.plot(pl[:, 0], pl[:, 1], color=color, lw=1.2, zorder=1)
    ax.set_xlim(-c.L2 - margen, c.L2 + margen)
    ax.set_ylim(-c.W2 - margen, c.W2 + margen)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    return ax


def tiros(partido, ax=None):
    """Mapa de tiros: A ataca a la derecha, B a la izquierda; tamaño = xG, relleno = gol."""
    t = partido.tiros()
    ax = cancha(partido.cancha, ax)
    for eq, g in t.groupby("equipo"):
        s = 1 if eq == "A" else -1
        gol = g["resultado"] == "Goal"
        tam = 30 + 900 * g["xg"].clip(lower=0.01)
        ax.scatter(s * g.loc[~gol, "x"], s * g.loc[~gol, "y"], s=tam[~gol], facecolors="none", edgecolors=COLORES[eq], lw=1.5, zorder=3)
        ax.scatter(s * g.loc[gol, "x"], s * g.loc[gol, "y"], s=tam[gol], color=COLORES[eq], edgecolors="#111", lw=1.2, zorder=4,
                   label=f"{partido.nombres[eq]} · xG {g['xg'].sum():.2f} · {int(gol.sum())} goles")
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    return ax


def red_de_pases(partido, equipo: str = "A", ax=None, min_pases: int = 3):
    """Red de pases hasta el primer cambio (el equipo ataca hacia la derecha)."""
    nodos, aristas = partido.red_de_pases(equipo, min_pases=min_pases)
    ax = cancha(partido.cancha, ax)
    pos = nodos.set_index("jugador") if len(nodos) else nodos
    if len(aristas):
        mx = aristas["n"].max()
        for r in aristas.itertuples():
            a, b = pos.loc[r.de], pos.loc[r.a]
            ax.plot([a.x, b.x], [a.y, b.y], color=COLORES[equipo], lw=0.5 + 6 * r.n / mx, alpha=0.55, zorder=2)
    if len(nodos):
        ax.scatter(nodos["x"], nodos["y"], s=60 + 6 * nodos["pases"], color=COLORES[equipo], edgecolors="white", lw=1.5, zorder=3)
        for r in nodos.itertuples():
            ax.annotate(str(r.jugador).split()[-1], (r.x, r.y), xytext=(0, 11), textcoords="offset points", ha="center", fontsize=8)
    ax.set_title(f"Red de pases · {partido.nombres[equipo]}")
    return ax


def calor(x, y, cancha_: Cancha | None = None, ax=None, equipo: str = "A", bins=(21, 14)):
    c = cancha_ or Cancha()
    ax = cancha(c, ax)
    H, xe, ye = np.histogram2d(x, y, bins=bins, range=[[-c.L2, c.L2], [-c.W2, c.W2]])
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("eq", ["#ffffff00", COLORES[equipo]])
    ax.imshow(H.T, origin="lower", extent=[-c.L2, c.L2, -c.W2, c.W2], cmap=cmap, alpha=0.85, zorder=0, aspect="auto")
    return ax


def control_de_cancha(partido, t: float, ax=None, **kw):
    """Quién llega antes a cada punto de la cancha en el instante t (tracking)."""
    from matplotlib.colors import LinearSegmentedColormap
    xs, ys, P = partido.control_de_cancha(t, **kw)
    ax = cancha(partido.cancha, ax)
    cmap = LinearSegmentedColormap.from_list("ctrl", [COLORES["B"], "#f4f4f4", COLORES["A"]])
    ax.imshow(P, origin="lower", extent=[xs.min(), xs.max(), ys.min(), ys.max()], cmap=cmap, vmin=0, vmax=1, alpha=0.6,
              zorder=0, aspect="auto")
    j = partido.instante(t)
    for eq, g in j[j["equipo"].isin(["A", "B"])].groupby("equipo"):
        ax.scatter(g["x"], g["y"], s=70, color=COLORES[eq], edgecolors="white", lw=1.5, zorder=3, label=partido.nombres[eq])
        if {"vx", "vy"} <= set(g.columns):
            ax.quiver(g["x"], g["y"], g["vx"], g["vy"], color=COLORES[eq], angles="xy", scale_units="xy", scale=1, width=0.003, zorder=2)
    b = partido.balon_en(t)
    if b is not None:
        ax.scatter([b[0]], [b[1]], s=40, color="#111", zorder=4)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False)
    return ax


def flujo_xg(partido, ax=None):
    plt = _plt()
    f = partido.flujo_xg()
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.5))
    for eq, g in f.groupby("equipo"):
        ax.step(g["minuto"], g["xg"], where="post", color=COLORES[eq], lw=2, label=partido.nombres[eq])
    for g in partido.goles().itertuples():
        ax.axvline(float(g.t) / 60, color=COLORES[g.equipo], ls=":", lw=1)
    ax.set_xlabel("minuto")
    ax.set_ylabel("xG acumulado")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    return ax
