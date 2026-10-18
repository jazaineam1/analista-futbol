"""Reporte HTML de un partido con datos de eventos (StatsBomb u otro proveedor en el esquema común)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ..analisis import eventos as ae
from .base import SvgCancha, e, kpi, leyenda, num, pagina, svg_calor, svg_lineas, svg_red


def _corto(nombre: str) -> str:
    """'Lionel Messi' -> 'L. Messi'; 'Mac Allister' se conserva si ya es corto."""
    if not nombre:
        return ""
    p = str(nombre).split()
    if len(p) <= 1:
        return str(nombre)
    if len(p) == 2:
        return f"{p[0][0]}. {p[1]}"
    return f"{p[0][0]}. {' '.join(p[1:])}" if len(" ".join(p[1:])) <= 14 else f"{p[0][0]}. {p[-1]}"


def svg_tiros(svg: SvgCancha, tiros: pd.DataFrame, nombres: dict) -> str:
    """Ambos equipos en la misma cancha: A ataca a la derecha, B a la izquierda."""
    out = []
    for r in tiros.sort_values("xg").itertuples():
        s = 1 if r.equipo == "A" else -1
        u, v = svg.p(s * r.x, s * r.y)
        rad = 3 + 16 * np.sqrt(max(r.xg, 0.005))
        gol = r.resultado == "Goal"
        tip = (f"{e(nombres.get(r.equipo))} · {e(r.jugador)}<br>min {r.minuto} · xG {num(r.xg, 2)}"
               f"{' · modelo propio ' + num(r.xg_modelo, 2) if hasattr(r, 'xg_modelo') else ''}<br>{e(r.resultado)}")
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="{rad:.1f}" class="shot shot-{r.equipo}{" gol" if gol else ""}" '
                   f'data-tip="{tip}"/>')
    return svg.envolver("".join(out), "Mapa de tiros")


def reporte_eventos(partido, ruta: Path, titulo: str | None = None) -> Path:
    ev = partido.eventos
    L = partido.cancha.largo
    nombres = partido.nombres
    r = ae.resumen(ev, L)
    pe = r["por_equipo"]
    marcador = f'{pe["A"]["goles"]} – {pe["B"]["goles"]}'
    titulo = titulo or f'{nombres["A"]} {marcador} {nombres["B"]}'
    info = partido.info
    sub = " · ".join(str(x) for x in (info.get("competicion"), info.get("temporada"), info.get("fase"),
                                       info.get("fecha"), info.get("estadio")) if x)
    a, b = pe["A"], pe["B"]
    kpis = "".join([
        kpi("Goles esperados (xG)", num(a["xg"], 2), num(b["xg"], 2), f"goles: {a['goles']} · {b['goles']}"),
        kpi("Tiros (al arco)", f'{a["tiros"]} ({a["tiros_al_arco"]})', f'{b["tiros"]} ({b["tiros_al_arco"]})',
            "xG por tiro: " + num(a["xg_por_tiro"], 2) + " · " + num(b["xg_por_tiro"], 2)),
        kpi("Posesión", num(a["posesion"], 0, "%"), num(b["posesion"], 0, "%"), "por tiempo con el balón"),
        kpi("Pases completos", num(a["precision_pases"], 0, "%"), num(b["precision_pases"], 0, "%"),
            f'{a["pases"]} · {b["pases"]} pases'),
        kpi("PPDA", num(a["ppda"], 1), num(b["ppda"], 1), "pases permitidos por acción defensiva; bajo = presión alta"),
        kpi("Field tilt", num(a["field_tilt"], 0, "%"), num(b["field_tilt"], 0, "%"), "parte de los pases en el último tercio"),
    ])
    filas_cmp = [("Goles", a["goles"], b["goles"], 0), ("xG (StatsBomb)", a["xg"], b["xg"], 2),
                 ("xG (modelo propio)", a["xg_modelo"], b["xg_modelo"], 2), ("Tiros", a["tiros"], b["tiros"], 0),
                 ("Tiros al arco", a["tiros_al_arco"], b["tiros_al_arco"], 0), ("Pases", a["pases"], b["pases"], 0),
                 ("Precisión de pase", a["precision_pases"], b["precision_pases"], 1),
                 ("Pases progresivos", a["pases_progresivos"], b["pases_progresivos"], 0),
                 ("Presiones", a["presiones"], b["presiones"], 0),
                 ("Recuperaciones en el último tercio", a["recuperaciones_altas"], b["recuperaciones_altas"], 0),
                 ("PPDA", a["ppda"], b["ppda"], 1), ("Field tilt (%)", a["field_tilt"], b["field_tilt"], 1)]
    tabla = "".join(f"<tr><td>{e(n)}</td><td>{num(x, d)}</td><td></td><td>{num(y, d)}</td></tr>"
                    for n, x, y, d in filas_cmp)
    tabla = (f'<table class="cmp"><tr><th>Indicador</th><th style="text-align:right">{e(nombres["A"])}</th><th></th>'
             f'<th>{e(nombres["B"])}</th></tr>{tabla}</table>')

    # Flujo de xG
    fx = ae.flujo_xg(ev, L)
    series = {eq: g.set_index("minuto")["xg"] for eq, g in fx.groupby("equipo")}
    gl = ae.goles(ev)
    marcas = [(float(g.t) / 60, g.equipo, f"{g.minuto}' {g.jugador}") for g in gl.itertuples()]
    flujo = svg_lineas(series, nombres, "xG acumulado", escalon=False, marcas=marcas,
                       etiqueta_x=lambda m: f"{int(m)}'")

    t = ae.tiros(ev, L)
    svg = SvgCancha(partido.cancha, ancho_px=640)
    mapa_tiros = svg_tiros(svg, t, nombres) if len(t) else '<p class="nota">Sin tiros.</p>'

    svg_s = SvgCancha(partido.cancha, ancho_px=520)
    redes = []
    for eq in ("A", "B"):
        n, ar = ae.red_de_pases(ev, eq)
        redes.append(f'<div class="card">{svg_red(svg_s, n, ar, eq, _corto, f"Red de pases {nombres[eq]}") or "<p class=nota>Sin pases.</p>"}'
                     f'<div class="nota">Red de pases de {e(nombres[eq])} hasta el primer cambio · grosor = pases entre ambos (≥ 3) · '
                     f'el equipo ataca hacia la derecha</div></div>')
    calores = []
    for eq in ("A", "B"):
        pr = ev[(ev["equipo"] == eq) & (ev["tipo"] == "presion") & (ev["periodo"] < 5)]
        calores.append(f'<div class="card">{svg_calor(svg_s, pr["x"], pr["y"], eq, (12, 8), "Presiones") or ""}'
                       f'<div class="nota">Dónde presiona {e(nombres[eq])} ({len(pr)} presiones) · ataca hacia la derecha</div></div>')

    # Jugadores destacados
    juego = ev[ev["periodo"] < 5]
    prog = ae.pases_progresivos(ev, L)
    jug = juego[juego["jugador"].notna()].groupby(["jugador", "equipo"]).agg(
        acciones=("id", "size")).reset_index()
    if len(t):
        jug = jug.merge(t.groupby("jugador").agg(xg=("xg", "sum"), tiros=("id", "size"), goles=("gol", "sum")),
                        on="jugador", how="left")
    jug = jug.merge(prog.groupby("jugador").size().rename("progresivos"), on="jugador", how="left")
    jug = jug.merge(juego[juego["tipo"] == "presion"].groupby("jugador").size().rename("presiones"), on="jugador", how="left")
    jug = jug.fillna(0)
    top = jug.sort_values(["xg", "progresivos"], ascending=False).head(12) if "xg" in jug else jug.head(12)
    filas_top = "".join(
        f'<tr><td><span class="sw sw-{x.equipo}"></span>{e(_corto(x.jugador))}</td><td class="num">{int(getattr(x, "goles", 0))}</td>'
        f'<td class="num">{num(float(getattr(x, "xg", 0)), 2)}</td><td class="num">{int(getattr(x, "tiros", 0))}</td>'
        f'<td class="num">{int(x.progresivos)}</td><td class="num">{int(x.presiones)}</td></tr>' for x in top.itertuples())

    cuerpo = f"""
<h1>{e(titulo)}</h1>
<p class="sub">{e(sub)} · datos de eventos: {e(info.get("fuente", ""))}</p>
{leyenda(nombres)}
<div class="kpis">{kpis}</div>
<h2>Cómo se fue dando el partido: xG acumulado</h2>
<div class="card">{flujo}<div class="nota">Cada escalón es un tiro; su altura es la probabilidad de gol (xG). Las líneas verticales son los goles.</div></div>
<h2>Mapa de tiros</h2>
<div class="card">{mapa_tiros}<div class="nota">{e(nombres["A"])} ataca hacia la derecha y {e(nombres["B"])} hacia la izquierda.
Tamaño = xG; borde oscuro = gol. Pasa el cursor para ver el tirador y el xG de nuestro modelo.</div></div>
<h2>Comparación</h2>
<div class="card scroll">{tabla}</div>
<h2>Redes de pases</h2>
<div class="grid2">{"".join(redes)}</div>
<h2>Presión</h2>
<div class="grid2">{"".join(calores)}</div>
<h2>Jugadores destacados</h2>
<div class="card scroll"><table><tr><th>Jugador</th><th>Goles</th><th>xG</th><th>Tiros</th><th>Pases progresivos</th><th>Presiones</th></tr>{filas_top}</table></div>
<h2>Notas</h2>
<div class="card"><ul class="nota">
<li>xG de StatsBomb y xG de nuestro modelo (regresión logística entrenada con 9.817 tiros de la Premier League 2015/16;
en 4.709 tiros sin penales de torneos internacionales predice 434,7 goles frente a 435 reales; el penal vale 0,78).</li>
<li>Pases progresivos: completos, sin balón parado, que acercan el balón al menos un 25% al arco rival y terminan en su mitad.</li>
<li>PPDA: pases del rival en su 60% de cancha divididos por duelos, intercepciones, faltas y bloqueos propios en esa zona.</li>
<li>Se excluye la tanda de penales. Fuente de los datos: {e(info.get("fuente", ""))}{" (Hudl StatsBomb)" if "StatsBomb" in str(info.get("fuente")) else ""}.</li>
</ul></div>"""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(pagina(titulo, cuerpo, {"resumen": r}), encoding="utf-8")
    return ruta
