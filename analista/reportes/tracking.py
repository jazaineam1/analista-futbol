"""Reporte HTML de un partido con datos de tracking (SkillCorner u otro proveedor)."""
from __future__ import annotations

from pathlib import Path

from .base import SvgCancha, e, fmt_t, kpi, leyenda, num, pagina, svg_calor, svg_control, svg_lineas


def svg_formacion(svg: SvgCancha, form: dict, equipo: str, nombres_jug: dict, titulo: str) -> str:
    """Forma media relativa del equipo, dibujada en su propio campo (ataca hacia la derecha)."""
    if not form.get("posiciones"):
        return ""
    out = []
    # La forma está centrada en el centroide: se dibuja con el centroide a 12 m de la mitad propia
    for p in form["posiciones"]:
        u, v = svg.p(p["x"] - 12, p["y"])
        nombre = nombres_jug.get(p["id"], p["id"])
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="9" class="dot dot-{equipo}" data-tip="{e(nombre)}"/>'
                   f'<text x="{u:.1f}" y="{v - 13:.1f}" class="node-label" text-anchor="middle">{e(nombre)}</text>')
    return svg.envolver("".join(out), titulo)


def reporte_tracking(partido, ruta: Path, titulo: str | None = None) -> Path:
    info = partido.info
    if info.get("fuente") == "video":
        from .video import reporte_video_partido
        return reporte_video_partido(partido, ruta, titulo)
    nombres = partido.nombres
    m = partido.metricas()
    r = m["resumen"]
    tr, forma, fis, pos = partido.tracking, m["forma"], partido.fisico(), m["poseedor"]
    marc = info.get("marcador") or {}
    titulo = titulo or f'{nombres["A"]} {marc.get("A", "")} – {marc.get("B", "")} {nombres["B"]}'
    sub = " · ".join(str(x) for x in (info.get("competicion"), info.get("fecha"), info.get("estadio")) if x)
    kits = {k: v.get("color") for k, v in info.get("equipos", {}).items()}
    goles = partido.goles()
    marcas = [(float(g.t) / 60, g.equipo, f"{fmt_t(g.t)} gol de {nombres.get(g.equipo)}"
               + (f" ({g.jugador})" if isinstance(getattr(g, "jugador", None), str) else "")) for g in goles.itertuples()]
    nombres_jug = tr.drop_duplicates("id").set_index("id")["nombre"].to_dict() if "nombre" in tr else {}

    fA, fB = partido.formacion("A"), partido.formacion("B")
    fm = r.get("forma", {})
    fe = r.get("fisico_equipo", {})
    pp = r.get("posesion_pct", {})
    kpis = "".join([
        kpi("Posesión", num(pp.get("A"), 0, "%"), num(pp.get("B"), 0, "%"), "jugador a ≤2 m del balón"),
        kpi("Formación sin balón", fA.get("formacion") or "–", fB.get("formacion") or "–", "a partir de la forma media"),
        kpi("Altura del bloque", num(fm.get("A", {}).get("altura_bloque"), 0, " m"), num(fm.get("B", {}).get("altura_bloque"), 0, " m"),
            "mediana, del arco propio a la última línea"),
        kpi("Ancho", num(fm.get("A", {}).get("ancho"), 0, " m"), num(fm.get("B", {}).get("ancho"), 0, " m"), "mediana de los 10 de campo"),
        kpi("Distancia recorrida", num(fe.get("A", {}).get("distancia_total_m", 0) / 1000, 1, " km"),
            num(fe.get("B", {}).get("distancia_total_m", 0) / 1000, 1, " km"), "suma del equipo"),
        kpi("Presión recibida", num(r.get("presion_sufrida", {}).get("A", {}).get("rivales_a_5m"), 2),
            num(r.get("presion_sufrida", {}).get("B", {}).get("rivales_a_5m"), 2), "rivales a ≤5 m del poseedor (media)"),
    ])

    # Series por minuto
    def por_minuto(df, col, func="median"):
        out = {}
        if df is None or df.empty:
            return out
        d = df.assign(minuto=(df["t"] // 60).astype(int))
        for eq, g in d.groupby("equipo"):
            if eq in ("A", "B"):
                s = getattr(g.groupby("minuto")[col], func)()
                out[eq] = s.rolling(5, center=True, min_periods=1).mean()
        return out
    serie_bloque = por_minuto(forma, "altura_bloque")
    serie_ancho = por_minuto(forma, "ancho")
    serie_pos = {}
    if len(pos):
        d = pos.dropna(subset=["equipo"]).assign(tramo=(pos.dropna(subset=["equipo"])["t"] // 300).astype(int))
        c = d.groupby(["tramo", "equipo"]).size().unstack(fill_value=0)
        tot = c.sum(axis=1)
        for eq in ("A", "B"):
            if eq in c:
                s = 100 * c[eq] / tot
                s.index = (s.index * 5 + 2.5).astype(float)
                serie_pos[eq] = s
    etiq_min = lambda v: f"{int(v)}'"  # noqa: E731

    # Control de cancha en cada gol
    svg_m = SvgCancha(partido.cancha, ancho_px=520)
    momentos = []
    for g in goles.itertuples():
        t0 = float(g.t) - 1.0
        xs, ys, P = partido.control_de_cancha(t0)
        j = partido.instante(t0)
        j = j[j["equipo"].isin(["A", "B"])]
        mapa = svg_control(svg_m, xs, ys, P, j, partido.balon_en(t0), "Control de cancha")
        quien = f" ({g.jugador})" if isinstance(getattr(g, "jugador", None), str) else ""
        momentos.append(f'<div class="card momento"><h3>{fmt_t(g.t)} · gol de {e(nombres.get(g.equipo))}{e(quien)}</h3>{mapa}'
                        f'<div class="nota">Un segundo antes del remate. Zonas azules: llega antes {e(nombres["A"])}; '
                        f'naranjas: {e(nombres["B"])}. Las rayas muestran hacia dónde corre cada jugador.</div></div>')

    form_html = "".join(
        f'<div class="card">{svg_formacion(svg_m, f, eq, nombres_jug, f"Forma de {nombres[eq]}")}'
        f'<div class="nota">{e(nombres[eq])}: <b>{e(f.get("formacion") or "–")}</b> sin balón · posición media relativa '
        f'al centro del equipo, atacando hacia la derecha</div></div>' for eq, f in (("A", fA), ("B", fB)))

    calores = "".join(
        f'<div class="card">{svg_calor(svg_m, tr.loc[(tr["equipo"] == eq) & (tr["rol"] != "arquero"), "x"] * m["direccion"].get(eq, 1), tr.loc[(tr["equipo"] == eq) & (tr["rol"] != "arquero"), "y"] * m["direccion"].get(eq, 1), eq)}'
        f'<div class="nota">Ocupación de {e(nombres[eq])} (ataca hacia la derecha)</div></div>' for eq in ("A", "B"))

    top = fis[fis["equipo"].isin(["A", "B"]) & (fis["rol"] != "arquero")].sort_values("distancia_m", ascending=False).head(12)
    filas_top = "".join(
        f'<tr><td><span class="sw sw-{x.equipo}"></span>{e(getattr(x, "nombre", None) or x.id)}</td>'
        f'<td class="num">{x.segundos_visible / 60:.0f}</td><td class="num">{x.distancia_m / 1000:.2f}</td>'
        f'<td class="num">{x.vel_max * 3.6:.1f}</td><td class="num">{x.seg_sprint:.0f}</td></tr>'
        for x in top.itertuples())

    var_html, var_css = "", ""
    try:
        from .var import seccion_var
        rev = partido.var()
        var_html, var_css = seccion_var(rev, partido._posiciones, nombres_jug, nombres, partido.cancha.largo,
                                        partido.cancha.ancho, info.get("fuente", "tracking"))
    except Exception as ex:  # noqa: BLE001  (el reporte sale igual sin la sección VAR)
        var_html = f'<h2>VAR</h2><p class="nota">No se pudo hacer la revisión: {e(ex)}</p>'

    cuerpo = f"""
<h1>{e(titulo)}</h1>
<p class="sub">{e(sub)} · tracking: {e(info.get("fuente", ""))} ({len(tr["cuadro"].unique()):,} cuadros)</p>
{leyenda(nombres, kits)}
<div class="kpis">{kpis}</div>
<h2>Posesión cada 5 minutos</h2>
<div class="card">{svg_lineas(serie_pos, nombres, "% posesión", marcas=marcas, etiqueta_x=etiq_min)}</div>
<h2>Altura del bloque (m desde el propio arco)</h2>
<div class="card">{svg_lineas(serie_bloque, nombres, "metros", marcas=marcas, etiqueta_x=etiq_min)}
<div class="nota">Mediana por minuto de la distancia entre el arco propio y la última línea. Más alto = el equipo defiende más arriba.</div></div>
<h2>Ancho del equipo (m)</h2>
<div class="card">{svg_lineas(serie_ancho, nombres, "metros", marcas=marcas, etiqueta_x=etiq_min)}</div>
<h2>Formación</h2>
<div class="grid2">{form_html}</div>
<h2>Control de cancha en los goles</h2>
<div class="grid2">{"".join(momentos) or '<p class="nota">Sin goles registrados.</p>'}</div>
<h2>Ocupación</h2>
<div class="grid2">{calores}</div>
<h2>Físico</h2>
<div class="card scroll"><table><tr><th>Jugador</th><th>Min. con tracking</th><th>Distancia (km)</th>
<th>Vel. máx (km/h)</th><th>Seg. a +25 km/h</th></tr>{filas_top}</table>
<div class="nota">Distancia durante los minutos con tracking (la transmisión no cubre repeticiones ni primeros planos).</div></div>
{var_html}
<h2>Notas</h2>
<div class="card"><ul class="nota">
<li>Posesión, pases y presión se infieren del tracking: tiene el balón quien está a ≤2 m de él.</li>
<li>VAR: pases y receptor buscado de los eventos dinámicos de SkillCorner; posiciones a 10 Hz interpoladas al instante del pase.
Un jugador fuera de cámara tiene posición estimada (borde punteado) y más incertidumbre.</li>
<li>Control de cancha: modelo de tiempo de llegada (reacción 0,7 s, 5 m/s, transición logística).</li>
<li>Velocidad máxima sostenida durante ~1 s, solo con posiciones observadas por la cámara.</li>
<li>Fuente: {e(info.get("fuente", ""))}. El equipo {e(nombres["A"])} se dibuja atacando hacia la derecha en todo el partido.</li>
</ul></div>"""
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(pagina(titulo, cuerpo, {"resumen": r}, extra_css=var_css), encoding="utf-8")
    return ruta
