"""Reporte HTML del análisis de un video (lo que produce `analista analizar`)."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ..cancha import Cancha
from .base import SvgCancha, e, fmt_t, kpi, leyenda, num, pagina, svg_barra_posesion, svg_calor, svg_lineas


def svg_posiciones(svg: SvgCancha, fis: pd.DataFrame, direccion: dict, nombres: dict, min_seg: float = 3.0) -> str:
    """Posición media de cada pista (cada equipo atacando hacia la derecha)."""
    if fis.empty:
        return ""
    out = []
    f = fis[(fis["segundos_visible"] >= min_seg) & fis["equipo"].isin(["A", "B"])]
    for r in f.itertuples():
        d = direccion.get(r.equipo, 1)
        u, v = svg.p(r.x_media * d, r.y_media * d)
        tip = (f"{nombres.get(r.equipo, r.equipo)} · pista {r.id}<br>{'Arquero · ' if r.rol == 'arquero' else ''}"
               f"visible {r.segundos_visible:.0f} s · {r.distancia_m:.0f} m recorridos")
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="7" class="dot dot-{r.equipo}" data-tip="{e(tip)}"/>')
    return svg.envolver("".join(out), "Posiciones medias por jugador")


def svg_momento(svg: SvgCancha, df: pd.DataFrame, balon: pd.DataFrame | None, cuadro: int, nombres: dict,
                segundos_estela: float = 3.0) -> str:
    """Fotograma táctico: dónde estaba cada jugador en el instante del evento y el recorrido del balón."""
    cx, cy = ("x_suave", "y_suave") if "x_suave" in df.columns else ("x", "y")
    if df.empty:
        return ""
    cuadros = np.sort(df["cuadro"].unique())
    k = cuadros[np.argmin(np.abs(cuadros - cuadro))]
    g = df[df["cuadro"] == k]
    t_ev = float(g["t"].iloc[0]) if len(g) else 0.0
    out = []
    if balon is not None and len(balon):
        b = balon[(balon["t"] >= t_ev - segundos_estela) & (balon["t"] <= t_ev + 0.5)].sort_values("t")
        if len(b) >= 2:
            pts = " ".join(f"{u:.1f},{v:.1f}" for u, v in (svg.p(x, y) for x, y in zip(b["x"], b["y"])))
            out.append(f'<polyline points="{pts}" class="estela"/>')
    for r in g.itertuples():
        u, v = svg.p(getattr(r, cx), getattr(r, cy))
        eq = r.equipo if r.equipo in ("A", "B") else "O"
        tip = f"{nombres.get(r.equipo, 'árbitro / otro')} · pista {r.id}"
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="6" class="dot dot-{eq}" data-tip="{e(tip)}"/>')
    if balon is not None and len(balon):
        bb = balon.iloc[(balon["t"] - t_ev).abs().argsort()[:1]]
        if len(bb) and abs(float(bb["t"].iloc[0]) - t_ev) < 0.6:
            u, v = svg.p(float(bb["x"].iloc[0]), float(bb["y"].iloc[0]))
            out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="4.5" class="balon" data-tip="balón"/>')
    return svg.envolver("".join(out), "Posiciones en el momento del evento")


def _validacion(salida: Path, nombres: dict) -> str:
    ruta_eval = Path(salida) / "evaluacion.json"
    if not ruta_eval.exists():
        return ""
    ev_v = json.loads(ruta_eval.read_text(encoding="utf-8"))
    j, b, po, go, ca = (ev_v.get(k, {}) for k in ("jugadores", "balon", "posesion", "goles", "camara"))

    def fila(nombre, valor, que):
        return f"<tr><td>{e(nombre)}</td><td class='num'><b>{e(valor)}</b></td><td class='nota'>{e(que)}</td></tr>"

    def pct(v):
        return "–" if v is None else f"{v}%"

    def frac(v):
        return "–" if v is None else f"{100 * v:.1f}%"

    filas = [
        fila("Jugadores detectados (recall)", frac(j.get("recall")), "de los jugadores que se veían en cada cuadro"),
        fila("Detecciones correctas (precisión)", frac(j.get("precision")), "lo detectado que era un jugador real (a ≤2 m)"),
        fila("Error de posición", f'{j.get("error_mediana_m", "–")} m', f'mediana; el 90% queda bajo {j.get("error_p90_m", "–")} m'),
        fila("Equipo correcto", frac(j.get("equipo_correcto")), "color de camiseta asignado al equipo real (jugadores de campo)"),
        fila("Pureza de identidad", frac(j.get("pureza_identidad")),
             f'cada pista sigue a un mismo jugador; {j.get("pistas", "–")} pistas para {j.get("jugadores_distintos", "–")} jugadores'),
        fila("Balón ubicado a ≤2 m", pct(b.get("detectado_pct")),
             f'de los cuadros con balón visible; {pct(b.get("detectado_en_suelo_pct"))} cuando va por el suelo'),
    ]
    if ca:
        filas.append(fila("Calibración de la cámara", f'{ca.get("error_mediana_m", "–")} m',
                          f'error mediano en la cancha; calibrada en {pct(ca.get("calibrado_pct"))} de los cuadros de juego'))
    if po:
        def par(d):
            d = d or {}
            return f'{d.get("A") or 0:.0f}% · {d.get("B") or 0:.0f}%'
        filas.append(fila("Posesión", f'{par(po.get("estimada"))}',
                          f'estimada por la app; real {par(po.get("real"))} ({nombres.get("A")} · {nombres.get("B")})'))
    if go:
        filas.append(fila("Goles detectados", f'{go.get("aciertos", 0)} de {len(go.get("reales", []))}',
                          "gol = el balón cruza la línea de meta entre los postes (±2 s del gol real)"))
    return ('<h2>Validación contra la verdad</h2><p class="nota">Este video tiene posiciones reales conocidas, '
            'así que se puede medir el error de la app.</p>'
            '<div class="card scroll"><table><tr><th>Medida</th><th>Resultado</th><th>Qué significa</th></tr>'
            + "".join(filas) + "</table></div>")


def _seccion_var(salida: Path, var=None) -> tuple[str, str]:
    """Sección VAR del reporte de video. var = (RevisionVar, Posiciones, fotos, clips) o None para calcularla."""
    import json

    from .var import seccion_var
    try:
        from ..partido import Partido
        p = Partido.video(salida)
        if var is None:
            fotos, clips = {}, {}
            if (salida / "var.json").exists():
                guardado = json.loads((salida / "var.json").read_text(encoding="utf-8"))
                fotos = {float(k): v for k, v in guardado.get("fotos", {}).items() if (salida / v).exists()}
                clips = {float(k): v for k, v in guardado.get("clips", {}).items() if (salida / v).exists()}
            rev, pos = p.var(), p._posiciones
        else:
            rev, pos, fotos, clips = var
        ids = sorted(set(p.tracking["id"].dropna())) if len(p.tracking) else []
        nombres_jug = {i: f"#{int(i)}" if isinstance(i, (int, float)) and float(i).is_integer() else str(i) for i in ids}
        return seccion_var(rev, pos, nombres_jug, p.nombres, p.cancha.largo, p.cancha.ancho,
                           "video analizado por la app", fotos, clips)
    except Exception as ex:  # noqa: BLE001
        return f'<h2>VAR</h2><p class="nota">No se pudo hacer la revisión VAR: {e(ex)}</p>', ""


def generar_reporte(salida: Path, df: pd.DataFrame, calc: dict, resumen_proceso: dict, cancha: Cancha,
                    titulo: str = "Análisis del partido", nota_fuente: str = "", var=None) -> Path:
    salida = Path(salida)
    var_html, var_css = _seccion_var(salida, var)
    r = calc["resumen"]
    nombres = resumen_proceso.get("nombres_equipos", {"A": "Equipo A", "B": "Equipo B"})
    kits = resumen_proceso.get("colores_kit") or resumen_proceso.get("colores_equipos", {})
    direccion = calc["direccion"]
    fis, forma, pos, pases = calc["fisico"], calc["forma"], calc["poseedor"], calc["pases"]
    svg = SvgCancha(cancha)

    pos_pct = r.get("posesion_pct", {})
    psr = r.get("pases", {})
    pres = r.get("presion_sufrida", {})
    fm = r.get("forma", {})
    kpis = "".join([
        kpi("Posesión", num(pos_pct.get("A"), 0, "%"), num(pos_pct.get("B"), 0, "%"), "con poseedor claro (≤2 m del balón)"),
        kpi("Pases completados", f'{psr.get("A", {}).get("completados", 0)}/{psr.get("A", {}).get("intentos", 0)}',
            f'{psr.get("B", {}).get("completados", 0)}/{psr.get("B", {}).get("intentos", 0)}', "estimado por cambios de poseedor"),
        kpi("Altura del bloque", num(fm.get("A", {}).get("altura_bloque"), 0, " m"), num(fm.get("B", {}).get("altura_bloque"), 0, " m"),
            "mediana, del propio arco a la última línea visible"),
        kpi("Presión recibida", num(pres.get("A", {}).get("rivales_a_5m"), 2), num(pres.get("B", {}).get("rivales_a_5m"), 2),
            "rivales a ≤5 m del poseedor (media)"),
        kpi("Ancho del equipo", num(fm.get("A", {}).get("ancho"), 0, " m"), num(fm.get("B", {}).get("ancho"), 0, " m"),
            "mediana de los jugadores visibles"),
    ])

    evs = calc["eventos"]
    filas_ev = "".join(
        f'<tr><td>{fmt_t(x.t)}</td><td><b>{e(x.tipo)}</b></td><td>{e(nombres.get(x.equipo_atacante, x.equipo_atacante or "–"))}</td>'
        f'<td>{x.x:.0f}, {x.y:.0f}</td><td>{"" if x.velocidad is None or pd.isna(x.velocidad) else f"{x.velocidad:.0f} m/s"}</td></tr>'
        for x in evs.itertuples()) if len(evs) else '<tr><td colspan="5">Sin remates ni goles detectados en el clip.</td></tr>'

    # Fotogramas tácticos de los momentos clave (goles primero, luego remates)
    momentos_html = ""
    if len(evs) and "cuadro" in evs:
        orden = {"gol": 0, "posible gol": 1, "remate": 2}
        sel = evs[evs["tipo"].isin(orden)].copy()
        if len(sel):
            sel["o"] = sel["tipo"].map(orden)
            sel = sel.sort_values(["o", "t"]).head(3).sort_values("t")
            svg_m = SvgCancha(cancha, ancho_px=420)
            tarjetas = []
            for x in sel.itertuples():
                figura = svg_momento(svg_m, df, calc.get("balon"), int(x.cuadro), nombres)
                if figura:
                    quien = nombres.get(x.equipo_atacante, "")
                    tarjetas.append(f'<div class="card momento"><h3>{fmt_t(x.t)} · {e(x.tipo)}'
                                    f'{" de " + e(quien) if quien else ""}</h3>{figura}'
                                    f'<div class="nota">Línea punteada: recorrido del balón en los 3 s previos.</div></div>')
            if tarjetas:
                momentos_html = ('<h2>Cómo estaban parados en cada momento clave</h2><div class="grid2">'
                                 + "".join(tarjetas) + "</div>")

    # Series por segundo
    series_bloque, series_ancho = {}, {}
    if len(forma):
        f2 = forma.copy()
        f2["seg"] = f2["t"].astype(float).round().astype(int)
        for eq in ("A", "B"):
            g = f2[f2["equipo"] == eq].groupby("seg")
            if len(g):
                series_bloque[eq] = g["altura_bloque"].median().rolling(3, min_periods=1).median()
                series_ancho[eq] = g["ancho"].median().rolling(3, min_periods=1).median()
    por_seg = {}
    if len(pos) and pos["equipo"].notna().any():
        p2 = pos.dropna(subset=["equipo"]).copy()
        p2["seg"] = p2["t"].astype(float).round().astype(int)
        por_seg = p2.groupby("seg")["equipo"].agg(lambda s: s.value_counts().index[0]).to_dict()

    # Red de pases (estimada por cambios de poseedor)
    red = ""
    if len(pases) and len(fis):
        from .base import svg_red
        for eq in ("A",):
            p = pases[(pases["equipo"] == eq) & pases["completado"]]
            if len(p):
                d = direccion.get(eq, 1)
                med = fis.set_index("id")
                nodos = pd.DataFrame([{"jugador": pid, "x": med.at[pid, "x_media"] * d, "y": med.at[pid, "y_media"] * d,
                                       "pases": int(((p["de"] == pid) | (p["a"] == pid)).sum())}
                                      for pid in set(p["de"]) | set(p["a"]) if pid in med.index])
                ar = p.assign(par=p.apply(lambda q: tuple(sorted((q["de"], q["a"]))), axis=1)).groupby("par").size()
                ar = pd.DataFrame([{"de": a, "a": b, "n": n} for (a, b), n in ar.items()])
                red = svg_red(svg, nodos, ar, eq, lambda j: f"{j}", "Red de pases")

    top = ""
    if len(fis):
        f = fis[fis["equipo"].isin(["A", "B"])].sort_values("distancia_m", ascending=False).head(10)
        top = "".join(f'<tr><td><span class="sw sw-{x.equipo}"></span>{e(nombres.get(x.equipo, x.equipo))}</td>'
                      f'<td>{x.id}</td><td>{e(x.rol)}</td><td class="num">{x.segundos_visible:.0f}</td><td class="num">{x.distancia_m:.0f}</td>'
                      f'<td class="num">{x.vel_max * 3.6:.1f}</td><td class="num">{x.m_por_min:.0f}</td></tr>' for x in f.itertuples())

    SIN_BALON = ('<p class="nota">El balón no se detectó con suficiente frecuencia para estimar posesión, '
                 'toques y pases. Con un modelo afinado para fútbol (balón pequeño) esta sección se llena.</p>')
    cal = r.get("calidad", {})
    dur = float(df["t"].max() - df["t"].min()) if len(df) else 0
    nota_cobertura = ("Cámara fija: se ve toda la zona calibrada; forma, ancho y altura usan a todos los jugadores detectados."
                      if resumen_proceso.get("camara_fija") else
                      "La transmisión solo muestra parte de la cancha: forma, ancho y altura se miden sobre los jugadores visibles.")

    cuerpo = f"""
<h1>{e(titulo)}</h1>
<p class="sub">{e(nota_fuente)} · {fmt_t(dur)} analizados · {cal.get("pct_cuadros_calibrados", "–")}% de cuadros con cámara calibrada
· generado {datetime.now():%Y-%m-%d %H:%M}</p>
{leyenda(nombres, kits)}
<div class="kpis">{kpis}</div>
<h2>Momentos clave detectados</h2>
<div class="card scroll"><table><tr><th>Tiempo</th><th>Evento</th><th>Equipo</th><th>Posición (m)</th><th>Velocidad balón</th></tr>{filas_ev}</table></div>
{momentos_html}
<h2>Posesión segundo a segundo</h2>
<div class="card">{svg_barra_posesion(por_seg, nombres) or SIN_BALON}</div>
<h2>Altura del bloque (m desde el propio arco)</h2>
<div class="card">{svg_lineas(series_bloque, nombres, "metros")}</div>
<h2>Ancho del equipo (m)</h2>
<div class="card">{svg_lineas(series_ancho, nombres, "metros")}</div>
<h2>Posiciones medias y ocupación</h2>
<p class="nota">Cada equipo se dibuja atacando hacia la derecha. Pasa el cursor sobre los puntos.</p>
<div class="grid2"><div class="card">{svg_posiciones(svg, fis, direccion, nombres)}</div>
<div class="card">{svg_calor(svg, df.loc[df["equipo"] == "A", "x"] * direccion.get("A", 1), df.loc[df["equipo"] == "A", "y"] * direccion.get("A", 1), "A") if len(df) else ""}
<div class="nota">Ocupación — {e(nombres.get("A"))}</div></div>
<div class="card">{svg_calor(svg, df.loc[df["equipo"] == "B", "x"] * direccion.get("B", 1), df.loc[df["equipo"] == "B", "y"] * direccion.get("B", 1), "B") if len(df) else ""}
<div class="nota">Ocupación — {e(nombres.get("B"))}</div></div>
<div class="card">{red or '<p class="nota">Sin pases suficientes para una red.</p>'}<div class="nota">Red de pases — {e(nombres.get("A"))} (estimada)</div></div></div>
<h2>Físico (jugadores más visibles)</h2>
<div class="card scroll"><table><tr><th>Equipo</th><th>Pista</th><th>Rol</th><th>Seg. visible</th><th>Distancia (m)</th>
<th>Vel. máx (km/h)</th><th>m/min</th></tr>{top}</table></div>
{var_html}
{_validacion(salida, nombres)}
<h2>Calidad de los datos y límites</h2>
<div class="card"><ul class="nota">
<li>Cámara calibrada en {cal.get("pct_cuadros_calibrados", "–")}% de los cuadros; cortes de plano detectados: {cal.get("cortes_detectados", "–")}.</li>
<li>{e(nota_cobertura)}</li>
<li>Las "pistas" son tramos de seguimiento; un mismo jugador puede tener varias pistas tras salir de cuadro o en un corte.
Para identidad real hace falta lectura de dorsales.</li>
<li>Pases, toques y posesión se infieren de quién está a ≤2 m del balón; el balón en el aire se proyecta mal.</li>
<li>Procesado a {resumen_proceso.get("cuadros_por_segundo", "–")} cuadros/s · detector y parámetros en <code>resumen_proceso.json</code>.</li>
</ul></div>"""
    ruta = salida / "reporte.html"
    ruta.write_text(pagina(titulo, cuerpo, {"resumen": r}, extra_css=var_css), encoding="utf-8")
    return ruta


def reporte_video_partido(partido, ruta: Path, titulo: str | None = None) -> Path:
    """Reporte de un `Partido.video(...)` (reutiliza las métricas ya calculadas)."""
    calc = partido.metricas()
    info = partido.info
    proc = dict(info.get("proceso", {}))
    proc.setdefault("nombres_equipos", partido.nombres)
    carpeta = Path(info.get("carpeta", ruta.parent))
    out = generar_reporte(carpeta, partido.tracking, calc, proc, partido.cancha,
                          titulo or proc.get("titulo") or "Análisis del partido", proc.get("nota") or carpeta.name)
    if Path(ruta) != out:
        Path(ruta).write_text(out.read_text(encoding="utf-8"), encoding="utf-8")
    return Path(ruta)
