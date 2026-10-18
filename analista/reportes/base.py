"""Piezas comunes de los reportes HTML: estilos (claro y oscuro), cancha SVG, gráficos y página."""
from __future__ import annotations

import html
import json
from datetime import datetime

import numpy as np
import pandas as pd

from ..cancha import Cancha
from ..util import json_limpio

# Identidad de equipo en gráficos: paleta categórica fija (no el color de la camiseta,
# que puede ser blanco o igual al fondo). El color de camiseta se muestra como muestra.
SERIE = {"A": ("#2a78d6", "#3987e5"), "B": ("#eb6834", "#d95926")}


def e(s) -> str:
    return html.escape("" if s is None else str(s))


def fmt_t(seg: float) -> str:
    seg = max(0, float(seg))
    return f"{int(seg // 60)}:{int(seg % 60):02d}"


def num(v, dec: int = 1, sufijo: str = "") -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "–"
    if isinstance(v, (int, np.integer)):
        return f"{v}{sufijo}"
    return f"{v:.{dec}f}".replace(".", ",") + sufijo


# ------------------------------------------------------------------ cancha SVG
class SvgCancha:
    def __init__(self, cancha: Cancha, ancho_px: int = 520, margen: float = 3.0):
        self.c, self.m = cancha, margen
        self.esc = ancho_px / (cancha.largo + 2 * margen)
        self.w = ancho_px
        self.h = int((cancha.ancho + 2 * margen) * self.esc)

    def p(self, x, y):
        return ((x + self.c.L2 + self.m) * self.esc, (self.c.W2 + self.m - y) * self.esc)

    def base(self) -> str:
        partes = [f'<rect x="0" y="0" width="{self.w}" height="{self.h}" rx="6" class="pitch-bg"/>']
        for pl in self.c.lineas(paso=0.8):
            pts = " ".join(f"{u:.1f},{v:.1f}" for u, v in (self.p(x, y) for x, y in pl))
            partes.append(f'<polyline points="{pts}" class="pitch-line"/>')
        return "".join(partes)

    def envolver(self, contenido: str, titulo: str) -> str:
        return (f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-label="{e(titulo)}" '
                f'class="pitch">{self.base()}{contenido}</svg>')


def svg_calor(svg: SvgCancha, x, y, equipo: str, bins=(21, 14), titulo: str = "Mapa de calor") -> str:
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() == 0:
        return ""
    H, xe, ye = np.histogram2d(x[ok], y[ok], bins=bins, range=[[-svg.c.L2, svg.c.L2], [-svg.c.W2, svg.c.W2]])
    H = H / max(H.max(), 1)
    out = []
    for i in range(bins[0]):
        for j in range(bins[1]):
            if H[i, j] <= 0.02:
                continue
            u0, v0 = svg.p(xe[i], ye[j + 1])
            u1, v1 = svg.p(xe[i + 1], ye[j])
            out.append(f'<rect x="{u0:.1f}" y="{v0:.1f}" width="{u1 - u0 - 1:.1f}" height="{v1 - v0 - 1:.1f}" '
                       f'rx="2" class="heat-{equipo}" fill-opacity="{0.12 + 0.88 * H[i, j]:.2f}" '
                       f'data-tip="{e(f"{100 * H[i, j]:.0f}% de la zona más ocupada")}"/>')
    return svg.envolver("".join(out), titulo)


def svg_red(svg: SvgCancha, nodos: pd.DataFrame, aristas: pd.DataFrame, equipo: str, etiqueta=None,
            titulo: str = "Red de pases") -> str:
    """Red de pases: nodos (jugador, x, y, pases) y aristas (de, a, n)."""
    if nodos is None or nodos.empty:
        return ""
    pos = nodos.set_index("jugador")
    out = []
    nmax = max(1, int(aristas["n"].max())) if len(aristas) else 1
    for r in aristas.itertuples():
        if r.de not in pos.index or r.a not in pos.index:
            continue
        u0, v0 = svg.p(pos.at[r.de, "x"], pos.at[r.de, "y"])
        u1, v1 = svg.p(pos.at[r.a, "x"], pos.at[r.a, "y"])
        out.append(f'<line x1="{u0:.1f}" y1="{v0:.1f}" x2="{u1:.1f}" y2="{v1:.1f}" class="edge-{equipo}" '
                   f'stroke-width="{1 + 5 * r.n / nmax:.1f}" data-tip="{e(f"{r.de} ↔ {r.a}: {r.n} pases")}"/>')
    pmax = max(1, int(pos["pases"].max()))
    for jugador, r in pos.iterrows():
        u, v = svg.p(r["x"], r["y"])
        rad = 5 + 7 * r["pases"] / pmax
        txt = etiqueta(jugador) if etiqueta else str(jugador)
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="{rad:.1f}" class="dot dot-{equipo}" '
                   f'data-tip="{e(f"{jugador}: {int(r.pases)} pases dados o recibidos")}"/>'
                   f'<text x="{u:.1f}" y="{v - rad - 3:.1f}" class="node-label" text-anchor="middle">{e(txt)}</text>')
    return svg.envolver("".join(out), titulo)


# ------------------------------------------------------------------ gráficos de tiempo
def svg_lineas(series: dict[str, pd.Series], nombres: dict, titulo_y: str, ancho=640, alto=220,
               escalon: bool = False, marcas: list[tuple[float, str, str]] | None = None, etiqueta_x=fmt_t) -> str:
    """Series en el tiempo (índice = x). `marcas`: [(x, equipo, texto)] líneas verticales (goles)."""
    series = {k: s.dropna() for k, s in series.items() if s is not None and len(s.dropna())}
    if not series:
        return '<p class="nota">Sin datos suficientes.</p>'
    pl, pr, pt, pb = 44, 12, 12, 30
    xs = np.concatenate([s.index.to_numpy(float) for s in series.values()])
    ys = np.concatenate([s.to_numpy(float) for s in series.values()])
    x0, x1 = float(xs.min()), float(xs.max()) if xs.max() > xs.min() else float(xs.min()) + 1
    y0, y1 = 0.0 if ys.min() >= 0 else float(ys.min()), float(ys.max()) * 1.08 or 1.0
    X = lambda v: pl + (v - x0) / (x1 - x0) * (ancho - pl - pr)  # noqa: E731
    Y = lambda v: alto - pb - (v - y0) / (y1 - y0) * (alto - pt - pb)  # noqa: E731
    out = []
    for tv in np.linspace(y0, y1, 4):
        out.append(f'<line x1="{pl}" x2="{ancho - pr}" y1="{Y(tv):.1f}" y2="{Y(tv):.1f}" class="grid"/>'
                   f'<text x="{pl - 6}" y="{Y(tv) + 4:.1f}" class="tick" text-anchor="end">{num(tv, 1 if y1 < 10 else 0)}</text>')
    for k, tv in enumerate(np.linspace(x0, x1, 5)):
        anc = "start" if k == 0 else ("end" if k == 4 else "middle")
        out.append(f'<text x="{X(tv):.1f}" y="{alto - 8}" class="tick" text-anchor="{anc}">{etiqueta_x(tv)}</text>')
    out.append(f'<text x="10" y="{pt + 2}" class="tick" transform="rotate(-90 10 {pt + 2})" text-anchor="end">{e(titulo_y)}</text>')
    for xv, eq, txt in marcas or []:
        if x0 <= xv <= x1:
            out.append(f'<line x1="{X(xv):.1f}" x2="{X(xv):.1f}" y1="{pt}" y2="{alto - pb}" class="marca marca-{eq}"/>'
                       f'<circle cx="{X(xv):.1f}" cy="{pt + 4}" r="4" class="dot-{eq}" data-tip="{e(txt)}"/>')
    etiquetas = []
    for eq, s in series.items():
        if escalon:
            pts = []
            prev = None
            for xv, yv in zip(s.index, s.to_numpy()):
                if prev is not None:
                    pts.append(f"{X(xv):.1f},{Y(prev):.1f}")
                pts.append(f"{X(xv):.1f},{Y(yv):.1f}")
                prev = yv
            pts = " ".join(pts)
        else:
            pts = " ".join(f"{X(xv):.1f},{Y(yv):.1f}" for xv, yv in zip(s.index, s.to_numpy()))
        out.append(f'<polyline points="{pts}" class="ln ln-{eq}"/>')
        etiquetas.append([Y(s.iloc[-1]) - 6, eq])
    # Etiquetas al final de cada línea, separadas para que no se pisen
    etiquetas.sort()
    for k in range(1, len(etiquetas)):
        if etiquetas[k][0] - etiquetas[k - 1][0] < 14:
            etiquetas[k][0] = etiquetas[k - 1][0] + 14
    for yv, eq in etiquetas:
        yv = min(max(yv, pt + 10), alto - pb - 4)
        out.append(f'<text x="{ancho - pr:.1f}" y="{yv:.1f}" class="direct direct-{eq}" text-anchor="end">'
                   f'{e(nombres.get(eq, eq))}</text>')
    # Zonas para tooltip con línea vertical
    paso = (ancho - pl - pr) / 60
    idx = np.linspace(x0, x1, 61)
    for k in range(60):
        xv = idx[k]
        partes = []
        for eq, s in series.items():
            j = np.searchsorted(s.index.to_numpy(float), xv, side="right") - 1
            if j >= 0:
                partes.append(f"{nombres.get(eq, eq)}: {num(float(s.iloc[j]), 2 if y1 < 10 else 0)}")
        tip = f"{etiqueta_x(xv)}<br>" + "<br>".join(e(p) for p in partes)
        out.append(f'<rect x="{X(xv):.1f}" y="{pt}" width="{paso:.1f}" height="{alto - pt - pb}" class="hit" '
                   f'data-x="{X(xv):.1f}" data-tip="{tip}"/>')
    out.append(f'<line class="xhair" y1="{pt}" y2="{alto - pb}"/>')
    return f'<svg viewBox="0 0 {ancho} {alto}" class="chart" role="img" aria-label="{e(titulo_y)}">{"".join(out)}</svg>'


def svg_barra_posesion(por_seg: dict, nombres: dict, ancho=640, alto=56) -> str:
    """Franja de posesión: {segundo: 'A'/'B'}."""
    if not por_seg:
        return ""
    t0, t1 = min(por_seg), max(por_seg)
    t1 = t1 if t1 > t0 else t0 + 1
    X = lambda t: 8 + (t - t0) / (t1 - t0) * (ancho - 16)  # noqa: E731
    w = (ancho - 16) / max(1, len(por_seg))
    out = [f'<rect x="8" y="10" width="{ancho - 16}" height="24" rx="4" class="track"/>']
    for s, eq in por_seg.items():
        out.append(f'<rect x="{X(s):.1f}" y="10" width="{max(w - 0.5, 1.2):.1f}" height="24" '
                   f'class="fill-{eq}" data-tip="{e(f"{fmt_t(s)} · {nombres.get(eq, eq)}")}"/>')
    for n, tv in enumerate(np.linspace(t0, t1, 5)):
        anc = "start" if n == 0 else ("end" if n == 4 else "middle")
        out.append(f'<text x="{X(tv):.1f}" y="{alto - 6}" class="tick" text-anchor="{anc}">{fmt_t(tv)}</text>')
    return f'<svg viewBox="0 0 {ancho} {alto}" class="chart" role="img" aria-label="Posesión en el tiempo">{"".join(out)}</svg>'


def svg_control(svg: SvgCancha, xs, ys, P, jugadores: pd.DataFrame, balon=None, titulo="Control de cancha") -> str:
    """Mapa de control de cancha (P = prob. de que A controle) con los jugadores encima."""
    out = []
    dx = (xs[1] - xs[0]) if len(xs) > 1 else 1
    dy = (ys[1] - ys[0]) if len(ys) > 1 else 1
    for i, yv in enumerate(ys):
        for j, xv in enumerate(xs):
            pa = P[i, j]
            if abs(pa - 0.5) < 0.08:
                continue
            eq = "A" if pa > 0.5 else "B"
            u0, v0 = svg.p(xv - dx / 2, yv + dy / 2)
            u1, v1 = svg.p(xv + dx / 2, yv - dy / 2)
            out.append(f'<rect x="{u0:.1f}" y="{v0:.1f}" width="{u1 - u0 + 0.3:.1f}" height="{v1 - v0 + 0.3:.1f}" '
                       f'class="heat-{eq}" fill-opacity="{0.55 * abs(pa - 0.5) * 2:.2f}"/>')
    for r in jugadores.itertuples():
        eq = r.equipo if r.equipo in ("A", "B") else "O"
        u, v = svg.p(r.x, r.y)
        if hasattr(r, "vx") and np.isfinite(getattr(r, "vx", np.nan)):
            u2, v2 = svg.p(r.x + r.vx, r.y + r.vy)
            out.append(f'<line x1="{u:.1f}" y1="{v:.1f}" x2="{u2:.1f}" y2="{v2:.1f}" class="vel"/>')
        nombre = getattr(r, "nombre", None) or f"pista {r.id}"
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="6" class="dot dot-{eq}" data-tip="{e(nombre)}"/>')
    if balon is not None:
        u, v = svg.p(*balon)
        out.append(f'<circle cx="{u:.1f}" cy="{v:.1f}" r="4" class="balon" data-tip="balón"/>')
    return svg.envolver("".join(out), titulo)


# ------------------------------------------------------------------ página
def kpi(etiqueta: str, a, b, nota: str = "") -> str:
    if all(str(v).startswith(("–", "0/0", "None")) for v in (a, b)):
        return (f'<div class="kpi"><div class="kpi-l">{e(etiqueta)}</div><div class="kpi-v sin">sin datos</div>'
                f'<div class="kpi-n">{e(nota)}</div></div>')
    return (f'<div class="kpi"><div class="kpi-l">{e(etiqueta)}</div><div class="kpi-v">'
            f'<span class="sw sw-A"></span>{a}<span class="sep">·</span><span class="sw sw-B"></span>{b}</div>'
            f'<div class="kpi-n">{e(nota)}</div></div>')


def css(extra: str = "") -> str:
    a1, a2 = SERIE["A"]
    b1, b2 = SERIE["B"]
    return f"""
:root{{--bg:#f6f5f2;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#8a8984;--line:#e4e2dc;
--a:{a1};--b:{b1};--o:#b8960c;--pitch:#eef1ec;--pitchline:#b9bfb4}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;
--muted:#8f8e86;--line:#2e2d2a;--a:{a2};--b:{b2};--o:#e0bf3a;--pitch:#1f2420;--pitchline:#47503f;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#121211;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#8f8e86;--line:#2e2d2a;
--a:{a2};--b:{b2};--o:#e0bf3a;--pitch:#1f2420;--pitchline:#47503f;color-scheme:dark}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif}}
main{{max-width:1080px;margin:0 auto;padding-block:24px 48px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
h1{{font-size:26px;margin:0 0 4px;text-wrap:balance}}h2{{font-size:17px;margin:32px 0 10px}}.sub{{color:var(--ink2);margin:0 0 18px}}
.card{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px;min-width:0}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px}}
.kpi{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:12px 14px}}
.kpi-l{{color:var(--ink2);font-size:13px}}.kpi-v{{font-size:20px;font-weight:650;margin:2px 0;display:flex;align-items:center;gap:6px;flex-wrap:wrap;font-variant-numeric:tabular-nums}}
.kpi-n{{color:var(--muted);font-size:12px}}.sin{{color:var(--muted);font-size:16px;font-weight:500}}.sep{{color:var(--muted);font-weight:400}}
.sw{{display:inline-block;width:10px;height:10px;border-radius:3px}}.sw-A{{background:var(--a)}}.sw-B{{background:var(--b)}}
.legend{{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:13px;margin:6px 0 14px}}
.legend span{{display:inline-flex;align-items:center;gap:6px}}.kit{{width:12px;height:12px;border-radius:50%;border:1px solid var(--muted)}}
.grid2{{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:12px}}
svg{{width:100%;height:auto;display:block}}.pitch-bg{{fill:var(--pitch)}}.pitch-line{{fill:none;stroke:var(--pitchline);stroke-width:1.2}}
.dot{{stroke:var(--surface);stroke-width:2}}.dot-A{{fill:var(--a)}}.dot-B{{fill:var(--b)}}.dot-O{{fill:var(--o)}}
.node-label{{font-size:9px;fill:var(--ink2);pointer-events:none}}
.balon{{fill:var(--ink);stroke:var(--surface);stroke-width:1.5}}.estela{{fill:none;stroke:var(--ink);stroke-width:1.6;stroke-dasharray:4 3;opacity:.7}}
.vel{{stroke:var(--ink2);stroke-width:1.2;opacity:.7}}
.heat-A{{fill:var(--a)}}.heat-B{{fill:var(--b)}}.edge-A{{stroke:var(--a);stroke-opacity:.5;stroke-linecap:round}}.edge-B{{stroke:var(--b);stroke-opacity:.5;stroke-linecap:round}}
.grid{{stroke:var(--line);stroke-width:1}}.tick{{font-size:11px;fill:var(--muted)}}.direct{{font-size:11px;fill:var(--ink2);paint-order:stroke;stroke:var(--surface);stroke-width:3px;font-weight:600}}
.direct-A{{fill:var(--a)}}.direct-B{{fill:var(--b)}}
.ln{{fill:none;stroke-width:2;stroke-linejoin:round}}.ln-A{{stroke:var(--a)}}.ln-B{{stroke:var(--b)}}
.marca{{stroke-width:1;stroke-dasharray:3 3}}.marca-A{{stroke:var(--a)}}.marca-B{{stroke:var(--b)}}
.hit{{fill:transparent}}.xhair{{stroke:var(--muted);stroke-dasharray:3 3;visibility:hidden}}
.track{{fill:var(--line)}}.fill-A{{fill:var(--a)}}.fill-B{{fill:var(--b)}}
.shot{{stroke:var(--surface);stroke-width:1.2;fill-opacity:.55}}.shot.gol{{fill-opacity:1;stroke:var(--ink);stroke-width:1.6}}
.shot-A{{fill:var(--a)}}.shot-B{{fill:var(--b)}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{text-align:left;padding:7px 8px;border-bottom:1px solid var(--line)}}
th{{color:var(--ink2);font-weight:600;font-size:13px}}td .sw{{margin-right:6px}}.scroll{{overflow-x:auto}}.num{{font-variant-numeric:tabular-nums;white-space:nowrap}}
.cmp td:nth-child(2),.cmp td:nth-child(4){{font-variant-numeric:tabular-nums;font-weight:600;width:18%}}
.cmp td:nth-child(2){{text-align:right}}.cmp td:nth-child(3){{text-align:center;color:var(--ink2)}}
#tip{{position:fixed;pointer-events:none;background:var(--ink);color:var(--surface);font-size:12px;padding:6px 8px;border-radius:6px;
opacity:0;transition:opacity .1s;z-index:9;max-width:260px}}
.nota{{color:var(--ink2);font-size:13px}}ul.nota{{padding-left:18px}}code{{font-size:12px}}
.momento h3{{font-size:14px;margin:0 0 6px}}.pie{{margin-top:36px;color:var(--muted);font-size:12px}}
{extra}"""


SCRIPT_TOOLTIP = """
(()=>{const tip=document.getElementById('tip');
document.addEventListener('mousemove',e=>{const t=e.target.closest('[data-tip]');
document.querySelectorAll('.xhair').forEach(x=>x.style.visibility='hidden');
if(!t){tip.style.opacity=0;return}tip.innerHTML=t.dataset.tip;tip.style.opacity=1;
const x=Math.min(e.clientX+12,innerWidth-tip.offsetWidth-8);tip.style.left=x+'px';tip.style.top=(e.clientY+12)+'px';
if(t.classList.contains('hit')){const xh=t.ownerSVGElement.querySelector('.xhair');xh.setAttribute('x1',t.dataset.x);
xh.setAttribute('x2',t.dataset.x);xh.style.visibility='visible'}});})();
"""


def pagina(titulo: str, cuerpo: str, datos: dict | None = None, extra_css: str = "") -> str:
    datos_json = json.dumps(json_limpio(datos or {}), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(titulo)}</title>
<style>{css(extra_css)}</style></head><body><main>
{cuerpo}
<p class="pie">Generado con Analista ({datetime.now():%Y-%m-%d %H:%M}) · github.com/jazaineam1/analista-futbol</p>
<script type="application/json" id="datos">{datos_json}</script>
</main><div id="tip"></div><script>{SCRIPT_TOOLTIP}</script></body></html>"""


def leyenda(nombres: dict, kits: dict | None = None) -> str:
    kits = kits or {}
    partes = []
    for k in ("A", "B"):
        kit = f' <span class="kit" style="background:{e(kits[k])}" title="color de camiseta"></span>' if kits.get(k) else ""
        partes.append(f'<span><span class="sw sw-{k}"></span>{e(nombres.get(k, k))}{kit}</span>')
    return f'<div class="legend">{"".join(partes)}</div>'
