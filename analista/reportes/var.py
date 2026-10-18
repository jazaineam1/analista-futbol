"""Sección "VAR" de los reportes: lista de jugadas revisadas y repetición 2D animada de cada una."""
from __future__ import annotations

import json
from pathlib import Path

from ..util import json_limpio
from .base import e, fmt_t

ACTIVOS = Path(__file__).resolve().parent / "activos"

TIPO = {"fuera de lugar": "Fuera de lugar", "línea de gol": "Gol", "revisión de gol": "Gol", "balón fuera": "Balón fuera",
        "falta en el área": "Falta", "posible penal": "Posible penal", "ocasión manifiesta": "Ocasión manifiesta"}
CLASE = {"fuera de lugar": "mal", "no entró": "mal", "salió": "mal", "revisar": "duda", "dentro del área": "mal",
         "habilitado": "ok", "gol": "ok", "gol válido": "ok", "fuera del área": "ok", "dudoso": "duda",
         "ocasión manifiesta": "mal", "sin intervención": ""}

CSS_VAR = """
.var-grid{display:grid;grid-template-columns:minmax(0,1.5fr) minmax(0,1fr);gap:14px;align-items:start}
@media (max-width:860px){.var-grid{grid-template-columns:1fr}}
.var-lista{list-style:none;margin:0;padding:0;display:grid;gap:6px;max-height:620px;overflow:auto}
.var-lista button{width:100%;text-align:left;display:grid;grid-template-columns:auto 1fr;gap:2px 10px;align-items:baseline;
  background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:9px 11px;cursor:pointer;color:var(--ink);font:inherit}
.var-lista button:hover{border-color:var(--muted)}
.var-lista button[aria-current="true"]{border-color:var(--ink);box-shadow:inset 0 0 0 1px var(--ink)}
.var-lista .min{font:600 13px ui-monospace,Menlo,Consolas,monospace;color:var(--ink2)}
.var-lista .tit{font-weight:600;font-size:14px}
.var-lista .txt{grid-column:2;font-size:13px;color:var(--ink2)}
.ver-chip{display:inline-block;font-size:11px;font-weight:700;letter-spacing:.04em;text-transform:uppercase;padding:2px 7px;border-radius:5px;
  background:var(--line);color:var(--ink2);margin-left:6px;vertical-align:1px}
.ver-chip.mal{background:#e0362b;color:#fff}.ver-chip.ok{background:#1d9a57;color:#fff}.ver-chip.duda{background:#c98500;color:#fff}
.var-filtros{display:flex;gap:6px;flex-wrap:wrap;margin:0 0 10px}
.var-filtros button{font:inherit;font-size:13px;border:1px solid var(--line);background:var(--surface);color:var(--ink2);border-radius:999px;padding:4px 11px;cursor:pointer}
.var-filtros button[aria-pressed="true"]{background:var(--ink);color:var(--surface);border-color:var(--ink)}
.var-foto{margin-top:10px}.var-foto img,.var-foto video{width:100%;height:auto;border-radius:8px;display:block;background:#000}
.var-foto .nota{margin:6px 0 0}
"""

SCRIPT_VAR = """
(()=>{const D=JSON.parse(document.getElementById('var-datos').textContent);const visor=document.getElementById('var-visor');
const lista=document.getElementById('var-lista');const foto=document.getElementById('var-foto');let actual=null,filtro='todo';
function abrir(i,auto){const r=D.items[i];if(actual)actual.destruir();
lista.querySelectorAll('button').forEach(b=>b.setAttribute('aria-current',b.dataset.i==i));
actual=Repeticion.montar(visor,r.datos,{auto:!!auto,inicio:auto?undefined:'evento'});
if(foto){foto.innerHTML=r.clip?`<video src="${r.clip}" autoplay muted loop playsinline controls aria-label="Repetición VAR sobre el video"></video><p class="nota">Repetición sobre el video: cámara lenta, imagen congelada y las líneas trazadas sobre el césped.</p>`:
r.foto?`<img src="${r.foto}" alt="Cuadro del video en el instante revisado, con las líneas del VAR">`:'';}}
function pintar(){lista.innerHTML='';D.items.forEach((r,i)=>{if(filtro!=='todo'&&r.grupo!==filtro)return;const li=document.createElement('li');
li.innerHTML=`<button type="button" data-i="${i}"><span class="min">${r.min}</span><span class="tit">${r.titulo}<span class="ver-chip ${r.clase}">${r.veredicto}</span></span><span class="txt">${r.texto}</span></button>`;
li.firstChild.onclick=()=>abrir(i,false);lista.appendChild(li)});}
document.querySelectorAll('.var-filtros button').forEach(b=>b.onclick=()=>{filtro=b.dataset.f;
document.querySelectorAll('.var-filtros button').forEach(x=>x.setAttribute('aria-pressed',x===b));pintar();});
pintar();if(D.items.length)abrir(D.inicial||0,true);})();
"""


def activos() -> tuple[str, str]:
    return ((ACTIVOS / "repeticion.css").read_text(encoding="utf-8"), (ACTIVOS / "repeticion.js").read_text(encoding="utf-8"))


def _grupo(r: dict) -> str:
    t = r.get("tipo")
    if t == "fuera de lugar":
        return "fuera"
    if t in ("línea de gol", "revisión de gol"):
        return "goles"
    if t in ("falta en el área", "posible penal", "ocasión manifiesta"):
        return "faltas"
    return "otras"


def items_var(rev, pos, nombres_jug: dict | None, nombres_eq: dict, largo: float, ancho: float,
              maximo: int = 40, fotos: dict | None = None, clips: dict | None = None) -> list[dict]:
    """Jugadas destacadas con sus datos de repetición (para el reporte y el panel)."""
    from ..analisis.var import para_repeticion
    destacadas = rev.destacadas()
    # Prioridad si hay demasiadas: goles, fueras de lugar, sancionadas, dudosas, el resto
    def prioridad(r):
        v = r.get("veredicto")
        return (0 if r.get("tipo") in ("línea de gol", "revisión de gol") else 1 if v in ("fuera de lugar", "ocasión manifiesta") else
                2 if r.get("sancionado") or r.get("en_gol") or r.get("tipo") == "falta en el área" else
                3 if v in ("dudoso", "revisar") else 4)
    if len(destacadas) > maximo:
        destacadas = sorted(sorted(destacadas, key=prioridad)[:maximo], key=lambda r: r.get("t") or 0)
    items = []
    from ..analisis.var import id_texto
    nj = {id_texto(k): v for k, v in (nombres_jug or {}).items()}
    for k, r in enumerate(destacadas):
        tipo = r.get("tipo")
        eq = r.get("equipo")
        nj_r = {**nj, **{id_texto(k): v for k, v in (r.get("nombres") or {}).items() if v}}
        quien = nombres_eq.get(eq, "") if eq in ("A", "B") else ""
        if tipo == "fuera de lugar" and r.get("motivo") in ("rebote", "interferencia", "sin intervención"):
            rec = nj_r.get(id_texto(r.get("receptor")), "")
            que = {"rebote": "Rebote en fuera de lugar", "interferencia": "Interferencia con el arquero",
                   "sin intervención": "Adelantados en el remate"}[r["motivo"]]
            titulo = f"{que} · {quien}" + (f" · {rec}" if rec else "")
        elif tipo == "fuera de lugar":
            rec = nj_r.get(id_texto(r.get("receptor")), "")
            titulo = f"Fuera de lugar · {quien}" + (f" → {rec}" if rec else "")
        elif tipo == "falta en el área":
            que = "Mano" if r.get("motivo") == "mano" else "Falta"
            oc = (r.get("ocasion") or {}).get("veredicto")
            titulo = f"{que} a {quien}" if quien else que
            if oc == "ocasión manifiesta":
                titulo += " · ocasión manifiesta"
            if r.get("tarjeta"):
                titulo += f" · {r['tarjeta'].lower().replace('yellow card', 'amarilla').replace('red card', 'roja').replace('second yellow', '2ª amarilla')}"
        elif tipo == "posible penal":
            titulo = f"Posible penal · {quien}" if quien else "Posible penal"
        elif tipo in ("línea de gol", "revisión de gol"):
            titulo = f"Gol de {quien}" if quien else "Gol"
        else:
            titulo = TIPO.get(tipo, tipo or "")
        ver = r.get("decision_var") if tipo == "línea de gol" and r.get("decision_var") else r.get("veredicto", "")
        if r.get("sancionado"):
            titulo += " · pitado"
        t = float(r.get("t") or 0)
        minuto = f"{int(r['minuto'])}:{int(r.get('segundo', 0)):02d}" if r.get("minuto") is not None else fmt_t(t)
        items.append({"min": minuto, "titulo": e(titulo), "veredicto": e(ver.split(":")[0]),
                      "clase": CLASE.get(ver.split(":")[0], ""), "texto": e(r.get("explicacion", "")), "grupo": _grupo(r),
                      "datos": para_repeticion(r, pos, nj_r, largo, ancho),
                      "foto": (fotos or {}).get(round(t, 2)), "clip": (clips or {}).get(round(t, 2))})
    return items


def seccion_var(rev, pos, nombres_jug: dict | None, nombres_eq: dict, largo: float, ancho: float,
                fuente: str, fotos: dict | None = None, clips: dict | None = None) -> tuple[str, str]:
    """HTML de la sección y CSS extra. Incluye el componente de repetición (sin archivos externos)."""
    css_rep, js_rep = activos()
    items = items_var(rev, pos, nombres_jug, nombres_eq, largo, ancho, fotos=fotos, clips=clips)
    res = rev.resumen()
    kpis = (f'<div class="kpis">'
            f'<div class="kpi"><div class="kpi-l">Pases revisados</div><div class="kpi-v">{res["pases_revisados"]}</div>'
            f'<div class="kpi-n">receptor en campo rival o cerca</div></div>'
            f'<div class="kpi"><div class="kpi-l">Fuera de lugar</div><div class="kpi-v">{res["fuera_de_lugar"]}</div>'
            f'<div class="kpi-n">receptor adelantado fuera del error</div></div>'
            f'<div class="kpi"><div class="kpi-l">Dudosos</div><div class="kpi-v">{res["dudosos"]}</div>'
            f'<div class="kpi-n">margen dentro del error de medición</div></div>'
            f'<div class="kpi"><div class="kpi-l">Goles revisados</div><div class="kpi-v">{res["goles_revisados"]}</div>'
            f'<div class="kpi-n">línea de gol y jugada previa</div></div>'
            f'<div class="kpi"><div class="kpi-l">Remates revisados</div><div class="kpi-v">{res.get("tiros_revisados", 0)}</div>'
            f'<div class="kpi-n">{res.get("rebotes_fuera_de_lugar", 0)} rebote(s) en fuera de lugar · {res.get("interferencias", 0)} interferencia(s)</div></div>'
            f'<div class="kpi"><div class="kpi-l">Faltas junto al área</div><div class="kpi-v">{res["faltas_revisadas"]}</div>'
            f'<div class="kpi-n">{res.get("ocasiones_manifiestas", 0)} ocasión(es) manifiesta(s) · {res.get("manos", 0)} mano(s)</div></div></div>')
    if not items:
        cuerpo = f'<h2>VAR</h2>{kpis}<p class="nota">No hay jugadas para revisar.</p>'
        return cuerpo, ""
    datos = json.dumps(json_limpio({"items": items, "inicial": next(
        (i for i, it in enumerate(items) if it["grupo"] == "fuera" and it["clase"] in ("mal", "duda")), 0)}),
        ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    hay_fotos = any(it.get("foto") or it.get("clip") for it in items)
    cuerpo = f"""<h2>VAR: jugadas revisadas</h2>
{kpis}
<p class="nota" style="margin:10px 0">Fuera de lugar medido en el instante del pase (y del remate, para rebotes e interferencias) con la
línea del penúltimo defensor o del balón. El margen lleva su incertidumbre: si cae dentro de ±2σ la jugada es dudosa. Las faltas se
ubican respecto del área y se mide si cortaron una ocasión manifiesta de gol. Fuente: {e(fuente)}.</p>
<div class="var-grid">
  <div class="card"><div id="var-visor"></div>{'<div id="var-foto" class="var-foto"></div>' if hay_fotos else ''}</div>
  <div><div class="var-filtros" role="group" aria-label="Filtrar jugadas">
    <button type="button" data-f="todo" aria-pressed="true">Todas ({len(items)})</button>
    <button type="button" data-f="fuera" aria-pressed="false">Fuera de lugar</button>
    <button type="button" data-f="goles" aria-pressed="false">Goles</button>
    <button type="button" data-f="faltas" aria-pressed="false">Faltas y penales</button>
    <button type="button" data-f="otras" aria-pressed="false">Balón fuera</button></div>
  <ul class="var-lista" id="var-lista"></ul></div>
</div>
<script type="application/json" id="var-datos">{datos}</script>
<script>{js_rep}</script><script>{SCRIPT_VAR}</script>"""
    return cuerpo, css_rep + CSS_VAR
