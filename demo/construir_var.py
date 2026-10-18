"""Página del VAR en el sitio (docs/var/index.html) con jugadas reales y la validación.

    python demo/construir_var.py

Usa: SkillCorner 1886347 (tracking), los clips sintéticos analizados en demo/salidas/ (video), la final
del Mundial 2022 con capturas 360 (StatsBomb) y demo/salidas/validacion_var.json (demo/validar_var.py).
"""
from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from analista import Partido
from analista.analisis import var as av
from analista.reportes.var import ACTIVOS
from analista.util import json_limpio
from analista.vision.var_imagen import afinar_con_cuerpo, clip_var, dibujar_var, leer_cuadro

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "docs"
SAL = RAIZ / "demo" / "salidas"
CLIPS = RAIZ / "demo" / "clips"
DEST = DOCS / "var"
REPO = "https://github.com/jazaineam1/analista-futbol"


def e(s) -> str:
    return html.escape(str(s))


def m(v, d=2) -> str:
    return "–" if v is None else f"{v:+.{d}f}".replace(".", ",").replace("+", "+").replace("-", "−")


def c(v, d=2) -> str:
    return "–" if v is None else f"{v:.{d}f}".replace(".", ",")


def version_web(mp4: Path):
    """Póster (un cuadro con las líneas ya trazadas) y WebM para navegadores sin H.264."""
    import subprocess

    from analista.vision.pipeline import ruta_ffmpeg
    cap = cv2.VideoCapture(str(mp4))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, n - 40))
    ok, f = cap.read()
    cap.release()
    if ok:
        if f.shape[1] > 1280:
            f = cv2.resize(f, (1280, int(f.shape[0] * 1280 / f.shape[1])), interpolation=cv2.INTER_AREA)
        cv2.imwrite(str(mp4.with_suffix(".jpg")), f, [cv2.IMWRITE_JPEG_QUALITY, 82])
    subprocess.run([ruta_ffmpeg(), "-y", "-loglevel", "error", "-i", str(mp4), "-vf", "scale='min(1280,iw)':-2",
                    "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36", "-deadline", "good", "-cpu-used", "5", "-row-mt", "1",
                    "-an", str(mp4.with_suffix(".webm"))], check=True)


def imagen(salida: str, clip: str, rev: dict, t: float, nombre: str) -> str | None:
    cam = pd.read_csv(SAL / salida / "camara.csv")
    proc = json.loads((SAL / salida / "resumen_proceso.json").read_text(encoding="utf-8"))
    i = int(np.argmin(np.abs(cam["t"].to_numpy() - t)))
    H = cam.iloc[i][[f"h{k}" for k in range(9)]].to_numpy(float).reshape(3, 3)
    frame = leer_cuadro(CLIPS / clip / "video.mp4", t)
    s = int(proc.get("ancho_proceso") or 1280) / frame.shape[1]
    img = dibujar_var(frame, H @ np.diag([s, s, 1.0]), rev, proc["cancha"]["largo"], proc["cancha"]["ancho"])
    if img.shape[1] > 1280:
        img = cv2.resize(img, (1280, int(img.shape[0] * 1280 / img.shape[1])), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(DEST / nombre), img, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return nombre


def casos() -> tuple[list[dict], dict]:
    datos, extra = [], {}
    # ---------------------------------------------------------------- 1. SkillCorner: el fuera de lugar pitado
    sc = Partido.skillcorner(1886347)
    rev = sc.var()
    nombres = sc.tracking.drop_duplicates("id").set_index("id")["nombre"].to_dict()
    pitado = min((r for r in rev.fueras_de_lugar if r.get("sancionado")), key=lambda r: abs(r["t"] - 3038.8))
    datos.append({"id": "sc", "titulo": "Fuera de lugar pitado · Auckland FC – Newcastle Jets, 50:38",
                  "fuente": "Tracking de SkillCorner (10 Hz), A-League 2024/25",
                  "texto": "El árbitro sancionó fuera de lugar a E. Adams. Con el tracking, en el instante del pase "
                           "Adams y el penúltimo defensor (N. Pijnaker) están en la misma línea: la diferencia está "
                           "muy por debajo del error de medición. Una jugada para revisar con más cámaras, no para "
                           "decidir con estos datos.",
                  "rep": av.para_repeticion(pitado, sc._posiciones, nombres, sc.cancha.largo, sc.cancha.ancho)})
    reb = next((r for r in rev.tiros if r.get("motivo") == "rebote" and r["veredicto"] == "fuera de lugar"), None)
    if reb:
        quien = nombres.get(reb.get("receptor"), "")
        datos.append({"id": "sc-rebote", "titulo": f"Fuera de lugar en el rebote · {int(reb['t'] // 60)}:{int(reb['t'] % 60):02d}",
                      "fuente": "Tracking de SkillCorner · remate y rebote",
                      "texto": f"Cuando su compañero remata, {quien or 'el atacante'} está por delante de la línea; el balón vuelve tras la "
                               f"atajada y lo juega él. {reb.get('explicacion', '')} Una atajada o un desvío no lo habilitan; un despeje "
                               "deliberado del rival, sí.",
                      "rep": av.para_repeticion(reb, sc._posiciones, nombres, sc.cancha.largo, sc.cancha.ancho)})
    inter = next((r for r in rev.tiros if r.get("motivo") == "interferencia" and r["veredicto"] == "revisar"), None)
    if inter:
        quien = nombres.get(inter.get("receptor"), "")
        datos.append({"id": "sc-interferencia", "titulo": f"Interferencia con el arquero · {int(inter['t'] // 60)}:{int(inter['t'] % 60):02d}",
                      "fuente": "Tracking de SkillCorner · remate",
                      "texto": f"{quien or 'Un atacante'} está adelantado en el remate y parado en la línea entre el balón y el arquero "
                               "(línea punteada). La app lo marca para revisar: si le impidió ver o jugar el balón, es fuera de lugar; "
                               "eso lo decide el árbitro.",
                      "rep": av.para_repeticion(inter, sc._posiciones, nombres, sc.cancha.largo, sc.cancha.ancho)})
    gol = rev.goles[0] if rev.goles else None
    if gol:
        datos.append({"id": "sc-gol", "titulo": "Revisión del gol de N. Moreno · 84:05",
                      "fuente": "Tracking de SkillCorner", "texto": gol.get("explicacion", ""),
                      "rep": av.para_repeticion(gol, sc._posiciones, nombres, sc.cancha.largo, sc.cancha.ancho)})
    # ---------------------------------------------------------------- 2. la misma jugada desde el video
    pc = Partido.video(SAL / "fuera_de_lugar_completo")
    pc.var()
    par = av.parametros_para("video", 25, pc.cancha.largo, pc.cancha.ancho, True)
    manual = av.revisar_instante(pc._posiciones, 9.80, "B", pc.direccion()["B"], par)
    centro, linea_centro = manual.get("margen_m"), manual.get("linea_x")
    video_fdl = CLIPS / "fuera_de_lugar_completo" / "video.mp4"
    proc_fdl = json.loads((SAL / "fuera_de_lugar_completo" / "resumen_proceso.json").read_text(encoding="utf-8"))
    afinar_con_cuerpo([manual], pc.tracking, pc.camara, video_fdl, int(proc_fdl.get("ancho_proceso") or 1920), 0.0, par.k_dudoso)
    cam_fdl = pc.camara.sort_values("t")

    def h_por_t(t):
        i = int(np.argmin(np.abs(cam_fdl["t"].to_numpy() - t)))
        H = cam_fdl.iloc[i][[f"h{k}" for k in range(9)]].to_numpy(float).reshape(3, 3)
        s_ = int(proc_fdl.get("ancho_proceso") or 1920) / 1920
        return H @ np.diag([s_, s_, 1.0])
    (DOCS / "media").mkdir(parents=True, exist_ok=True)
    (DEST / "fdl_video.mp4").unlink(missing_ok=True)
    clip_fdl = clip_var(video_fdl, h_por_t, manual, DOCS / "media" / "var_clip.mp4", pc.cancha.largo, pc.cancha.ancho)
    if clip_fdl is not None:
        version_web(clip_fdl)
    gj = pd.read_csv(CLIPS / "fuera_de_lugar_completo" / "gt_jugadores.csv").assign(detectado=True)
    gb = pd.read_csv(CLIPS / "fuera_de_lugar_completo" / "gt_balon.csv")
    exacto = av.ParametrosVar(pc.cancha.largo, pc.cancha.ancho, 0, 0, 0, 0, 0)
    verdad = av.revisar_instante(av.Posiciones(gj, gb), 9.80, "B", pc.direccion()["B"], exacto)
    extra["video_vs_verdad"] = {"video": centro, "cuerpo": manual.get("margen_m") if manual.get("margen_centro_m") is not None else None,
                                "sigma": manual.get("sigma_m"), "verdad": verdad.get("margen_m"),
                                "linea_video": linea_centro, "linea_verdad": verdad.get("linea_x")}
    nombres_v = {i: f"#{int(i)}" for i in pc.tracking["id"].dropna().unique()}
    datos.append({"id": "video", "titulo": "La misma jugada, desde el video (plano completo)",
                  "fuente": "Video analizado por la app · revisión manual en el cuadro del pase",
                  "texto": f"Eligiendo el cuadro del pase, la app traza la línea del penúltimo defensor y mide al atacante más "
                           f"adelantado: con el centro de los pies, {m(centro)} m (la verdad del clip: {m(verdad.get('margen_m'))} m)" +
                           (f"; con la parte del cuerpo más adelantada de cada uno, {m(manual.get('margen_m'))} m"
                            if manual.get("margen_centro_m") is not None else "") +
                           f". Con ±{c(manual.get('sigma_m'))} m de error de medición, la jugada es dudosa: está en línea.",
                  "foto": imagen("fuera_de_lugar_completo", "fuera_de_lugar_completo", manual, 9.80, "fdl_video.jpg"),
                  "clip": "../media/var_clip.mp4" if clip_fdl is not None else None,
                  "rep": av.para_repeticion(manual, pc._posiciones, nombres_v, pc.cancha.largo, pc.cancha.ancho)})
    # ---------------------------------------------------------------- 3. con la cámara de TV: no revisable
    pt = Partido.video(SAL / "fuera_de_lugar")
    rt = pt.var()
    norev = next((r for r in rt.fueras_de_lugar if r.get("fuera_de_camara")), None)
    if norev:
        shutil.copy(SAL / "fuera_de_lugar" / "var_00.jpg", DEST / "fdl_tv.jpg") if (SAL / "fuera_de_lugar" / "var_00.jpg").exists() else None
        datos.append({"id": "tv", "titulo": "La misma jugada con la cámara de TV",
                      "fuente": "Video de la transmisión (la cámara sigue el balón)",
                      "texto": "En el instante del pase la cámara sigue al pasador: el receptor y el último defensor están "
                               "fuera de la imagen. La app lo detecta y no inventa una decisión: la marca como no revisable.",
                      "foto": "fdl_tv.jpg" if (DEST / "fdl_tv.jpg").exists() else None,
                      "rep": av.para_repeticion(norev, pt._posiciones, {}, pt.cancha.largo, pt.cancha.ancho)})
    # ---------------------------------------------------------------- 4. gol en el video de TV
    pg = Partido.video(SAL / "tv")
    rg = pg.var()
    var_tv = json.loads((SAL / "tv" / "var.json").read_text(encoding="utf-8")) if (SAL / "tv" / "var.json").exists() else {}
    clips_tv = var_tv.get("clips", {})
    for t_txt, nombre in clips_tv.items():
        r = next((x for x in rg.fueras_de_lugar if abs(x["t"] - float(t_txt)) < 0.05), None)
        if r is None or not (SAL / "tv" / nombre).exists():
            continue
        shutil.copy(SAL / "tv" / nombre, DEST / "clip_tv.mp4")
        version_web(DEST / "clip_tv.mp4")
        datos.append({"id": "tv-clip", "titulo": "La repetición que arma la app sobre el video",
                      "fuente": "Video de la transmisión · clip automático",
                      "texto": f"Cámara lenta hasta el pase, imagen congelada con zoom, la línea del penúltimo defensor (azul) y la "
                               f"del receptor (rojo) trazadas en perspectiva sobre el césped, y el veredicto. {r.get('explicacion', '')}",
                      "clip": "clip_tv.mp4",
                      "rep": av.para_repeticion(r, pg._posiciones, {}, pg.cancha.largo, pg.cancha.ancho)})
        break
    if rg.goles:
        g = rg.goles[0]
        datos.append({"id": "tv-gol", "titulo": "Gol de M. Mata: línea de gol y jugada previa",
                      "fuente": "Video de la transmisión", "texto": g.get("explicacion", ""),
                      "foto": imagen("tv", "tv", g, float(g.get("t_max") or g["t"]), "gol_tv.jpg"),
                      "rep": av.para_repeticion(g, pg._posiciones, {}, pg.cancha.largo, pg.cancha.ancho)})
    # ---------------------------------------------------------------- 5. Mundial 2022, capturas 360
    from analista.var_partido import var_statsbomb
    fin = Partido.statsbomb(3869685, 43, 106)
    rf, _ = var_statsbomb(fin)
    for f in rf.faltas:
        oc = (f.get("ocasion") or {}).get("veredicto")
        if oc == "ocasión manifiesta" or f.get("motivo") == "mano":
            que = "Mano en el área" if f.get("motivo") == "mano" else "Falta: ¿ocasión manifiesta de gol?"
            datos.append({"id": f"wc-falta{f['minuto']}", "titulo": f"Final del Mundial 2022 · {f['minuto']}:{f['segundo']:02d} · {que}",
                          "fuente": "Evento de StatsBomb y captura 360 en el instante de la falta",
                          "texto": f"{f.get('explicacion', '')} Infractor: {f.get('jugador_nombre') or '–'}.",
                          "rep": av.para_repeticion({**f, "jugadores": f.get("jugadores", [])}, None, {}, fin.cancha.largo, fin.cancha.ancho)})
    pit = [r for r in rf.fueras_de_lugar if r.get("sancionado")]
    for r in sorted(pit, key=lambda r: -(r.get("margen_m") or -99))[:2]:
        datos.append({"id": f"wc{r['minuto']}", "titulo": f"Final del Mundial 2022 · {r['minuto']}:{r['segundo']:02d} · fuera de lugar pitado",
                      "fuente": "Captura 360 de StatsBomb (jugadores visibles en el instante del pase)",
                      "texto": f"{r.get('pasador_nombre')} → {r.get('receptor_nombre') or 'receptor estimado'}. {r.get('explicacion', '')}",
                      "rep": av.para_repeticion(r, None, r.get("nombres"), fin.cancha.largo, fin.cancha.ancho)})
    return datos, extra


def miles(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def fila_faltas(fa: dict) -> str:
    """Ocasión manifiesta según la app, contra la tarjeta que mostró el árbitro."""
    tab = fa.get("ocasion_vs_tarjeta") or {}
    if not tab:
        return ""
    om = tab.get("ocasión manifiesta", {})
    rojas = sum(om.get(k, 0) for k in ("Red Card", "Second Yellow"))
    amarillas = om.get("Yellow Card", 0)
    total_om = sum(om.values())
    pp = fa.get("penales_por_posicion") or {}
    pen, tl = pp.get("penal", {}), pp.get("tiro libre", {})
    fila_area = (f"<tr><td>StatsBomb: ¿dentro o fuera del área?</td><td class=\"num\">{pen.get('dentro del área', 0)} de {sum(pen.values())} penales dentro</td>"
                 f"<td>De {sum(pen.values())} penales pitados, la app ubica {pen.get('dentro del área', 0)} dentro del área y {pen.get('dudoso', 0)} sobre la línea; "
                 f"de {sum(tl.values())} tiros libres junto al área, {tl.get('dentro del área', 0)} dentro, {tl.get('dudoso', 0)} sobre la línea (dentro del error de "
                 f"±2 m de la ubicación del evento) y {tl.get('fuera del área', 0)} fuera. Ninguna contradicción con el árbitro.</td></tr>")
    return fila_area + (f"<tr><td>StatsBomb 360: ocasión manifiesta y tarjetas</td><td class=\"num\">{total_om} marcadas · {rojas} roja(s) · {amarillas} amarilla(s)</td>"
            f"<td>De {fa.get('revisadas', '–')} faltas junto al área o con posible ocasión, la app marca {total_om} como ocasión manifiesta "
            f"y el árbitro mostró {amarillas} amarilla(s) y {rojas} roja(s) en ellas (en el área la ocasión manifiesta se sanciona con penal "
            f"y amarilla si el infractor intentó jugar el balón). La única roja de la muestra no fue por ocasión manifiesta. Es un criterio "
            f"para revisar, no una decisión: la foto 360 no dice hacia dónde corría el atacante. {fa.get('manos', 0)} manos revisadas.</td></tr>")


def pagina(casos_: list[dict], val: dict, extra: dict) -> str:
    v, sk, sb, ti = val.get("video", {}), val.get("skillcorner", {}), val.get("statsbomb360", {}), val.get("tiros", {})
    geo = sk.get("geometria_vs_skillcorner", {})
    ts, tb = sk.get("tasas", {}), sb.get("tasas", {})
    bloques = []
    for k, cs in enumerate(casos_):
        foto = f'<figure class="var-foto"><img src="{e(cs["foto"])}" alt="{e(cs["titulo"])}: cuadro del video con las líneas" loading="lazy"></figure>' if cs.get("foto") else ""
        if cs.get("clip"):
            base = cs["clip"][:-4]
            foto = (f'<figure class="var-foto"><video autoplay muted loop playsinline controls preload="metadata" poster="{e(base)}.jpg" '
                    f'aria-label="{e(cs["titulo"])}"><source src="{e(base)}.webm" type="video/webm"><source src="{e(cs["clip"])}" type="video/mp4"></video>'
                    '<figcaption class="nota">La repetición que arma la app sobre el video.</figcaption></figure>')
        bloques.append(f"""<article class="caso" id="{e(cs['id'])}">
  <header><p class="eti">{e(cs['fuente'])}</p><h3>{e(cs['titulo'])}</h3><p>{e(cs['texto'])}</p></header>
  <div class="caso-grid"><div class="repeticion" data-caso="{k}"></div>{foto}</div>
</article>""")
    datos = json.dumps(json_limpio([cs["rep"] for cs in casos_]), ensure_ascii=False, allow_nan=False).replace("</", "<\\/")
    fila = lambda a, b, cc: f"<tr><td>{a}</td><td class=\"num\">{b}</td><td>{cc}</td></tr>"  # noqa: E731
    return f"""<!doctype html><html lang="es"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>VAR · Analista de fútbol</title>
<meta name="description" content="Revisión tipo VAR con datos: fuera de lugar en el instante del pase, goles, balón fuera y faltas en el área.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="../estilo.css"><link rel="stylesheet" href="repeticion.css">
<style>
.caso{{border-top:1px solid var(--line);padding:40px 0}}
.caso header{{max-width:820px;margin-bottom:18px}}.caso h3{{font-size:clamp(26px,3.2vw,34px);margin:4px 0 10px}}
.caso header p:last-child{{color:var(--ink2);margin:0}}
.caso-grid{{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:18px;align-items:start}}
.caso-grid:has(> :only-child){{grid-template-columns:minmax(0,1fr);max-width:860px}}
@media (max-width:900px){{.caso-grid{{grid-template-columns:minmax(0,1fr)}}}}
.var-foto{{margin:0}}.var-foto img,.var-foto video{{border-radius:10px;width:100%;display:block;background:#000}}
.reglas{{display:grid;grid-template-columns:repeat(auto-fit,minmax(240px,1fr));gap:14px;margin-top:20px}}
.reglas div{{background:var(--surface);border:1px solid var(--line);border-radius:12px;padding:16px}}
.reglas b{{display:block;font:700 20px/1.1 var(--f-display);text-transform:uppercase;margin-bottom:6px}}
.reglas p{{margin:0;color:var(--ink2);font-size:15px}}
td.num{{font-family:var(--f-mono);white-space:nowrap}}
</style></head><body>
<div class="barra"><div class="envoltura">
  <a class="marca" href="../index.html"><svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="var(--accent)"/><rect x="6" y="9" width="20" height="14" rx="1.5" fill="none" stroke="var(--accent-ink)" stroke-width="2"/><path d="M16 9v14" stroke="var(--accent-ink)" stroke-width="2"/><circle cx="16" cy="16" r="3" fill="none" stroke="var(--accent-ink)" stroke-width="2"/></svg>Analista</a>
  <nav aria-label="Sitio"><a href="../index.html">Inicio</a><a href="#casos">Jugadas</a><a href="#validacion">Validación</a><a href="../guia.html#var">Cómo se usa</a><a class="gh" href="{REPO}">GitHub</a></nav>
</div></div>
<main>
<section class="portada" style="padding-bottom:24px"><div class="envoltura angosta" style="max-width:1100px;display:block"><div>
  <p class="eti">Revisión tipo VAR con datos</p>
  <h1 style="font-size:clamp(44px,6.6vw,80px)">Fuera de lugar, medido</h1>
  <p class="lead">La app revisa cada pase en el instante en que sale del pie: traza la línea del penúltimo defensor (o del balón),
  mide al receptor y dice cuánto margen hay y con qué error. Hace lo mismo en cada remate (rebotes e interferencia con el arquero),
  revisa los goles (si el balón entró entero y la jugada previa), los balones que salieron sin que se parara el juego, y las faltas
  junto al área: dentro o fuera, manos y ocasión manifiesta de gol.</p>
  <div class="reglas">
    <div><b>El instante del pase</b><p>Se busca cuándo el balón sale del pie ajustando su trayectoria, entre dos cuadros si hace falta. Un atacante a 7 m/s se mueve 70 cm en una décima.</p></div>
    <div><b>Margen ± error</b><p>Error de posición de la fuente, de la cámara en ese cuadro, de la parte del cuerpo y del instante. Si el margen cae dentro de ±2σ, es dudoso.</p></div>
    <div><b>Lo que no se ve</b><p>Si el arquero no está en imagen se supone en su arco; si un defensor podría estar fuera de cámara, un "fuera de lugar" pasa a dudoso.</p></div>
    <div><b>Reglas de la IFAB</b><p>En línea es habilitado; no hay fuera de lugar en su campo, en saques de banda, de meta ni de esquina, ni si el pase no le llega.</p></div>
    <div><b>La parte del cuerpo</b><p>Con video se mide la parte del cuerpo más adelantada que puede jugar el balón (pie, rodilla, hombro, cabeza), no el brazo.</p></div>
    <div><b>Rebotes e interferencias</b><p>En cada remate: el adelantado que juega el rebote tras una atajada es fuera de lugar; el que tapa la visión del arquero queda para revisar.</p></div>
    <div><b>Ocasión manifiesta</b><p>En cada falta, los criterios de la Ley 12: distancia al arco, dirección, control del balón y defensores que podían llegar.</p></div>
  </div>
</div></div></section>
<section id="casos" style="padding-top:12px"><div class="envoltura">
{''.join(bloques)}
</div></section>
<section id="validacion"><div class="envoltura">
  <header><p class="eti">Medido, no supuesto</p><h2>Validación del VAR</h2>
  <p>Tres comprobaciones distintas: contra la verdad exacta (clips sintéticos), contra la geometría del proveedor y contra las
  decisiones de los árbitros.</p></header>
  <div class="tabla"><table>
  <tr><th>Medida</th><th>Resultado</th><th>Qué significa</th></tr>
  {fila("Video contra la verdad: decisiones", f"{v.get('decisiones_correctas', '–')} de {v.get('decididos', '–')} correctas · {v.get('dudosos', '–')} dudosas",
        f"Pases de los clips de video revisados por la app y comparados con las posiciones reales. Error mediano del margen {c(v.get('error_margen_mediana_m'))} m.")}
  {fila("Video: fuera de lugar pitado (plano completo)", f"{m(extra['video_vs_verdad']['video'])} m vs {m(extra['video_vs_verdad']['verdad'])} m",
        "Revisión manual en el cuadro del pase contra la verdad: la línea queda a centímetros.")}
  {fila("SkillCorner: misma línea que el proveedor", f"{c(geo.get('mismo_lado_pct'), 1)}% · {c(geo.get('diferencia_mediana_m'))} m",
        f"En {geo.get('pases', '–')} pases, el receptor queda del mismo lado de la línea que con las métricas de SkillCorner (diferencia mediana y p90 {c(geo.get('diferencia_p90_m'))} m).")}
  {fila("SkillCorner: pitados por el árbitro", f"{c((ts.get('sancionado') or {}).get('adelantado_pct'), 1)}% adelantados",
        f"De {sk.get('sancionados', '–')} fueras de lugar pitados en 20 partidos, el receptor buscado aparece adelantado en ese porcentaje (en pases completados: {c((ts.get('completado') or {}).get('adelantado_pct'), 1)}%). El tracking de TV estima a los jugadores fuera de cámara, justo los que suelen estar en juego en un pase largo.")}
  {fila("StatsBomb: remates y fueras de lugar pitados después", f"{c(ti.get('pitados_marcados_pct'), 0)}% marcados · {c(ti.get('no_pitados_marcados_pct'), 1)}% en el resto",
        f"De {ti.get('pitados_tras_remate', '–')} fueras de lugar pitados justo después de un remate (rebote o interferencia), la app marca el {c(ti.get('pitados_marcados_pct'), 0)}% (fuera de lugar, para revisar o dudoso) y el jugador sancionado estaba adelantado en la foto del tiro en el {c(ti.get('pitados_con_el_jugador_adelantado_pct'), 0)}%. En los otros {miles(ti.get('remates', 0) - (ti.get('pitados_tras_remate') or 0))} remates marca solo el {c(ti.get('no_pitados_marcados_pct'), 1)}%.")}
  {fila_faltas(sb.get("faltas", {}))}
  {fila("StatsBomb 360: pitados por el árbitro", f"{c((tb.get('sancionado') or {}).get('adelantado_pct'), 1)}% adelantados",
        f"De {sb.get('sancionados', '–')} fueras de lugar pitados (Mundial 2022 y Euro 2024), con el receptor estimado; en pases completados: {c((tb.get('completado') or {}).get('adelantado_pct'), 1)}%. La mayoría queda como dudosa: las capturas 360 no alcanzan la precisión de una decisión al centímetro.")}
  </table></div>
  <p class="nota" style="margin-top:10px">Reproducible con <code>python demo/validar_var.py skillcorner|statsbomb|tiros|video</code>. La app no reemplaza al VAR:
  mide con los datos disponibles y dice cuándo no alcanzan.</p>
</div></section>
</main>
<footer><div class="envoltura"><span>Analista de fútbol · <a href="{REPO}">GitHub</a> · <a href="../guia.html">Guía</a></span>
<span>Datos: SkillCorner Open Data (MIT) · Hudl StatsBomb Open Data</span></div></footer>
<script type="application/json" id="datos-casos">{datos}</script>
<script src="repeticion.js"></script>
<script>
(() => {{ const D = JSON.parse(document.getElementById('datos-casos').textContent);
document.querySelectorAll('.repeticion').forEach(el => Repeticion.montar(el, D[+el.dataset.caso], {{auto: true}})); }})();
</script>
</body></html>"""


def tabla_readme(val: dict, extra: dict) -> str:
    v, sk, sb, ti = val.get("video", {}), val.get("skillcorner", {}), val.get("statsbomb360", {}), val.get("tiros", {})
    geo = sk.get("geometria_vs_skillcorner", {})
    ts, tb = sk.get("tasas", {}), sb.get("tasas", {})
    vv = extra.get("video_vs_verdad", {})
    filas = [
        ("Video contra la verdad (clips sintéticos)", f"{v.get('decisiones_correctas', '–')} de {v.get('decididos', '–')} decisiones correctas, {v.get('dudosos', '–')} dudosas",
         f"error mediano del margen {c(v.get('error_margen_mediana_m'))} m"),
        ("Fuera de lugar pitado, desde el video", f"{m(vv.get('video'))} m (verdad {m(vv.get('verdad'))} m)", f"±{c(vv.get('sigma'))} m: dudoso, como corresponde a una jugada en línea"),
        ("Línea contra la de SkillCorner", f"{c(geo.get('mismo_lado_pct'), 1)} % del mismo lado", f"{geo.get('pases', '–')} pases, diferencia mediana {c(geo.get('diferencia_mediana_m'))} m"),
        ("Remates de StatsBomb con fuera de lugar pitado después", f"{c(ti.get('pitados_marcados_pct'), 0)} % marcados",
         f"{ti.get('pitados_tras_remate', '–')} casos; el sancionado estaba adelantado en la foto del tiro en el {c(ti.get('pitados_con_el_jugador_adelantado_pct'), 0)} %; en los otros {miles(ti.get('remates', 0) - (ti.get('pitados_tras_remate') or 0))} remates marca el {c(ti.get('no_pitados_marcados_pct'), 1)} %"),
        ("Faltas: dentro o fuera del área (StatsBomb)", f"{(sb.get('faltas', {}).get('penales_por_posicion', {}).get('penal', {}) or {}).get('dentro del área', '–')} de {sum((sb.get('faltas', {}).get('penales_por_posicion', {}).get('penal', {}) or {}).values()) or '–'} penales dentro",
         "ningún tiro libre ubicado dentro del área; los que caen sobre la línea salen como dudosos"),
        ("Pitados por el árbitro, SkillCorner", f"{c((ts.get('sancionado') or {}).get('adelantado_pct'), 1)} % adelantados",
         f"contra {c((ts.get('completado') or {}).get('adelantado_pct'), 1)} % en pases completados ({sk.get('sancionados', '–')} pitados)"),
        ("Pitados por el árbitro, StatsBomb 360", f"{c((tb.get('sancionado') or {}).get('adelantado_pct'), 1)} % adelantados",
         f"contra {c((tb.get('completado') or {}).get('adelantado_pct'), 1)} % en pases completados ({sb.get('sancionados', '–')} pitados)"),
    ]
    out = ["Validación del VAR (ver la [página del VAR](https://jazaineam1.github.io/analista-futbol/var/#validacion)):", "",
           "| Medida | Resultado | Detalle |", "|---|---|---|"]
    out += [f"| {a} | {b_} | {cc} |" for a, b_, cc in filas]
    return "\n".join(out)


def actualizar_readme(val: dict, extra: dict):
    ruta = RAIZ / "README.md"
    txt = ruta.read_text(encoding="utf-8")
    a, b_ = "<!-- VALIDACION_VAR -->", "<!-- /VALIDACION_VAR -->"
    if a in txt and b_ in txt:
        i, j = txt.index(a) + len(a), txt.index(b_)
        ruta.write_text(txt[:i] + "\n" + tabla_readme(val, extra) + "\n" + txt[j:], encoding="utf-8")


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    for f in ("repeticion.js", "repeticion.css"):
        shutil.copy(ACTIVOS / f, DEST / f)
    cs, extra = casos()
    val = json.loads((SAL / "validacion_var.json").read_text(encoding="utf-8")) if (SAL / "validacion_var.json").exists() else {}
    (DEST / "index.html").write_text(pagina(cs, val, extra), encoding="utf-8")
    actualizar_readme(val, extra)
    print(f"{DEST / 'index.html'}: {len(cs)} jugadas · video {extra.get('video_vs_verdad')}")


if __name__ == "__main__":
    main()
