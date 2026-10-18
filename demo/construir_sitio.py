"""Arma el sitio de documentación (docs/) con las cifras reales de la validación.

    python demo/construir_sitio.py              # docs/index.html + reportes de video + miniaturas
    python demo/construir_sitio.py --artefacto  # además, una copia sin <html>/<head> para publicar como artefacto

Lee demo/salidas/*/evaluacion.json (de demo/evaluar.py) y analista/modelos/xg.json.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
DOCS = RAIZ / "docs"
REPO = "https://github.com/jazaineam1/analista-futbol"


def e(s) -> str:
    return html.escape(str(s))


def pct(v, dec=1):
    return "–" if v is None else f"{100 * v:.{dec}f}%".replace(".", ",")


def num(v, dec=2, suf=""):
    return "–" if v is None else f"{v:.{dec}f}{suf}".replace(".", ",")


def miles(n) -> str:
    return f"{int(n):,}".replace(",", ".")


def leer(ruta: Path) -> dict:
    return json.loads(ruta.read_text(encoding="utf-8")) if ruta.exists() else {}


REPORTES = [
    ("reportes/statsbomb_3869685.html", "media/rep_final.jpg", "Eventos · StatsBomb", "Argentina 3–3 Francia",
     "Final del Mundial 2022: xG, tiros, redes de pases y presión."),
    ("reportes/skillcorner_1886347.html", "media/rep_aleague.jpg", "Tracking · SkillCorner", "Auckland FC 2–0 Newcastle Jets",
     "A-League 2024/25: posesión, bloque, formación, control de cancha y físico."),
    ("reportes/video_tv.html", "media/rep_tv.jpg", "Video · cámara de TV", "Dos goles de Auckland",
     "Lo que la app sacó de 38 s de transmisión, con la validación contra la verdad."),
    ("reportes/video_completo.html", "media/rep_completo.jpg", "Video · plano completo", "El gol de N. Moreno",
     "Cámara fija que ve toda la cancha: los 22 jugadores a la vez."),
    ("reportes/video_real.html", "media/rep_real.jpg", "Video · partido real", "SoccerTrack, Tsukuba",
     "Cámara gran angular de un partido universitario real."),
]


def filas_validacion(c: dict, t: dict) -> str:
    def g(d, *ks):
        for k in ks:
            d = (d or {}).get(k)
        return d

    def pos(d):
        p = g(d, "posesion") or {}
        r, s = p.get("real") or {}, p.get("estimada") or {}
        if not r or not s:
            return "–"
        return f'{num(s.get("A"), 0)}% / {num(r.get("A"), 0)}%'

    def goles(d):
        go = g(d, "goles") or {}
        return f'{go.get("aciertos", 0)} de {len(go.get("reales", []))}'

    filas = [
        ("Jugadores detectados", pct(g(c, "jugadores", "recall")), pct(g(t, "jugadores", "recall")),
         "de los que se ven en cada cuadro"),
        ("Detecciones correctas", pct(g(c, "jugadores", "precision")), pct(g(t, "jugadores", "precision")),
         "lo detectado que era un jugador real, a ≤2 m"),
        ("Error de posición", f'{num(g(c, "jugadores", "error_mediana_m"))} m', f'{num(g(t, "jugadores", "error_mediana_m"))} m',
         f'mediana; el 90% bajo {num(g(c, "jugadores", "error_p90_m"))} m y {num(g(t, "jugadores", "error_p90_m"))} m'),
        ("Equipo correcto", pct(g(c, "jugadores", "equipo_correcto")), pct(g(t, "jugadores", "equipo_correcto")),
         "jugadores de campo, por el color de la camiseta"),
        ("Identidad estable", pct(g(c, "jugadores", "pureza_identidad")), pct(g(t, "jugadores", "pureza_identidad")),
         "una pista sigue siempre al mismo jugador"),
        ("Balón ubicado", f'{num(g(c, "balon", "detectado_pct"), 0)}%', f'{num(g(t, "balon", "detectado_pct"), 0)}%',
         "cuadros con balón visible en que la app lo ubica a ≤2 m"),
        ("Cámara calibrada", f'{num(g(c, "camara", "calibrado_pct"), 0)}%', f'{num(g(t, "camara", "calibrado_pct"), 0)}%',
         f'error mediano en la cancha: {num(g(c, "camara", "error_mediana_m"))} m y {num(g(t, "camara", "error_mediana_m"))} m'),
        ("Posesión de Auckland", pos(c), pos(t), "estimada / real, en los cuadros analizados"),
        ("Goles detectados", goles(c), goles(t), "el balón cruza la línea entre los postes (±2 s)"),
        ("Velocidad", f'{num(g(c, "velocidad", "cuadros_por_segundo"), 1)} cuadros/s', f'{num(g(t, "velocidad", "cuadros_por_segundo"), 1)} cuadros/s',
         f'CPU de 2 núcleos sin GPU, a {g(c, "velocidad", "ancho_proceso")} y {g(t, "velocidad", "ancho_proceso")} px'),
    ]
    return "".join(f"<tr><td>{e(a)}</td><td class='n'>{e(b)}</td><td class='n'>{e(cc)}</td><td class='nota'>{e(d)}</td></tr>"
                   for a, b, cc, d in filas)


def pagina(evals: dict, xg: dict, artefacto: bool) -> str:
    c, t = evals.get("completo", {}), evals.get("tv", {})
    ctx = xg["modelos"]["contexto"]["prueba"]
    bas = xg["modelos"]["basico"]["prueba"]
    ref = xg["referencia_statsbomb"]
    rec_tv = pct((t.get("jugadores") or {}).get("recall"), 0)
    _p = t.get("posesion") or {}
    pos_tv = (f'{num((_p.get("estimada") or {}).get("A"), 0)}% estimado frente a {num((_p.get("real") or {}).get("A"), 0)}% real'
              if _p.get("estimada") and _p.get("real") else "ver la tabla de validación")
    err_c = num((c.get("jugadores") or {}).get("error_mediana_m"), 2)
    tarjetas = "".join(
        f'<a class="reporte" href="{r}"><img src="{img}" alt="Vista previa del reporte {e(tit)}" loading="lazy">'
        f'<div><span class="tipo">{e(tipo)}</span><strong>{e(tit)}</strong><span>{e(desc)}</span></div></a>'
        for r, img, tipo, tit, desc in REPORTES if (DOCS / r).exists())
    cabeza = "" if artefacto else (
        '<!doctype html>\n<html lang="es">\n<head>\n<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">\n')
    fin_cabeza = "" if artefacto else "</head>\n<body>\n"
    pie = "" if artefacto else "</body>\n</html>\n"
    return f"""{cabeza}<title>Analista de fútbol</title>
<meta name="claude:viewport-fit-sides" content="cover">
<meta name="description" content="Herramienta de análisis de datos de fútbol: lee un partido desde el video y analiza eventos y tracking abiertos.">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<link rel="stylesheet" href="estilo.css">
{fin_cabeza}<div class="barra"><div class="envoltura">
  <a class="marca" href="index.html"><svg viewBox="0 0 32 32" aria-hidden="true"><rect width="32" height="32" rx="7" fill="var(--accent)"/><rect x="6" y="9" width="20" height="14" rx="1.5" fill="none" stroke="var(--accent-ink)" stroke-width="2"/><path d="M16 9v14" stroke="var(--accent-ink)" stroke-width="2"/><circle cx="16" cy="16" r="3" fill="none" stroke="var(--accent-ink)" stroke-width="2"/></svg>Analista</a>
  <nav aria-label="Sitio"><a href="#fuentes">Qué analiza</a><a href="#video">Video</a><a href="#var">VAR</a><a href="#reportes">Reportes</a><a href="#validacion">Validación</a><a href="#instalar">Instalar</a><a href="guia.html">Guía</a><a class="gh" href="{REPO}">GitHub</a></nav>
</div></div>

<main>
<section class="portada"><div class="envoltura">
  <div>
    <p class="eti">Análisis de datos de fútbol · código abierto</p>
    <h1>Del partido a los datos</h1>
    <p class="lead">Analista lee un partido desde el video, sea la señal del televisor, la página donde lo ves en vivo o una
    grabación, y ubica a cada jugador y el balón en metros. Con eso, y con datos abiertos de eventos y tracking, calcula xG,
    posesión, presión, formación y control de cancha, revisa las jugadas como un VAR y entrega reportes listos para compartir.</p>
    <div class="acciones">
      <a class="boton pri" href="#instalar">Instalar</a>
      <a class="boton" href="{REPO}">Código en GitHub</a>
      <a class="boton" href="var/index.html">Ver el VAR</a>
    </div>
    <div class="datos-portada"><span><b>{rec_tv}</b> de jugadores detectados con cámara de TV</span>
      <span><b>{err_c} m</b> de error con cámara fija</span><span><b>AUC {num(ctx["auc"], 3)}</b> del modelo xG</span></div>
  </div>
  <figure style="margin:0">
    <div class="pantalla"><video autoplay muted loop playsinline preload="metadata" poster="media/portada.jpg" aria-label="Demostración del panel">
      <source src="media/demo.webm" type="video/webm"><source src="media/demo.mp4" type="video/mp4"></video></div>
    <figcaption class="nota" style="margin-top:8px">El panel: datos de la final del Mundial 2022, tracking de la A-League y el análisis de una transmisión.
    <a href="media/demo.mp4">Abrir el video en grande</a>.</figcaption>
  </figure>
</div></section>

<section id="fuentes"><div class="envoltura">
  <header><p class="eti">Tres fuentes, un mismo esquema</p><h2>Qué analiza</h2>
  <p>Todo llega a las mismas tablas, en metros y con el origen en el centro de la cancha. El mismo código sirve para un partido de
  StatsBomb, uno de SkillCorner o tu propio video.</p></header>
  <div class="fuentes">
    <article class="fuente"><span class="tipo">Tu video</span><h3>Televisor, página en vivo o archivo</h3>
      <ul><li>La capturadora HDMI del televisor, la zona de la pantalla donde ves el partido o un video que subes</li>
      <li>Jugadores y balón con YOLO, equipos por color de camiseta</li>
      <li>Calibración con 4 clics y seguimiento del paneo y el zoom</li><li>Posesión, pases, presión, bloque, físico, goles y VAR</li></ul>
      <a class="ir" href="#video">Cómo funciona →</a></article>
    <article class="fuente"><span class="tipo">Eventos · StatsBomb Open Data</span><h3>Cada pase, tiro y presión</h3>
      <ul><li>Casi 4.000 partidos: Mundiales, Euro, Copa América, ligas</li><li>xG de StatsBomb y de un modelo propio validado</li>
      <li>Flujo de xG, mapa de tiros, redes de pases</li><li>PPDA, field tilt, pases progresivos, recuperaciones altas</li></ul>
      <a class="ir" href="reportes/statsbomb_3869685.html">Reporte de la final 2022 →</a></article>
    <article class="fuente"><span class="tipo">Tracking · SkillCorner Open Data</span><h3>Los 22 jugadores, 10 veces por segundo</h3>
      <ul><li>20 partidos de la A-League 2024/25 (licencia MIT)</li><li>Formación sin balón y altura del bloque minuto a minuto</li>
      <li>Control de cancha en cada gol</li><li>Distancia, velocidad máxima y sprints por jugador</li></ul>
      <a class="ir" href="reportes/skillcorner_1886347.html">Reporte de Auckland – Newcastle →</a></article>
  </div>
</div></section>

<section id="video"><div class="envoltura">
  <header><p class="eti">Con video</p><h2>Cuatro pasos en el panel</h2>
  <p>Se instala en tu PC y se usa desde el navegador. No sube nada a internet.</p></header>
  <div class="flujo">
    <div class="paso"><b>1</b><h3>Fuente</h3><p>Sube un video, marca en la pantalla la página donde ves el partido en vivo o elige la capturadora HDMI del televisor.</p></div>
    <div class="paso"><b>2</b><h3>Calibrar</h3><p>Cuatro o más clics en la imagen y su punto en la cancha. Las líneas rojas muestran si calza.</p></div>
    <div class="paso"><b>3</b><h3>Analizar</h3><p>Ves el cuadro anotado y la posesión en vivo mientras procesa.</p></div>
    <div class="paso"><b>4</b><h3>Resultados</h3><p>Video con minimapa, reporte HTML y tablas CSV de posiciones, pases y eventos.</p></div>
  </div>
  <div class="detecta">
    <img src="media/etapas.jpg" alt="La etapa de detección del balón, con los candidatos y el elegido" loading="lazy">
    <div><p class="eti">Cómo detecta</p><h3 style="font-size:30px">Las 10 etapas, sobre el cuadro que elijas</h3>
      <ol class="etapas"><li>1 cuadro</li><li>2 césped</li><li>3 líneas</li><li>4 calibración</li><li>5 personas</li>
      <li>6 equipos</li><li>7 balón</li><li>8 a metros</li><li>9 seguimiento</li><li>10 táctica</li></ol>
      <p style="color:var(--ink2)">El panel muestra qué hizo la app en cada etapa y por qué. Sirve para confiar en el resultado y para entender un error.</p>
      <div class="acciones" style="margin-top:16px"><a class="boton" href="deteccion/index.html">Recorrer las etapas</a></div></div>
  </div>
</div></section>

<section id="var"><div class="envoltura">
  <div class="dos">
    <div class="texto">
      <p class="eti">Revisión tipo VAR</p><h2>Fuera de lugar, medido</h2>
      <p style="color:var(--ink2)">En cada pase y en cada remate la app busca el instante exacto en que el balón sale del pie, traza la línea
      del penúltimo defensor y mide al atacante con la parte del cuerpo más adelantada. Cada decisión lleva su margen y su error: si no
      alcanza, la jugada sale como dudosa.</p>
      <ul class="lista-var"><li>Fuera de lugar en pases, rebotes e interferencia con el arquero</li><li>Línea de gol y la jugada previa de cada gol</li>
      <li>Balones que salieron sin que se parara el juego</li><li>Faltas dentro o fuera del área, manos y ocasión manifiesta de gol</li>
      <li>Revisión manual cuadro a cuadro sobre tu video</li></ul>
      <div class="acciones" style="margin-top:18px"><a class="boton pri" href="var/index.html">Ver las jugadas revisadas</a></div>
    </div>
    <figure style="margin:0"><div class="pantalla"><video autoplay muted loop playsinline preload="metadata" poster="media/var_clip.jpg" aria-label="Repetición VAR sobre el video: imagen congelada y líneas del fuera de lugar">
      <source src="media/var_clip.webm" type="video/webm"><source src="media/var_clip.mp4" type="video/mp4"></video></div>
      <figcaption class="nota" style="margin-top:8px">La repetición que arma la app sobre el video: cámara lenta, imagen congelada, líneas trazadas sobre el césped y el veredicto con su margen.</figcaption></figure>
  </div>
</div></section>

<section id="reportes"><div class="envoltura">
  <header><p class="eti">Ejemplos</p><h2>Reportes</h2><p>Archivos HTML autocontenidos, con tema claro y oscuro, que se abren en cualquier navegador.</p></header>
  <div class="reportes">{tarjetas}</div>
</div></section>

<section id="validacion"><div class="envoltura">
  <header><p class="eti">Medido, no supuesto</p><h2>Validación</h2>
  <p>Para medir el error hace falta saber la verdad de cada cuadro. Por eso la app se prueba con transmisiones generadas a partir del
  tracking real de SkillCorner: los jugadores son recortes de video real, la cámara de TV repite el paneo y el zoom de la transmisión
  original y cada toma se calibra con clics simulados con 2 px de error (el clip de TV tiene dos tomas, separadas por una repetición).</p></header>
  <div class="tabla"><table>
    <tr><th>Medida</th><th>Plano completo<br><span class="nota">cámara fija, 1920 px, 28 s</span></th><th>Cámara de TV<br><span class="nota">paneo y zoom, 1280 px, 38 s</span></th><th>Qué significa</th></tr>
    {filas_validacion(c, t)}
  </table></div>
  <p class="nota" style="margin-top:10px">{una_toma(evals.get("tv_una", {}))}Reproducible con <code>demo/generar_transmision.py</code> y
  <code>demo/evaluar.py</code>. El detector es YOLO11n entrenado con COCO, sin ajuste para fútbol.</p>

  <div class="dos" style="margin-top:56px">
    <div class="texto">
      <p class="eti">Modelo xG propio</p><h3 style="font-size:30px;margin-bottom:12px">Entrenado en una liga, probado en torneos</h3>
      <p style="color:var(--ink2)">Regresión logística entrenada con {miles(xg["datos"]["tiros_entrenamiento"])} tiros de la Premier League 2015/16 y evaluada
      en {miles(xg["datos"]["tiros_prueba"])} tiros sin penales del Mundial 2022, la Euro 2020 y 2024 y la Copa América 2024, que no vio al entrenar.
      Usa distancia, ángulo, cabeza, balón parado, presión, primer toque, defensores en el cono de tiro y la posición del arquero.</p>
      <div class="cifras" style="margin-top:22px">
        <div><b>{num(ctx["auc"], 3)}</b><span>AUC (StatsBomb: {num(ref["auc"], 3)}; solo geometría: {num(bas["auc"], 3)})</span></div>
        <div><b>{num(ctx["brier"], 3)}</b><span>Brier (StatsBomb: {num(ref["brier"], 3)})</span></div>
        <div><b>{num(ctx["xg_total"], 1)}</b><span>goles esperados frente a {ctx["goles"]} reales</span></div>
      </div>
    </div>
    <figure class="figura" style="margin:0"><img src="media/xg_calibracion.png" alt="Curva de calibración del modelo xG" width="900" height="740">
      <figcaption class="nota" style="margin-top:6px">Calibración en la muestra de prueba: probabilidad predicha frente a la proporción real de goles.</figcaption></figure>
  </div>
</div></section>

<section id="instalar"><div class="envoltura">
  <header><p class="eti">Empezar</p><h2>Instalar y usar</h2><p>Python 3.10 o más nuevo, en Windows, macOS o Linux.</p></header>
  <div class="dos codigo">
    <div><h3>1. Instalar</h3>
<div class="copiar"><pre>pip install "analista-futbol[todo] @ git+{REPO}"</pre></div>
      <h3 style="margin-top:18px">2. Abrir el panel</h3>
<div class="copiar"><pre>analista panel</pre></div>
      <p class="nota" style="margin-top:8px">Se abre en el navegador, en localhost:8765. La <a href="guia.html">guía</a> explica cómo conectar la capturadora HDMI.</p></div>
    <div><h3>O desde la línea de comandos y Python</h3>
<div class="copiar"><pre><span class="c"># reporte de la final del Mundial 2022</span>
analista datos statsbomb reporte 3869685

<span class="c"># analizar un video ya calibrado</span>
analista analizar partido.mp4 -c calibracion.json

<span class="c"># en Python</span>
from analista import Partido
p = Partido.skillcorner(1886347)
p.formacion("A")["formacion"]      <span class="c"># '4-2-3-1'</span></pre></div></div>
  </div>
</div></section>

<section id="limites"><div class="envoltura">
  <header><p class="eti">Para saber antes</p><h2>Límites</h2></header>
  <ul class="limites">
    <li><b>Protección de contenido</b>La app lee solo lo que tu equipo puede reproducir. Una señal con HDCP se ve negra en la capturadora y la app no la evita.</li>
    <li><b>Detector general</b>YOLO entrenado con COCO no distingue arquero, árbitro y jugador; el arquero y el árbitro salen como "otro" por el color.</li>
    <li><b>Balón en el aire y posesión</b>Con una sola cámara un balón alto se proyecta al suelo lejos de donde está, cerca de otros jugadores. En centros y remates eso desvía la posesión (en la prueba de TV, {pos_tv}).</li>
    <li><b>Deriva de la cámara de TV</b>Siguiendo el paneo, la calibración se desvía con el tiempo: en tomas de hasta 28 s, el error mediano fue de 1 m y el 90% quedó bajo 2 m. Calibrar cada toma lo reduce; un modelo de puntos de la cancha lo corregiría en cada cuadro.</li>
    <li><b>Repeticiones y primeros planos</b>Se detectan y se saltan. Tras un corte la app vuelve a ubicar la cámara comparando el césped y las líneas con el cuadro calibrado; lo más preciso sigue siendo calibrar el primer cuadro de cada toma.</li>
    <li><b>VAR con una cámara</b>La cámara de TV no ve a todos: si el receptor o el último defensor quedan fuera de imagen, la jugada sale dudosa o no revisable. La parte del cuerpo más adelantada se aproxima con la silueta, y el contacto de una falta no se ve en las posiciones: las faltas vienen de los eventos o se marcan a mano.</li>
    <li><b>Licencias</b>El código es MIT. YOLO (opcional) es AGPL-3.0. Los datos de StatsBomb son para uso educativo y de investigación.</li>
  </ul>
</div></section>
</main>

<footer><div class="envoltura">
  <span>Analista de fútbol · licencia MIT · <a href="{REPO}">GitHub</a> · <a href="guia.html">Guía de uso</a></span>
  <span>Datos: Hudl StatsBomb Open Data · SkillCorner Open Data (MIT) · SoccerTrack (Universidad de Tsukuba)</span>
</div></footer>
{pie}"""


def una_toma(u: dict) -> str:
    """El clip de TV calibrando solo la primera toma: la cámara se reubica sola después de la repetición."""
    if not u:
        return ""
    j, cam = u.get("jugadores") or {}, u.get("camara") or {}
    return (f"Con una sola calibración (sin calibrar la toma que sigue a la repetición), la cámara de TV queda en "
            f"{num(j.get('error_mediana_m'))} m de error de posición y {pct(j.get('recall'))} de jugadores detectados: tras el corte la app "
            f"reubica la cámara comparando el césped y las líneas con el cuadro calibrado ({num(cam.get('calibrado_pct'), 0)}% de los "
            f"cuadros calibrados, error de la cámara {num(cam.get('error_mediana_m'))} m). ")


def tabla_readme(evals: dict) -> str:
    """Tabla de validación en Markdown para el README (entre los marcadores VALIDACION)."""
    import re
    c, t = evals.get("completo", {}), evals.get("tv", {})
    filas = re.findall(r"<tr><td>(.*?)</td><td class='n'>(.*?)</td><td class='n'>(.*?)</td><td class='nota'>(.*?)</td></tr>",
                       filas_validacion(c, t))
    out = ["Medido contra la verdad de cada cuadro en dos transmisiones generadas con el tracking real de SkillCorner "
           "(ver [validación](https://jazaineam1.github.io/analista-futbol/#validacion)):", "",
           "| Medida | Plano completo (cámara fija) | Cámara de TV (paneo y zoom) | Qué significa |", "|---|---|---|---|"]
    out += [f"| {html.unescape(a)} | {html.unescape(b)} | {html.unescape(cc)} | {html.unescape(d)} |" for a, b, cc, d in filas]
    extra = una_toma(evals.get("tv_una", {}))
    if extra:
        out += ["", extra.strip()]
    return "\n".join(out)


def actualizar_readme(evals: dict):
    ruta = RAIZ / "README.md"
    txt = ruta.read_text(encoding="utf-8")
    ini, fin = "<!-- VALIDACION -->", "<!-- /VALIDACION -->"
    nuevo = f"{ini}\n{tabla_readme(evals)}\n{fin}"
    if fin in txt:
        a, b = txt.index(ini), txt.index(fin) + len(fin)
        txt = txt[:a] + nuevo + txt[b:]
    else:
        txt = txt.replace(ini, nuevo)
    ruta.write_text(txt, encoding="utf-8")


def capturas(pares: list[tuple[Path, Path]]):
    from playwright.sync_api import sync_playwright
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1200, "height": 750}, device_scale_factor=1)
        for html_, jpg in pares:
            if not html_.exists():
                continue
            pg.goto(html_.resolve().as_uri())
            pg.wait_for_timeout(500)
            jpg.parent.mkdir(parents=True, exist_ok=True)
            pg.screenshot(path=str(jpg), type="jpeg", quality=78)
        b.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--salidas", default=str(RAIZ / "demo" / "salidas"))
    ap.add_argument("--sin-capturas", action="store_true")
    ap.add_argument("--artefacto", action="store_true")
    a = ap.parse_args()
    salidas = Path(a.salidas)
    evals = {k: leer(salidas / k / "evaluacion.json") for k in ("completo", "tv", "real", "tv_una")}
    import re
    for k in ("completo", "tv", "real"):
        rep = salidas / k / "reporte.html"
        if rep.exists():
            (DOCS / "reportes").mkdir(parents=True, exist_ok=True)
            for viejo in (DOCS / "reportes").glob(f"video_{k}_var_*"):
                viejo.unlink()
            html_ = rep.read_text(encoding="utf-8")
            # las imágenes y los clips del VAR van al lado del reporte, con el nombre del reporte delante
            for nombre in sorted(set(re.findall(r'"(var_clip_\d+\.mp4|var_\d+\.jpg)"', html_))):
                if (salidas / k / nombre).exists():
                    shutil.copy(salidas / k / nombre, DOCS / "reportes" / f"video_{k}_{nombre}")
            html_ = re.sub(r'"(var_clip_\d+\.mp4|var_\d+\.jpg)"', lambda m_: f'"video_{k}_{m_.group(1)}"', html_)
            (DOCS / "reportes" / f"video_{k}.html").write_text(html_, encoding="utf-8")
    xg = leer(RAIZ / "analista" / "modelos" / "xg.json")
    if not a.sin_capturas:
        capturas([(DOCS / r, DOCS / img) for r, img, *_ in REPORTES])
    (DOCS / "index.html").write_text(pagina(evals, xg, False), encoding="utf-8")
    actualizar_readme(evals)
    (DOCS / ".nojekyll").write_text("", encoding="utf-8")
    print(f"Sitio: {DOCS / 'index.html'}")
    if a.artefacto:
        dest = RAIZ / "build" / "artefacto"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "index.html").write_text(pagina(evals, xg, True), encoding="utf-8")
        print(f"Artefacto: {dest / 'index.html'}")


if __name__ == "__main__":
    main()
