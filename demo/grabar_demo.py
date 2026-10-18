"""Graba el video de demostración del panel (docs/media/demo.mp4, .webm y .gif).

Necesita dos paneles corriendo:
  A  proyecto vacío (con la calibración del clip de TV):   analista panel --carpeta /tmp/demoA --puerto 8801 --sin-navegador
  B  proyecto con resultados y una explicación ya hechos:   analista panel --carpeta /tmp/demoB --puerto 8802 --sin-navegador

    python demo/grabar_demo.py --video-tv demo/clips/tv/video.mp4 -o docs/media
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

from analista.vision.pipeline import ruta_ffmpeg

LEYENDA_JS = """
(texto) => {
  let d = document.getElementById('leyenda-demo');
  if (!d) {
    d = document.createElement('div'); d.id = 'leyenda-demo';
    d.style.cssText = 'position:fixed;left:50%;bottom:22px;transform:translateX(-50%);z-index:99;max-width:86%;' +
      'background:rgba(10,18,14,.92);color:#fff;font:600 21px/1.3 system-ui,sans-serif;padding:12px 20px;border-radius:10px;' +
      'box-shadow:0 8px 30px rgba(0,0,0,.35);text-align:center;transition:opacity .25s';
    document.body.appendChild(d);
  }
  d.hidden = !texto; d.textContent = texto || '';
}
"""


class Grabacion:
    def __init__(self, pagina, t0):
        self.pg, self.t0 = pagina, t0
        self.rapidos: list[tuple[float, float]] = []    # tramos de espera que se aceleran al editar

    def ahora(self):
        return time.time() - self.t0

    def leyenda(self, texto: str | None):
        self.pg.evaluate(LEYENDA_JS, texto)

    def esperar(self, ms):
        self.pg.wait_for_timeout(ms)

    def espera_rapida(self, condicion, timeout=180000):
        a = self.ahora()
        condicion(timeout)
        self.rapidos.append((a + 0.6, self.ahora() - 0.2))

    def scroll_iframe(self, selector, hasta_px, duracion_ms, pasos=40):
        fr = self.pg.frame_locator(selector)
        for k in range(1, pasos + 1):
            fr.locator("body").evaluate(f"() => window.scrollTo(0, {hasta_px * k / pasos})")
            self.pg.wait_for_timeout(duracion_ms // pasos)


def escena_a(pw, url, video_tv, carpeta_video, cancha_txt):
    ctx = pw.chromium.launch().new_context(viewport={"width": 1280, "height": 720}, record_video_dir=str(carpeta_video),
                                           record_video_size={"width": 1280, "height": 720}, color_scheme="light")
    pg = ctx.new_page()
    g = Grabacion(pg, time.time())
    pg.goto(url + "/#datos")
    pg.wait_for_selector("#partidos [data-pid]", timeout=60000)
    g.leyenda("Analista de fútbol: del partido a los datos")
    g.esperar(2200)
    g.leyenda("1 · Casi 4.000 partidos con eventos de StatsBomb: cada pase, tiro y presión")
    g.esperar(1500)
    pg.click("#buscar")
    pg.keyboard.type("Argentina", delay=90)
    g.esperar(900)
    pg.click("#partidos tr:has-text('France') [data-pid]")
    g.espera_rapida(lambda to: pg.wait_for_selector("#dt-iframe:not([hidden])", timeout=to))
    g.esperar(700)
    pg.evaluate("document.querySelector('#datos-trabajo').scrollIntoView({behavior:'smooth'})")
    g.leyenda("La final del Mundial 2022: xG, tiros, redes de pases y presión")
    g.esperar(1800)
    g.scroll_iframe("#dt-iframe", 2100, 7000)
    g.esperar(600)
    # Tracking
    pg.evaluate("window.scrollTo({top:0,behavior:'smooth'})")
    g.leyenda("2 · Tracking de SkillCorner: los 22 jugadores, 10 veces por segundo")
    g.esperar(900)
    pg.fill("#buscar", "")
    pg.click("[data-fuente=skillcorner]")
    pg.wait_for_selector("#partidos tr:has-text('Auckland') [data-pid]", timeout=60000)
    g.esperar(700)
    pg.click("#buscar")
    pg.keyboard.type("Auckland", delay=80)
    g.esperar(700)
    pg.click("#partidos tr:has-text('Newcastle') [data-pid]")
    g.leyenda("Posesión, bloque, formación, control de cancha y físico de 90 minutos")
    g.espera_rapida(lambda to: pg.wait_for_selector("#dt-iframe:not([hidden])", timeout=to))
    g.esperar(700)
    pg.evaluate("document.querySelector('#datos-trabajo').scrollIntoView({behavior:'smooth'})")
    g.esperar(1200)
    g.scroll_iframe("#dt-iframe", 2600, 7500)
    g.esperar(500)
    # Video
    g.leyenda("3 · O analiza tu propio video: archivo, capturadora HDMI o pantalla")
    pg.click("nav.tabs button[data-tab=fuente]")
    g.esperar(700)
    pg.click("#spec")
    pg.keyboard.type(video_tv, delay=12)
    pg.click("#abrir")
    pg.wait_for_selector("#fuente-vista img", timeout=30000)
    g.esperar(2200)
    pg.click("#ir-calibrar")
    g.leyenda("Calibra con 4 o más clics: las líneas rojas muestran que la cancha calza")
    g.esperar(4200)
    pg.click("nav.tabs button[data-tab=analizar]")
    g.leyenda("Nombres y colores de los equipos, y a analizar")
    pg.click("#eq-a")
    pg.keyboard.type("Auckland FC", delay=40)
    pg.click("#eq-b")
    pg.keyboard.type("Newcastle Jets", delay=40)
    pg.evaluate("document.querySelector('#col-b').value='#f2f2f2'")
    g.esperar(500)
    pg.click("#an-iniciar")
    pg.wait_for_selector("#an-img:not([hidden])", timeout=120000)
    g.leyenda("Detecta, separa los equipos por color y ubica a todos en metros, en vivo")
    g.esperar(11000)
    pg.click("#an-detener")
    g.esperar(300)
    marcas = g.rapidos
    ruta = pg.video.path()
    ctx.close()
    return Path(ruta), marcas


def escena_b(pw, url, carpeta_video):
    ctx = pw.chromium.launch().new_context(viewport={"width": 1280, "height": 720}, record_video_dir=str(carpeta_video),
                                           record_video_size={"width": 1280, "height": 720}, color_scheme="light")
    pg = ctx.new_page()
    g = Grabacion(pg, time.time())
    pg.goto(url + "/#resultados")
    pg.wait_for_selector("#res-cont:not([hidden])", timeout=60000)
    pg.evaluate("document.querySelector('#res-video').play()")
    g.leyenda("4 · Resultados: video anotado con minimapa, métricas y reporte")
    g.esperar(6500)
    pg.evaluate("document.querySelector('#res-iframe').scrollIntoView({behavior:'smooth'})")
    g.esperar(1200)
    g.leyenda("El reporte: posesión, pases, presión y los goles, con la validación contra la verdad")
    g.scroll_iframe("#res-iframe", 1800, 6000)
    pg.click("nav.tabs button[data-tab=detecta]")
    pg.wait_for_selector("#ex-cont:not([hidden])", timeout=30000)
    g.leyenda("Y te muestra cómo detecta, etapa por etapa, sobre el cuadro que elijas")
    g.esperar(2200)
    for _ in range(9):
        pg.click("#ex-sig")
        g.esperar(1500)
    g.leyenda("Analista de fútbol · código abierto · github.com/jazaineam1/analista-futbol")
    g.esperar(3500)
    ruta = pg.video.path()
    ctx.close()
    return Path(ruta), g.rapidos


def editar(clips: list[tuple[Path, list]], salida: Path, factor=5.0):
    """Une las escenas acelerando las esperas y exporta MP4 (H.264), WebM (VP9) y un GIF corto."""
    ff = ruta_ffmpeg()
    tmp = salida / "_partes"
    tmp.mkdir(parents=True, exist_ok=True)
    partes = []
    for n, (ruta, rapidos) in enumerate(clips):
        cortes, t = [], 0.0
        for a, b in rapidos:
            if b - a < 0.5:
                continue
            cortes.append((t, a, 1.0))
            cortes.append((a, b, factor))
            t = b
        cortes.append((t, None, 1.0))          # hasta el final (el WebM de Playwright no trae duración)
        for k, (a, b, f) in enumerate(cortes):
            if b is not None and b - a < 0.05:
                continue
            p = tmp / f"{n}_{k}.mp4"
            hasta = ["-to", f"{b:.3f}"] if b is not None else []
            subprocess.run([ff, "-y", "-loglevel", "error", "-ss", f"{a:.3f}", *hasta, "-i", str(ruta),
                            "-vf", f"setpts=PTS/{f},fps=30,format=yuv420p", "-an", "-c:v", "libx264", "-crf", "18", str(p)], check=True)
            partes.append(p)
    lista = tmp / "lista.txt"
    lista.write_text("".join(f"file '{p.resolve()}'\n" for p in partes))
    mp4 = salida / "demo.mp4"
    subprocess.run([ff, "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lista), "-c:v", "libx264",
                    "-crf", "24", "-preset", "slow", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(mp4)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libvpx-vp9", "-b:v", "0", "-crf", "36",
                    "-deadline", "good", "-cpu-used", "4", "-an", str(salida / "demo.webm")], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-ss", "3", "-i", str(mp4), "-frames:v", "1", "-q:v", "3",
                    str(salida / "portada.jpg")], check=True)
    # GIF para el README: 960 px, 10 cuadros/s, paleta propia
    pal = tmp / "pal.png"
    filtro = "setpts=0.5*PTS,fps=6,scale=640:-1:flags=lanczos"   # el doble de rápido, liviano para el README
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", f"{filtro},palettegen=max_colors=96:stats_mode=diff",
                    str(pal)], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-i", str(pal), "-lavfi",
                    f"{filtro}[x];[x][1:v]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle", str(salida / "demo.gif")],
                   check=True)
    for p in tmp.iterdir():
        p.unlink()
    tmp.rmdir()
    return mp4


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="http://localhost:8801")
    ap.add_argument("--b", default="http://localhost:8802")
    ap.add_argument("--video-tv", required=True, help="ruta absoluta del clip de TV (se escribe en el panel)")
    ap.add_argument("-o", "--salida", default="docs/media")
    a = ap.parse_args()
    salida = Path(a.salida)
    crudo = salida / "_crudo"
    with sync_playwright() as pw:
        va, ma = escena_a(pw, a.a, a.video_tv, crudo, "")
        vb, mb = escena_b(pw, a.b, crudo)
    (crudo / "marcas.json").write_text(json.dumps({"a": ma, "b": mb}))
    mp4 = editar([(va, ma), (vb, mb)], salida)
    print(f"Demo: {mp4}")


if __name__ == "__main__":
    main()
