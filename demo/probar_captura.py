"""Prueba real de la captura de pantalla: una página web reproduce el partido y el panel lo analiza en vivo.

Simula a una persona que ve el partido en una página (como un servicio de streaming) y usa el panel
para capturar esa zona de la pantalla: elige el monitor, detecta el reproductor, calibra sobre el
cuadro congelado, analiza en vivo grabando la señal, y al final se comparan las posiciones con la
verdad del clip (alineando el tiempo del análisis con el tiempo del video de la página).

Necesita una pantalla (en Linux sin escritorio: Xvfb) y Playwright:

    Xvfb :99 -screen 0 1920x1080x24 &
    DISPLAY=:99 analista panel --carpeta /tmp/cap --puerto 8791 --sin-navegador &
    DISPLAY=:99 python demo/probar_captura.py --url http://localhost:8791 --clip demo/clips/tv --segundos 24
"""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
from playwright.sync_api import expect, sync_playwright

PAGINA = """<!doctype html><html lang="es"><head><meta charset="utf-8"><title>Canal Demo · En vivo</title>
<style>body{{margin:0;background:#0d0f14;color:#e9ecf2;font:15px system-ui,sans-serif;overflow:hidden}}
header{{height:64px;display:flex;align-items:center;gap:14px;padding:0 28px;background:#151923;border-bottom:1px solid #232a38}}
header b{{font-size:20px}}.vivo{{background:#e5372b;color:#fff;font-weight:700;font-size:12px;padding:3px 8px;border-radius:4px}}
main{{position:absolute;left:{x}px;top:{y}px}}video{{width:{w}px;height:{h}px;display:block;background:#000}}
aside{{position:absolute;left:{ax}px;top:{y}px;width:300px;height:{h}px;background:#151923;border-radius:8px;padding:14px;box-sizing:border-box}}
aside p{{margin:0 0 10px;color:#9aa3b5}}</style></head><body>
<header><b>Canal Demo</b><span class="vivo">EN VIVO</span><span>Auckland FC – Newcastle Jets · A-League (transmisión simulada)</span></header>
<main><video id="v" src="{src}" muted playsinline preload="auto"></video><p style="margin:12px 0 0;color:#9aa3b5">Página de prueba: un reproductor de video con el partido.</p></main>
<aside><p>Chat del partido</p><p>· vamos Auckland</p><p>· qué golazo</p></aside>
</body></html>"""


class Prueba:
    def __init__(self):
        self.ok, self.fallas = [], []

    def paso(self, nombre, fn):
        t0 = time.time()
        try:
            fn()
            self.ok.append(nombre)
            print(f"  ok   {nombre} ({time.time() - t0:.1f} s)", flush=True)
        except Exception as e:  # noqa: BLE001
            txt = (str(e).strip().splitlines() or [repr(e)])[0][:300]
            self.fallas.append((nombre, txt))
            print(f"  FALLA {nombre}: {txt}", flush=True)


def a_webm(mp4: Path, destino: Path) -> Path:
    """Chromium de Playwright no trae H.264: la página usa el clip en WebM (VP9)."""
    if destino.exists():
        return destino
    from analista.vision.pipeline import ruta_ffmpeg
    subprocess.run([ruta_ffmpeg(), "-y", "-loglevel", "error", "-i", str(mp4), "-c:v", "libvpx-vp9", "-b:v", "0",
                    "-crf", "30", "-deadline", "realtime", "-cpu-used", "8", "-row-mt", "1", "-an", str(destino)], check=True)
    return destino


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8791")
    ap.add_argument("--clip", default="demo/clips/tv", help="carpeta del clip sintético (video.mp4, calibracion.json, gt_*.csv)")
    ap.add_argument("--segundos", type=float, default=24.0, help="cuánto dura el análisis en vivo")
    ap.add_argument("--fps", type=int, default=5, help="cuadros por segundo del análisis en vivo")
    ap.add_argument("--capturas", default=None)
    ap.add_argument("--salida", default=None, help="json con los resultados de la prueba")
    a = ap.parse_args()
    clip = Path(a.clip).resolve()
    cal = json.loads((clip / "calibracion.json").read_text(encoding="utf-8"))
    pares = cal["claves"][0]["puntos"]
    ancho_v, alto_v = cal["resolucion"]
    caps = Path(a.capturas) if a.capturas else None
    if caps:
        caps.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix="pagina_partido_"))
    webm = a_webm(clip / "video.mp4", clip / "video.webm")
    X, Y = 160, 120                                          # dónde está el reproductor en la página
    (tmp / "partido.html").write_text(PAGINA.format(x=X, y=Y, w=ancho_v, h=alto_v, ax=X + ancho_v + 24, src=webm.as_uri()),
                                      encoding="utf-8")
    P = Prueba()
    errores_js: list[str] = []
    datos: dict = {}

    with sync_playwright() as pw:
        # La "TV": un navegador visible, a pantalla completa, con el partido
        tv = pw.chromium.launch(headless=False, args=["--kiosk", "--window-position=0,0", "--window-size=1920,1080",
                                                      "--autoplay-policy=no-user-gesture-required", "--disable-infobars"])
        tvp = tv.new_page(no_viewport=True)
        tvp.goto((tmp / "partido.html").as_uri())
        tvp.wait_for_function("document.getElementById('v').readyState >= 3", timeout=30000)
        tvp.evaluate("document.getElementById('v').play()")
        # Dónde quedó el reproductor en la pantalla (el navegador puede tener una barra arriba)
        sx, sy = tvp.evaluate("""(() => { const r = document.getElementById('v').getBoundingClientRect();
            return [Math.round(window.screenX + r.left), Math.round(window.screenY + (window.outerHeight - window.innerHeight) + r.top)]; })()""")
        datos["reproductor_en_pantalla"] = [sx, sy, ancho_v, alto_v]
        # El panel: otro navegador (sin ventana), como si estuviera en otro monitor o detrás
        nav = pw.chromium.launch()
        pg = nav.new_page(viewport={"width": 1366, "height": 900})
        pg.on("pageerror", lambda e: errores_js.append(str(e)))
        pg.on("console", lambda m: errores_js.append("consola: " + m.text) if m.type == "error" and "40" not in m.text else None)

        def foto(nombre):
            if caps:
                pg.screenshot(path=str(caps / f"{nombre}.png"), full_page=True)

        def detectar():
            pg.goto(a.url + "/#fuente")
            pg.click(".tipo[data-tipo=pantalla]")
            pg.wait_for_selector("#pan-marco:not([hidden])", timeout=20000)
            assert pg.locator("#pan-monitor option").count() >= 1
            t_a = tvp.evaluate("document.getElementById('v').currentTime")
            tvp.wait_for_timeout(600)
            assert tvp.evaluate("document.getElementById('v').currentTime") > t_a, "el video de la página no avanza"
            pg.click("#pan-detectar")
            expect(pg.locator("#pan-msg")).to_contain_text("Reproductor encontrado", timeout=20000)
            txt = pg.locator("#pan-zona-txt").inner_text()
            nums = [int(v) for v in __import__("re").findall(r"\d+", txt)]
            w, h, x, y = nums[:4]
            datos["zona_detectada"] = [x, y, w, h]
            assert abs(x - sx) <= 24 and abs(y - sy) <= 24 and abs(w - ancho_v) <= 48 and abs(h - alto_v) <= 48, \
                f"zona {x},{y},{w},{h} lejos de {sx},{sy},{ancho_v},{alto_v}"
            foto("1_pantalla")
            pg.click("#pan-usar")
            expect(pg.locator("#estado-pill")).to_have_text("pantalla en vivo", timeout=20000)
            pg.wait_for_selector("#vista-img", timeout=10000)
            info = pg.locator("#fuente-info").inner_text()
            assert "pantalla del PC" in info, info
            # la persona ajusta el rectángulo al borde exacto del reproductor (aquí, escribiendo la zona)
            pg.click(".tipo[data-tipo=ruta]")
            pg.fill("#spec", f"pantalla:1:{sx},{sy},{ancho_v},{alto_v}")
            pg.click("#abrir")
            expect(pg.locator("#fuente-msg")).to_have_text("Fuente lista.", timeout=20000)
        P.paso("Pantalla del PC: monitor, detectar el reproductor y usar la zona", detectar)

        def calibrar():
            # La persona pausa el video en una toma amplia y calibra sobre ese cuadro congelado
            tvp.evaluate("(() => { const v = document.getElementById('v'); v.pause(); v.currentTime = 0; })()")
            tvp.wait_for_timeout(800)
            pg.click("nav.tabs button[data-tab=calibrar]")
            expect(pg.locator("#cal-vivo-fila")).to_be_visible()
            pg.click("#cal-congelar")
            pg.wait_for_function("(() => { const i = document.querySelector('#cal-img'), c = document.querySelector('#cal-canvas');"
                                 " return i.complete && i.naturalWidth > 0 && c.width === i.naturalWidth; })()", timeout=30000)
            pg.fill("#largo", str(cal["cancha"]["largo"]))       # primero el tamaño de la cancha
            pg.keyboard.press("Tab")
            pg.wait_for_timeout(800)
            pg.click("#cal-borrar")
            zona_w = datos["zona_detectada"][2] if datos.get("zona_detectada") else ancho_v
            canvas = pg.locator("#cal-canvas").bounding_box()
            nat = pg.evaluate("[document.querySelector('#cal-img').naturalWidth, document.querySelector('#cal-img').naturalHeight]")
            esc = nat[0] / ancho_v
            for k, par in enumerate(pares[:5]):
                u, v = par["imagen"][0] * esc, par["imagen"][1] * esc
                pg.mouse.click(canvas["x"] + u * canvas["width"] / nat[0], canvas["y"] + v * canvas["height"] / nat[1])
                x, y = par["cancha"]
                pt = pg.evaluate("""([x, y]) => { const s = document.querySelector('#cal-cancha'); const p = s.createSVGPoint();
                    p.x = x; p.y = -y; const q = p.matrixTransform(s.getScreenCTM()); return [q.x, q.y]; }""", [x, y])
                pg.mouse.click(pt[0], pt[1])
                expect(pg.locator("#cal-lista li .n")).to_have_count(k + 1)
            pg.wait_for_timeout(800)
            errores = [float(t.replace(",", ".").replace(" m", "")) for t in pg.locator("#cal-lista .err").all_inner_texts() if t]
            datos["error_calibracion_m"] = max(errores) if errores else None
            assert errores and max(errores) < 1.0, f"errores de calibración {errores} (zona {zona_w} px)"
            pg.click("#cal-guardar")
            expect(pg.locator("#cal-msg")).to_contain_text("Calibración guardada", timeout=10000)
            foto("2_calibrar")
        P.paso("Calibrar sobre el cuadro congelado de la página", calibrar)

        registro: list[tuple[float, float]] = []

        def analizar():
            pg.click("nav.tabs button[data-tab=analizar]")
            expect(pg.locator("#an-vivo")).to_be_visible()
            pg.fill("#eq-a", "Auckland FC")
            pg.fill("#eq-b", "Newcastle Jets")
            pg.evaluate("document.querySelector('#col-b').value='#f2f2f2'")
            pg.select_option("#fps-vivo", str(a.fps))
            assert pg.locator("#grabar-senal").is_checked()
            tvp.evaluate("document.getElementById('v').play()")
            pg.click("#an-iniciar")
            t_fin = time.time() + a.segundos
            while time.time() < t_fin:
                ct = tvp.evaluate("document.getElementById('v').currentTime")
                registro.append((time.time(), float(ct)))
                time.sleep(0.4)
            expect(pg.locator("#an-estado")).to_contain_text("En vivo", timeout=5000)
            foto("3_en_vivo")
            pg.click("#an-detener")
            expect(pg.locator("#an-estado")).to_contain_text("Listo en", timeout=600000)
            expect(pg.locator("#an-msg")).to_contain_text("grabada", timeout=10000)
            foto("4_listo")
        P.paso(f"Analizar en vivo {a.segundos:.0f} s grabando la señal", analizar)

        def grabacion():
            pg.click("nav.tabs button[data-tab=fuente]")
            pg.click(".tipo[data-tipo=subir]")
            pg.wait_for_selector("#videos li .nom", timeout=10000)
            assert "grabación en vivo" in pg.locator("#videos").inner_text()
        P.paso("La grabación aparece en Videos del proyecto", grabacion)

        def var():
            pg.click("nav.tabs button[data-tab=var]")
            pg.wait_for_selector("#var-cont:not([hidden]), #var-vacio:not([hidden])", timeout=15000)
            foto("5_var")
        P.paso("Pestaña VAR del análisis en vivo", var)

        def subir():
            pg.click("nav.tabs button[data-tab=fuente]")
            pg.click(".tipo[data-tipo=subir]")
            pg.set_input_files("#archivo", str(clip / "video.mp4"))
            expect(pg.locator("#subir-msg")).to_contain_text("Subido", timeout=60000)
            expect(pg.locator("#estado-pill")).to_have_text("archivo listo", timeout=20000)
            assert "video" in pg.locator("#videos").inner_text()
            foto("6_subido")
        P.paso("Subir un video desde el navegador", subir)
        tv.close()
        nav.close()

    # ------------------------------------------------------------ comparación con la verdad
    def verdad():
        carpeta = Path(json.loads(__import__("urllib.request").request.urlopen(a.url + "/api/estado").read())["carpeta"])
        res = carpeta / "resultados"
        proc = json.loads((res / "resumen_proceso.json").read_text(encoding="utf-8"))
        assert proc.get("en_vivo") and proc.get("inicio_epoch"), "el resumen no marca el análisis en vivo"
        tr = pd.read_csv(res / "tracking.csv")
        reg = np.array(registro)
        jugando = np.r_[True, np.diff(reg[:, 1]) > 0]
        reg = reg[jugando]
        # tiempo del análisis -> hora real -> segundo del video de la página
        t_video = np.interp(tr["t"].to_numpy() + proc["inicio_epoch"], reg[:, 0], reg[:, 1], left=np.nan, right=np.nan)
        tr["cuadro"] = np.round(t_video * 25)
        tr = tr.dropna(subset=["cuadro"])
        import importlib.util
        spec = importlib.util.spec_from_file_location("evaluar", Path(__file__).with_name("evaluar.py"))
        ev = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ev)
        gt_todo = pd.read_csv(clip / "gt_jugadores.csv")
        gt = gt_todo[gt_todo["cuadro"].isin(set(tr["cuadro"].astype(int)))]
        m = ev.evaluar_jugadores(tr.assign(cuadro=tr["cuadro"].astype(int)), gt, cal["cancha"]["largo"], cal["cancha"]["ancho"], 2.0)
        # diagnóstico: ¿cuánto mejora si el tiempo se corre? (latencia de la captura)
        desfases = {}
        for d in np.arange(-1.0, 1.01, 0.2):
            c2 = (tr["cuadro"] + round(d * 25)).astype(int)
            g2 = gt_todo[gt_todo["cuadro"].isin(set(c2))]
            desfases[round(float(d), 1)] = ev.evaluar_jugadores(tr.assign(cuadro=c2), g2, cal["cancha"]["largo"], cal["cancha"]["ancho"], 2.0)["recall"]
        datos["recall_por_desfase_s"] = desfases
        datos["registro"] = [[round(a_, 3), round(b_, 3)] for a_, b_ in registro]
        cam = pd.read_csv(res / "camara.csv")
        grab = [v for v in (carpeta / "videos").glob("en_vivo_*.mp4")]
        import cv2
        dur_grab = None
        if grab:
            c = cv2.VideoCapture(str(grab[0]))
            dur_grab = c.get(cv2.CAP_PROP_FRAME_COUNT) / (c.get(cv2.CAP_PROP_FPS) or 25)
            c.release()
        c = cv2.VideoCapture(str(res / "video_anotado.mp4"))
        dur_anot = c.get(cv2.CAP_PROP_FRAME_COUNT) / (c.get(cv2.CAP_PROP_FPS) or 10)
        c.release()
        datos.update({"cuadros_analizados": int(proc["cuadros_procesados"]), "cuadros_por_s": proc["cuadros_por_segundo"],
                      "camara_calibrada_pct": round(100 * float(cam["calibrado"].mean()), 1),
                      "jugadores": {k: m.get(k) for k in ("recall", "precision", "error_mediana_m", "error_p90_m", "equipo_correcto")},
                      "duracion_analisis_s": round(float(tr["t"].max() - tr["t"].min()), 1),
                      "duracion_grabacion_s": round(dur_grab, 1) if dur_grab else None,
                      "duracion_video_anotado_s": round(dur_anot, 1)})
        print(json.dumps(datos, indent=2, ensure_ascii=False))
        # Con 2 núcleos, el navegador reproduciendo el video y la grabación en el mismo equipo, el
        # análisis va a ~3 cuadros/s: se exige que funcione, no la precisión del análisis de un archivo
        assert m["recall"] > 0.6 and m["error_mediana_m"] < 1.5, f"tracking en vivo pobre: {m}"
        assert dur_grab and abs(dur_grab - a.segundos) < 6, f"la grabación dura {dur_grab} s"
    if not P.fallas or len(P.ok) >= 3:
        P.paso("Posiciones del análisis en vivo contra la verdad del clip", verdad)

    if errores_js:
        P.fallas.append(("Errores de JavaScript", "; ".join(errores_js[:5])))
    print(f"\n{len(P.ok)} pasos bien, {len(P.fallas)} fallas")
    for n, t in P.fallas:
        print(f"  - {n}: {t}")
    if a.salida:
        Path(a.salida).write_text(json.dumps({"ok": P.ok, "fallas": P.fallas, "datos": datos}, indent=2, ensure_ascii=False),
                                  encoding="utf-8")
    raise SystemExit(1 if P.fallas else 0)


if __name__ == "__main__":
    main()
