"""Prueba de punta a punta del panel en un navegador real (Playwright + Chromium).

Recorre todas las pestañas como lo haría una persona: datos abiertos (StatsBomb y SkillCorner),
fuente, calibración con clics, análisis de un clip corto, resultados y "Cómo detecta". Falla si
alguna acción no produce lo esperado o si la página registra errores de JavaScript.

    analista panel --carpeta /tmp/prueba --puerto 8899 --sin-navegador &
    python demo/probar_panel.py --url http://localhost:8899 --video demo/clips/tv/video.mp4 \\
        --calibracion demo/clips/tv/calibracion.json
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import expect, sync_playwright


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


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8899")
    ap.add_argument("--video", required=True, help="clip corto para analizar (ruta absoluta o relativa)")
    ap.add_argument("--calibracion", required=True, help="calibracion.json del clip (se usan sus puntos para hacer clic)")
    ap.add_argument("--capturas", default=None, help="carpeta para guardar capturas de cada pestaña")
    a = ap.parse_args()
    video = str(Path(a.video).resolve())
    cal = json.loads(Path(a.calibracion).read_text(encoding="utf-8"))
    pares = cal["claves"][0]["puntos"]
    res_cal = cal["resolucion"]
    cap = Path(a.capturas) if a.capturas else None
    if cap:
        cap.mkdir(parents=True, exist_ok=True)
    P = Prueba()
    errores_js: list[str] = []

    with sync_playwright() as pw:
        nav = pw.chromium.launch()
        pg = nav.new_page(viewport={"width": 1366, "height": 900})
        pg.on("pageerror", lambda e: errores_js.append(str(e)))
        pg.on("console", lambda m: errores_js.append("consola: " + m.text) if m.type == "error" and "404" not in m.text and "status of 400" not in m.text else None)

        def foto(nombre):
            if cap:
                pg.screenshot(path=str(cap / f"{nombre}.png"))

        # ---------------------------------------------------------------- datos
        def statsbomb():
            pg.goto(a.url + "/#datos")
            pg.wait_for_selector("#partidos [data-pid]", timeout=90000)
            opciones = pg.locator("#comp option").count()
            assert opciones > 50, f"solo {opciones} competiciones"
            pg.fill("#buscar", "Argentina")
            filas = pg.locator("#partidos tr").count()
            assert 1 <= filas <= 8, f"el filtro dejó {filas} filas"
            pg.click("#partidos tr:has-text('France') [data-pid]")
            pg.wait_for_selector("#dt-iframe:not([hidden])", timeout=120000)
            expect(pg.locator("#dt-kpis .kpi")).to_have_count(6)
            texto = pg.locator("#dt-kpis").inner_text()
            assert "3" in texto and "2,76" in texto, f"KPIs inesperados: {texto!r}"
            fr = pg.frame_locator("#dt-iframe")
            expect(fr.locator("h1")).to_contain_text("Argentina")
            assert fr.locator("svg").count() >= 5, "el reporte tiene pocos gráficos"
            expect(pg.locator("#dt-abrir")).to_be_visible()
            pg.wait_for_selector("#reportes-lista a", timeout=10000)
            foto("1_datos_statsbomb")
        P.paso("Datos · StatsBomb: lista, búsqueda y reporte de la final 2022", statsbomb)

        def skillcorner():
            pg.fill("#buscar", "")
            pg.click("[data-fuente=skillcorner]")
            pg.wait_for_selector("#partidos tr:has-text('Newcastle') [data-pid]", timeout=90000)
            n = pg.locator("#partidos [data-pid]").count()
            assert n >= 10, f"{n} partidos de SkillCorner"
            pg.click("#partidos tr:has-text('Newcastle') [data-pid]")
            pg.wait_for_selector("#dt-iframe:not([hidden])", timeout=300000)
            texto = pg.locator("#dt-kpis").inner_text()
            assert "Posesión" in texto, f"KPIs de tracking: {texto!r}"
            fr = pg.frame_locator("#dt-iframe")
            expect(fr.locator("h1")).to_contain_text("Auckland")
            assert fr.locator("svg").count() >= 6
            foto("2_datos_skillcorner")
        P.paso("Datos · SkillCorner: reporte de tracking", skillcorner)

        # ---------------------------------------------------------------- fuente
        def fuente():
            pg.click("nav.tabs button[data-tab=fuente]")
            pg.fill("#spec", "/no/existe.mp4")
            pg.click("#abrir")
            expect(pg.locator("#fuente-msg")).to_have_class("msg err", timeout=10000)
            pg.fill("#spec", video)
            pg.click("#abrir")
            pg.wait_for_selector("#fuente-vista img", timeout=30000)
            expect(pg.locator("#estado-pill")).to_have_text("archivo listo")
            assert "1280" in pg.locator("#fuente-info").inner_text()
            foto("3_fuente")
        P.paso("Fuente: error con ruta inválida y apertura del video", fuente)

        # ---------------------------------------------------------------- calibrar
        def calibrar():
            pg.click("#ir-calibrar")
            pg.wait_for_function("(() => { const i = document.querySelector('#cal-img'), c = document.querySelector('#cal-canvas');"
                                 " return i.complete && i.naturalWidth > 0 && c.width === i.naturalWidth; })()", timeout=30000)
            pg.click("#cal-borrar")
            canvas = pg.locator("#cal-canvas").bounding_box()
            esc = 1280 / res_cal[0]
            for k, par in enumerate(pares[:5]):
                u, v = par["imagen"][0] * esc, par["imagen"][1] * esc
                pg.mouse.click(canvas["x"] + u * canvas["width"] / 1280, canvas["y"] + v * canvas["height"] / (res_cal[1] * esc))
                x, y = par["cancha"]
                pt = pg.evaluate("""([x, y]) => { const s = document.querySelector('#cal-cancha'); const p = s.createSVGPoint();
                    p.x = x; p.y = -y; const q = p.matrixTransform(s.getScreenCTM()); return [q.x, q.y]; }""", [x, y])
                pg.mouse.click(pt[0], pt[1])
                expect(pg.locator("#cal-lista li .n")).to_have_count(k + 1)
            pg.wait_for_timeout(800)
            errores = [float(t.replace(",", ".").replace(" m", "")) for t in pg.locator("#cal-lista .err").all_inner_texts() if t]
            assert errores and max(errores) < 0.6, f"errores de calibración {errores}"
            pg.click("#cal-deshacer")
            expect(pg.locator("#cal-lista li .n")).to_have_count(4)
            pg.wait_for_timeout(500)
            pg.click("#cal-guardar")
            expect(pg.locator("#cal-msg")).to_contain_text("Calibración guardada", timeout=10000)
            foto("4_calibrar")
        P.paso("Calibrar: 5 clics en imagen y cancha, deshacer y guardar", calibrar)

        # ---------------------------------------------------------------- analizar
        def analizar():
            pg.click("nav.tabs button[data-tab=analizar]")
            pg.fill("#eq-a", "Auckland FC")
            pg.fill("#eq-b", "Newcastle Jets")
            pg.evaluate("document.querySelector('#col-b').value='#f2f2f2'")
            pg.click("#an-iniciar")
            pg.wait_for_selector("#an-img:not([hidden])", timeout=180000)
            foto("5_analizar")
            expect(pg.locator("#an-estado")).to_contain_text("Listo en", timeout=900000)
            expect(pg.locator("#an-msg")).to_contain_text("Análisis terminado")
        P.paso("Analizar: video en vivo y fin del análisis", analizar)

        def resultados():
            pg.click("nav.tabs button[data-tab=resultados]")
            pg.wait_for_selector("#res-cont:not([hidden])", timeout=20000)
            expect(pg.locator("#res-kpis .kpi").first).to_be_visible()
            assert pg.locator("#res-archivos a").count() >= 8, "faltan archivos de resultados"
            fr = pg.frame_locator("#res-iframe")
            expect(fr.locator("h1")).to_be_visible(timeout=20000)
            for nombre in ("tracking.csv", "reporte.html", "video_anotado.mp4"):
                r = pg.request.get(f"{a.url}/resultados/{nombre}")
                assert r.ok and len(r.body()) > 1000, f"{nombre} no se sirve"
            foto("6_resultados")
        P.paso("Resultados: KPIs, archivos y reporte", resultados)

        # ---------------------------------------------------------------- cómo detecta
        def detecta():
            pg.click("nav.tabs button[data-tab=detecta]")
            pg.evaluate("document.querySelector('#ex-t').value = 3.0")
            pg.click("#ex-iniciar")
            pg.wait_for_selector("#ex-cont:not([hidden])", timeout=600000)
            expect(pg.locator("#ex-riel button")).to_have_count(10)
            for k in range(9):
                pg.keyboard.press("ArrowRight")
            expect(pg.locator("#ex-contador")).to_contain_text("Etapa 10 de 10")
            expect(pg.locator("#ex-tit")).to_contain_text("táctica")
            pg.click("#ex-riel button:nth-child(7)")
            expect(pg.locator("#ex-tit")).to_have_text("Balón")
            img = pg.locator("#ex-visor img").first
            assert pg.evaluate("i => i.complete && i.naturalWidth > 100", img.element_handle())
            foto("7_detecta")
        P.paso("Cómo detecta: 10 etapas, teclado y lupa", detecta)

        def recarga():
            pg.reload()
            pg.click("nav.tabs button[data-tab=detecta]")
            pg.wait_for_selector("#ex-cont:not([hidden])", timeout=10000)
            pg.click("nav.tabs button[data-tab=calibrar]")
            pg.wait_for_timeout(1500)
            assert pg.locator("#cal-lista li .n").count() == 4, "no recuperó la calibración guardada"
        P.paso("Al recargar se recuperan la explicación y la calibración", recarga)

        def tema_oscuro_y_movil():
            ctx = nav.new_context(viewport={"width": 390, "height": 844}, color_scheme="dark")
            m = ctx.new_page()
            errs = []
            m.on("pageerror", lambda e: errs.append(str(e)))
            m.goto(a.url + "/#datos")
            m.wait_for_selector("#partidos [data-pid]", timeout=60000)
            ancho = m.evaluate("document.documentElement.scrollWidth")
            assert ancho <= 392, f"la página se desborda en el teléfono ({ancho} px)"
            fondo = m.evaluate("getComputedStyle(document.body).backgroundColor")
            assert fondo != "rgb(244, 245, 242)", "el tema oscuro no se aplica"
            if cap:
                m.screenshot(path=str(cap / "8_movil_oscuro.png"))
            ctx.close()
            assert not errs, errs
        P.paso("Teléfono y tema oscuro", tema_oscuro_y_movil)
        nav.close()

    if errores_js:
        P.fallas.append(("Errores de JavaScript", "; ".join(errores_js[:5])))
    print(f"\n{len(P.ok)} pasos bien, {len(P.fallas)} fallas")
    for n, e in P.fallas:
        print(f"  - {n}: {e}")
    raise SystemExit(1 if P.fallas else 0)


if __name__ == "__main__":
    main()
