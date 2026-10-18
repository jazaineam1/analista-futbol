"""Panel web local: analizar un video o explorar datos abiertos de partidos.

    analista panel                     # abre http://localhost:8765
    analista panel --carpeta mi_proyecto --puerto 8765

Pestañas:
  Datos          explorar partidos de StatsBomb (eventos) y SkillCorner (tracking) y generar reportes
  Fuente         archivo de video, capturadora HDMI / cámara, pantalla o URL
  Calibrar       4 o más clics en la imagen y su punto en la cancha
  Analizar       procesar el video (o seguir en vivo) con vista del cuadro anotado
  Resultados     video anotado, métricas y reporte
  Cómo detecta   las 10 etapas de la detección sobre un momento elegido
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import webbrowser
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, abort, jsonify, request, send_from_directory

from ..cancha import Cancha, aplicar_h, h_desde_puntos
from ..util import json_limpio
from ..vision.calibracion import Calibracion
from ..vision.fuentes import FuenteArchivo, abrir_fuente, es_archivo

WEB = Path(__file__).resolve().parent / "web"


_json = json_limpio


class Estado:
    """Estado de la sesión: una persona usando el panel en su PC."""

    def __init__(self, carpeta: Path, modelo: str | None = None):
        self.carpeta = Path(carpeta)
        self.carpeta.mkdir(parents=True, exist_ok=True)
        self.modelo = modelo
        self.fuente_spec: str | None = None
        self.info_fuente: dict = {}
        self.cap_vivo = None
        self.lock = threading.Lock()
        self.trabajo: dict = {"estado": "inactivo"}
        self.jpeg: bytes | None = None
        self.detener = False
        self.explicacion: dict = {"estado": "inactivo"}
        self.datos: dict = {"estado": "inactivo"}

    @property
    def ruta_calibracion(self) -> Path:
        return self.carpeta / "calibracion.json"

    @property
    def resultados(self) -> Path:
        return self.carpeta / "resultados"

    @property
    def reportes(self) -> Path:
        return self.carpeta / "reportes"

    def cerrar_vista_previa(self):
        with self.lock:
            if self.cap_vivo is not None:
                try:
                    self.cap_vivo.close()
                except Exception:  # noqa: BLE001
                    pass
            self.cap_vivo = None


def crear_app(estado: Estado, ancho_proceso: int = 1280) -> Flask:
    app = Flask(__name__, static_folder=None)

    # ------------------------------------------------------------ utilidades
    def leer_cuadro(t: float = 0.0, ancho: int | None = ancho_proceso) -> np.ndarray | None:
        spec = estado.fuente_spec
        if spec is None:
            return None
        if es_archivo(spec):
            cap = cv2.VideoCapture(spec)
            fps = cap.get(cv2.CAP_PROP_FPS) or 25
            if t > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(t * fps)))
            ok, f = cap.read()
            cap.release()
            frame = f if ok else None
        else:
            with estado.lock:
                if estado.cap_vivo is None:
                    estado.cap_vivo = iter(abrir_fuente(spec, fps=5))
                try:
                    _, _, frame = next(estado.cap_vivo)
                except StopIteration:
                    estado.cap_vivo = None
                    frame = None
        if frame is None:
            return None
        h, w = frame.shape[:2]
        if ancho and w != ancho:
            frame = cv2.resize(frame, (ancho, int(round(h * ancho / w))), interpolation=cv2.INTER_AREA)
        return frame

    def jpeg(img, calidad=85) -> Response:
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, calidad])
        return Response(buf.tobytes(), mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    # ------------------------------------------------------------ páginas y archivos
    @app.get("/")
    def index():
        return send_from_directory(WEB, "index.html")

    @app.get("/web/<path:nombre>")
    def web(nombre):
        return send_from_directory(WEB, nombre)

    @app.get("/resultados/<path:nombre>")
    def resultados(nombre):
        return send_from_directory(estado.resultados, nombre)

    @app.get("/explicacion/<path:nombre>")
    def explicacion(nombre):
        return send_from_directory(estado.carpeta / "explicacion", nombre)

    @app.get("/reportes/<path:nombre>")
    def reportes(nombre):
        return send_from_directory(estado.reportes, nombre)

    @app.get("/api/estado")
    def api_estado():
        cal = estado.ruta_calibracion.exists()
        res = (estado.resultados / "reporte.html").exists()
        return jsonify(carpeta=str(estado.carpeta.resolve()), fuente=estado.info_fuente or None,
                       calibracion=cal, resultados=res, trabajo=estado.trabajo.get("estado"))

    # ------------------------------------------------------------ fuente
    @app.post("/api/fuente")
    def api_fuente():
        spec = (request.json or {}).get("spec", "").strip().strip('"')
        if not spec:
            return jsonify(error="Indica una fuente"), 400
        estado.cerrar_vista_previa()
        try:
            if es_archivo(spec):
                if not Path(spec).exists():
                    return jsonify(error=f"No existe el archivo: {spec}"), 400
                f = FuenteArchivo(spec)
                info = {"tipo": "archivo", "fps": f.fps, "duracion": (f.total or 0) / (f.fps or 25),
                        "ancho_nativo": f.ancho}
                f.cerrar()
            else:
                info = {"tipo": "vivo", "fps": None, "duracion": None}
            estado.fuente_spec = spec
            nativo = leer_cuadro(0, ancho=None)
            if nativo is None:
                return jsonify(error="No se pudo leer un cuadro de la fuente"), 400
            frame = leer_cuadro(0)
            info.update({"spec": spec, "ancho": frame.shape[1], "alto": frame.shape[0],
                         "ancho_nativo": info.get("ancho_nativo") or nativo.shape[1],
                         "alto_nativo": nativo.shape[0], "negro": float(nativo.mean()) < 3})
            estado.info_fuente = info
            return jsonify(info)
        except Exception as e:  # noqa: BLE001
            return jsonify(error=str(e)), 400

    @app.get("/api/cuadro.jpg")
    def api_cuadro():
        frame = leer_cuadro(float(request.args.get("t", 0)))
        if frame is None:
            abort(404)
        return jpeg(frame)

    # ------------------------------------------------------------ calibración
    @app.get("/api/cancha")
    def api_cancha():
        largo = float(request.args.get("largo", 105))
        ancho = float(request.args.get("ancho", 68))
        c = Cancha(largo=largo, ancho=ancho)
        return jsonify({"largo": largo, "ancho": ancho, "lineas": [np.round(pl, 2).tolist() for pl in c.lineas(paso=1.0)],
                        "puntos": c.puntos_referencia()})

    @app.get("/api/calibracion")
    def api_calibracion_get():
        if not estado.ruta_calibracion.exists():
            return jsonify({})
        return jsonify(json.loads(estado.ruta_calibracion.read_text(encoding="utf-8")))

    def _proyeccion(pares, largo, ancho):
        c = Cancha(largo=largo, ancho=ancho)
        H = h_desde_puntos([p["imagen"] for p in pares], [p["cancha"] for p in pares])
        Hinv = np.linalg.inv(H)
        # Para saber qué lado del plano está frente a la cámara se usa un punto del suelo visible
        w, h = estado.info_fuente.get("ancho", 1280), estado.info_fuente.get("alto", 720)
        ref = Hinv @ np.append(aplicar_h(H, [[w / 2, 0.9 * h]])[0], 1.0)
        signo = np.sign(ref[2]) or 1.0
        lineas = []
        for pl in c.lineas(paso=0.5):
            hom = np.hstack([pl, np.ones((len(pl), 1))]) @ Hinv.T
            ok = hom[:, 2] * signo > 1e-9
            pts = hom[:, :2] / np.where(np.abs(hom[:, 2:3]) < 1e-12, 1e-12, hom[:, 2:3])
            ok &= (np.abs(pts) < 2e4).all(1)
            tramo = []
            for k in range(len(pts)):
                if ok[k]:
                    tramo.append(pts[k])
                else:
                    if len(tramo) > 1:
                        lineas.append(np.round(tramo, 1).tolist())
                    tramo = []
            if len(tramo) > 1:
                lineas.append(np.round(tramo, 1).tolist())
        err = np.linalg.norm(aplicar_h(H, [p["imagen"] for p in pares]) - np.array([p["cancha"] for p in pares]), axis=1)
        return H, lineas, err

    def _avisos(pares) -> list[str]:
        av = []
        P = np.array([p["cancha"] for p in pares], float)
        if len(P) >= 3:
            # ¿todos los puntos de la cancha sobre una misma recta?
            c = P - P.mean(0)
            s = np.linalg.svd(c, compute_uv=False)
            if s[-1] < 1.0:
                av.append("Los puntos de la cancha están casi en línea recta: elige al menos uno fuera de esa línea.")
        return av

    @app.post("/api/calibracion/previsualizar")
    def api_cal_prev():
        d = request.json or {}
        pares = d.get("pares", [])
        if len(pares) < 4:
            return jsonify(lineas=[], errores=[], avisos=[])
        avisos = _avisos(pares)
        try:
            _, lineas, err = _proyeccion(pares, d.get("largo", 105), d.get("ancho", 68))
            if len(err) > 4 and err.max() > 2.0:
                k = int(np.argmax(err))
                avisos.append(f"El punto {k + 1} no calza con los demás ({err[k]:.1f} m): revisa dónde lo marcaste.")
            return jsonify(lineas=lineas, errores=np.round(err, 2).tolist(), avisos=avisos)
        except Exception as e:  # noqa: BLE001
            return jsonify(error=str(e), avisos=avisos), 400

    @app.post("/api/calibracion")
    def api_cal_guardar():
        d = request.json or {}
        pares = d.get("pares", [])
        if len(pares) < 4:
            return jsonify(error="Hacen falta al menos 4 pares de puntos"), 400
        largo, ancho = float(d.get("largo", 105)), float(d.get("ancho", 68))
        H, _, err = _proyeccion(pares, largo, ancho)
        fps = estado.info_fuente.get("fps") or 25
        cuadro = int(round(float(d.get("t", 0)) * fps)) if estado.info_fuente.get("tipo") == "archivo" else 0
        clave = {"cuadro": cuadro, "puntos": pares, "H_img2cancha": H.tolist()}
        claves = [clave]
        if d.get("agregar") and estado.ruta_calibracion.exists():
            # Otra toma (después de un corte o una repetición): se suma a las claves existentes
            previa = Calibracion.cargar(estado.ruta_calibracion)
            claves = sorted([k for k in previa.claves if int(k.get("cuadro") or 0) != cuadro] + [clave],
                            key=lambda k: int(k.get("cuadro") or 0))
        cal = Calibracion(Cancha(largo=largo, ancho=ancho),
                          (estado.info_fuente.get("ancho"), estado.info_fuente.get("alto")),
                          bool(d.get("camara_fija", False)), None, claves)
        cal.guardar(estado.ruta_calibracion)
        return jsonify(ok=True, error_medio_m=round(float(err.mean()), 2), ruta=str(estado.ruta_calibracion),
                       tomas=len(claves))

    # ------------------------------------------------------------ análisis de video
    def _config(opciones: dict, cal: Calibracion):
        from ..vision.pipeline import ConfigPipeline
        equipos = {e.get("nombre") or f"Equipo {k}": e["color"]
                   for k, e in zip("AB", opciones.get("equipos", [])) if e.get("color")}
        nativo = int(estado.info_fuente.get("ancho_nativo") or ancho_proceso)
        imgsz = int(opciones.get("imgsz", 1280))
        ancho = min(nativo, max(int(opciones.get("ancho_proceso", ancho_proceso)), imgsz))
        modelo = opciones.get("modelo") or estado.modelo or "yolo11n.pt"
        return ConfigPipeline(detector=opciones.get("detector", "yolo"), modelo_yolo=modelo, imgsz=imgsz,
                              ancho_proceso=ancho, paso=int(opciones.get("paso", 1)), equipos=equipos,
                              camara_fija=cal.camara_fija), equipos

    def _correr(opciones: dict):
        from ..postproceso import postprocesar
        from ..vision.pipeline import Pipeline
        estado.detener = False
        salida = estado.resultados
        salida.mkdir(parents=True, exist_ok=True)
        (salida / "video_anotado.webm").unlink(missing_ok=True)      # no mostrar un video de otro análisis
        try:
            cal = Calibracion.cargar(estado.ruta_calibracion)
            cfg, equipos = _config(opciones, cal)
            spec = estado.fuente_spec
            vivo = not es_archivo(spec)
            fuente = abrir_fuente(spec, fps=float(opciones.get("fps_vivo", 10)),
                                  inicio_s=float(opciones.get("inicio") or 0), fin_s=opciones.get("fin") or None)
            pipe = Pipeline(cal, cfg)
            cont = {"A": 0, "B": 0}
            t0 = time.time()

            def al_cuadro(res, anotado, n):
                ok, buf = cv2.imencode(".jpg", anotado, [cv2.IMWRITE_JPEG_QUALITY, 80])
                estado.jpeg = buf.tobytes()
                if res.balon is not None and res.jugadores:
                    d = [np.hypot(j["x"] - res.balon["x"], j["y"] - res.balon["y"]) for j in res.jugadores]
                    k = int(np.argmin(d))
                    if d[k] <= 2.0 and res.jugadores[k].get("equipo") in cont:
                        cont[res.jugadores[k]["equipo"]] += 1
                tot = sum(cont.values())
                visibles = {e: sum(1 for j in res.jugadores if j.get("equipo") == e) for e in ("A", "B")}
                estado.trabajo.update({
                    "n": n, "t_video": round(res.t, 1), "cuadros_por_s": round(n / max(1e-6, time.time() - t0), 2),
                    "calibrado": res.camara.calibrado, "visibles": visibles,
                    "posesion": {e: round(100 * v / tot) for e, v in cont.items()} if tot else None,
                    "total": (fuente.total // cfg.paso) if getattr(fuente, "total", None) else None})

            estado.trabajo = {"estado": "corriendo", "vivo": vivo, "n": 0, "inicio": time.time(),
                              "ancho_proceso": cfg.ancho_proceso}
            resumen = pipe.correr(fuente, salida, video_anotado=bool(opciones.get("grabar", True)) or not vivo,
                                  progreso=False, al_cuadro=al_cuadro, detener=lambda: estado.detener)
            fuente.cerrar()
            estado.trabajo["estado"] = "calculando métricas"
            kits = {k: v for k, v in zip("AB", equipos.values())} if equipos else None
            nota = Path(spec).name if not vivo else f"en vivo ({spec})"
            r = postprocesar(salida, cal.cancha, resumen, opciones.get("ataca_derecha") or None, kits,
                             opciones.get("titulo") or "Análisis del partido", nota)
            archivos = sorted(p.name for p in salida.iterdir() if p.is_file())
            estado.trabajo.update({"estado": "listo", "resumen": r["resumen"], "archivos": archivos,
                                   "segundos": round(time.time() - estado.trabajo["inicio"], 1)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            estado.trabajo.update({"estado": "error", "error": str(e)})

    @app.post("/api/analizar")
    def api_analizar():
        if estado.trabajo.get("estado") in ("corriendo", "calculando métricas"):
            return jsonify(error="Ya hay un análisis en curso"), 409
        if estado.fuente_spec is None:
            return jsonify(error="Primero elige una fuente"), 400
        if not estado.ruta_calibracion.exists():
            return jsonify(error="Primero guarda una calibración"), 400
        estado.cerrar_vista_previa()
        estado.jpeg = None
        # El estado pasa a "corriendo" antes de lanzar el hilo: si no, la primera consulta del
        # navegador puede ver "inactivo" y dejar de seguir el análisis
        estado.trabajo = {"estado": "corriendo", "n": 0, "inicio": time.time()}
        threading.Thread(target=_correr, args=(request.json or {},), daemon=True).start()
        return jsonify(ok=True)

    @app.get("/api/resultados")
    def api_resultados():
        """Resumen y archivos del último análisis guardado (también tras reiniciar el panel)."""
        carpeta = estado.resultados
        if not (carpeta / "metricas.json").exists():
            return jsonify(resumen=None, archivos=[])
        resumen = json.loads((carpeta / "metricas.json").read_text(encoding="utf-8"))
        proceso = {}
        if (carpeta / "resumen_proceso.json").exists():
            proceso = json.loads((carpeta / "resumen_proceso.json").read_text(encoding="utf-8"))
        archivos = sorted(f.name for f in carpeta.iterdir() if f.is_file() and not f.name.endswith(".webm"))
        return jsonify(_json({"resumen": resumen, "archivos": archivos, "proceso": proceso}))

    @app.post("/api/detener")
    def api_detener():
        estado.detener = True
        return jsonify(ok=True)

    @app.get("/api/trabajo")
    def api_trabajo():
        return jsonify(_json(estado.trabajo))

    @app.get("/api/trabajo/cuadro.jpg")
    def api_trabajo_cuadro():
        if estado.jpeg is None:
            abort(404)
        return Response(estado.jpeg, mimetype="image/jpeg", headers={"Cache-Control": "no-store"})

    @app.get("/api/trabajo/stream")
    def api_stream():
        def gen():
            ultimo = None
            while True:
                j = estado.jpeg
                if j is not None and j is not ultimo:
                    ultimo = j
                    yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + j + b"\r\n"
                if estado.trabajo.get("estado") not in ("corriendo", "inactivo") and ultimo is not None:
                    break
                time.sleep(0.05)
        return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")

    # ------------------------------------------------------------ cómo detecta
    def _explicar(t: float, opciones: dict):
        from ..vision.diagnostico import explicar_video
        try:
            cal = Calibracion.cargar(estado.ruta_calibracion)
            cfg, equipos = _config(opciones, cal)
            nombres = dict(zip("AB", equipos.keys())) if equipos else {"A": "Equipo A", "B": "Equipo B"}
            carpeta = estado.carpeta / "explicacion"
            prefijo = f"m{int(time.time())}"

            def progreso(n, t_act, objetivo):
                estado.explicacion.update({"n": n, "t_video": round(t_act, 1), "objetivo": objetivo})
            momentos = explicar_video(estado.fuente_spec, cal, cfg, [t], carpeta, prefijo, nombres, progreso)
            if not momentos:
                raise RuntimeError("No se llegó a ese instante del video")
            m = momentos[0]
            for e in m["etapas"]:
                if e["img"]:
                    e["img"] = "/explicacion/" + e["img"].split("/", 1)[1]
            estado.explicacion = {"estado": "listo", "t": m["t"], "etapas": m["etapas"], "nombres": nombres}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            estado.explicacion = {"estado": "error", "error": str(e)}

    @app.post("/api/explicar")
    def api_explicar():
        if estado.explicacion.get("estado") == "corriendo":
            return jsonify(error="Ya se está preparando una explicación"), 409
        if estado.fuente_spec is None or not es_archivo(estado.fuente_spec):
            return jsonify(error="Elige un archivo de video (la explicación necesita poder volver a un instante)"), 400
        if not estado.ruta_calibracion.exists():
            return jsonify(error="Primero guarda una calibración"), 400
        d = request.json or {}
        estado.explicacion = {"estado": "corriendo", "n": 0}
        threading.Thread(target=_explicar, args=(float(d.get("t", 0)), d.get("opciones", {})), daemon=True).start()
        return jsonify(ok=True)

    @app.get("/api/explicacion")
    def api_explicacion():
        return jsonify(_json(estado.explicacion))

    # ------------------------------------------------------------ datos abiertos
    @app.get("/api/datos/competiciones")
    def api_competiciones():
        from ..datos import statsbomb as sb
        try:
            c = sb.competiciones()
            c = c.sort_values(["competicion", "temporada"], ascending=[True, False])
            return jsonify(statsbomb=c.to_dict("records"),
                           skillcorner=[{"competicion": "A-League (Australia)", "temporada": "2024/2025",
                                         "fuente": "skillcorner"}])
        except Exception as e:  # noqa: BLE001
            return jsonify(error=f"No se pudo leer el catálogo de StatsBomb: {e}"), 502

    @app.get("/api/datos/partidos")
    def api_partidos():
        fuente = request.args.get("fuente", "statsbomb")
        try:
            if fuente == "skillcorner":
                from ..datos import skillcorner as sc
                p = sc.partidos()
            else:
                from ..datos import statsbomb as sb
                p = sb.partidos(int(request.args["competicion"]), int(request.args["temporada"]))
            hechos = {f.stem for f in estado.reportes.glob("*.html")}
            p["reporte"] = [f"{fuente}_{i}" in hechos for i in p["partido_id"]]
            return jsonify(partidos=_json(p.where(p.notna(), None).to_dict("records")))
        except Exception as e:  # noqa: BLE001
            return jsonify(error=str(e)), 502

    def _reporte_datos(fuente: str, partido_id: int, comp: int | None, temp: int | None):
        from ..partido import Partido
        try:
            estado.datos.update({"paso": "descargando datos" if fuente == "skillcorner" else "leyendo eventos"})
            p = Partido.skillcorner(partido_id) if fuente == "skillcorner" else Partido.statsbomb(partido_id, comp, temp)
            estado.datos.update({"paso": "calculando métricas y gráficos"})
            nombre = f"{fuente}_{partido_id}.html"
            p.reporte(estado.reportes / nombre)
            r = p.resumen()
            estado.datos = {"estado": "listo", "fuente": fuente, "partido_id": partido_id,
                            "url": f"/reportes/{nombre}", "equipos": p.nombres, "marcador": p.info.get("marcador"),
                            "resumen": _json(r)}
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            estado.datos = {"estado": "error", "error": str(e), "fuente": fuente, "partido_id": partido_id}

    @app.post("/api/datos/reporte")
    def api_datos_reporte():
        if estado.datos.get("estado") == "corriendo":
            return jsonify(error="Ya se está generando un reporte"), 409
        d = request.json or {}
        fuente = d.get("fuente", "statsbomb")
        pid = int(d["partido_id"])
        estado.datos = {"estado": "corriendo", "fuente": fuente, "partido_id": pid, "paso": "empezando",
                        "inicio": time.time()}
        threading.Thread(target=_reporte_datos, args=(fuente, pid, d.get("competicion"), d.get("temporada")),
                         daemon=True).start()
        return jsonify(ok=True)

    @app.get("/api/datos/trabajo")
    def api_datos_trabajo():
        return jsonify(_json(estado.datos))

    @app.get("/api/datos/reportes")
    def api_datos_reportes():
        out = []
        for f in sorted(estado.reportes.glob("*.html"), key=lambda p: p.stat().st_mtime, reverse=True):
            titulo = f.stem
            try:
                txt = f.read_text(encoding="utf-8", errors="ignore")[:4000]
                i, j = txt.find("<title>"), txt.find("</title>")
                if i >= 0 and j > i:
                    titulo = txt[i + 7:j]
            except OSError:
                pass
            out.append({"archivo": f.name, "url": f"/reportes/{f.name}", "titulo": titulo,
                        "fuente": f.stem.split("_")[0]})
        return jsonify(reportes=out)

    return app


def main(carpeta: str = "proyecto", puerto: int = 8765, ancho_proceso: int = 1280, navegador: bool = True,
         modelo: str | None = None, host: str = "127.0.0.1"):
    estado = Estado(Path(carpeta), modelo)
    app = crear_app(estado, ancho_proceso)
    url = f"http://localhost:{puerto}"
    print(f"Panel en {url}  (Ctrl+C para salir) · proyecto: {estado.carpeta.resolve()}")
    if navegador:
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()
    app.run(host=host, port=puerto, threaded=True, debug=False)
