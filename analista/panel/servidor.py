"""Panel web local: analizar un video o explorar datos abiertos de partidos.

    analista panel                     # abre http://localhost:8765
    analista panel --carpeta mi_proyecto --puerto 8765

Pestañas:
  Datos          explorar partidos de StatsBomb (eventos) y SkillCorner (tracking) y generar reportes
  Fuente         subir un video, la pantalla del PC (la página con el partido en vivo), la
                 capturadora HDMI conectada al televisor / decodificador, o una URL
  Calibrar       4 o más clics en la imagen y su punto en la cancha
  Analizar       procesar el video (o seguir en vivo) con vista del cuadro anotado
  Resultados     video anotado, métricas y reporte
  Cómo detecta   las 10 etapas de la detección sobre un momento elegido
"""
from __future__ import annotations

import json
import re
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
from ..vision.fuentes import (
    FuenteArchivo,
    abrir_fuente,
    captura_pantalla,
    es_archivo,
    interpretar_pantalla,
    listar_dispositivos,
    listar_pantallas,
    sugerir_zona,
)

EXT_VIDEO = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts", ".wmv", ".flv", ".mpg", ".mpeg"}

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
        self.datos_hechos: dict[str, dict] = {}         # reportes de datos abiertos ya generados en esta sesión
        self.ultimo_vivo: np.ndarray | None = None      # último cuadro leído de una fuente en vivo
        self.cuadro_cal: np.ndarray | None = None       # cuadro en vivo congelado para calibrar
        self.fuente_activa = None                       # la fuente que está usando el análisis
        self.miniaturas: dict[int, np.ndarray] = {}      # cuadro de prueba de cada capturadora

    @property
    def ruta_calibracion(self) -> Path:
        return self.carpeta / "calibracion.json"

    @property
    def resultados(self) -> Path:
        return self.carpeta / "resultados"

    @property
    def reportes(self) -> Path:
        return self.carpeta / "reportes"

    @property
    def videos(self) -> Path:
        return self.carpeta / "videos"

    def cerrar_vista_previa(self):
        with self.lock:
            if self.cap_vivo is not None:
                try:
                    self.cap_vivo.cerrar()
                except Exception:  # noqa: BLE001
                    pass
            self.cap_vivo = None


def crear_app(estado: Estado, ancho_proceso: int = 1280) -> Flask:
    app = Flask(__name__, static_folder=None)
    app.config["MAX_CONTENT_LENGTH"] = None          # videos de partidos completos (varios GB)

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
        elif estado.fuente_activa is not None and getattr(estado.fuente_activa, "en_vivo", False):
            # Mientras se analiza en vivo, la vista previa usa la misma captura (una capturadora
            # no se puede abrir dos veces)
            frame = estado.fuente_activa.ultimo_cuadro()
        elif spec.startswith("pantalla"):
            monitor, region = interpretar_pantalla(spec)
            frame = captura_pantalla(monitor, region)
            estado.ultimo_vivo = frame
        else:
            with estado.lock:
                if estado.cap_vivo is None:
                    estado.cap_vivo = abrir_fuente(spec, fps=10)
                frame = estado.cap_vivo.ultimo_cuadro()
            if frame is not None:
                estado.ultimo_vivo = frame
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

    @app.get("/activos/<path:nombre>")
    def activos(nombre):
        from ..reportes.var import ACTIVOS
        return send_from_directory(ACTIVOS, nombre)

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
        estado.cuadro_cal = None
        try:
            if es_archivo(spec):
                if not Path(spec).exists():
                    return jsonify(error=f"No existe el archivo: {spec}"), 400
                f = FuenteArchivo(spec)
                info = {"tipo": "archivo", "fps": f.fps, "duracion": (f.total or 0) / (f.fps or 25),
                        "ancho_nativo": f.ancho}
                f.cerrar()
            else:
                tipo = "pantalla" if spec.startswith("pantalla") else ("capturadora" if spec.startswith("disp:") else "red")
                info = {"tipo": "vivo", "subtipo": tipo, "fps": None, "duracion": None}
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
        try:
            frame = leer_cuadro(float(request.args.get("t", 0)))
        except Exception as e:  # noqa: BLE001   (pantalla o capturadora que dejó de responder)
            return jsonify(error=str(e)), 503
        if frame is None:
            abort(404)
        if request.args.get("cal") and estado.fuente_spec and not es_archivo(estado.fuente_spec):
            # En vivo la imagen cambia: el cuadro que se muestra para calibrar queda congelado
            estado.cuadro_cal = estado.ultimo_vivo
        return jpeg(frame)

    # ------------------------------------------------------------ subir videos
    @app.post("/api/subir")
    def api_subir():
        """Sube un video desde el navegador a la carpeta del proyecto (videos/)."""
        f = request.files.get("video")
        if f is None or not f.filename:
            return jsonify(error="No llegó ningún archivo"), 400
        nombre = Path(f.filename).name
        ext = Path(nombre).suffix.lower()
        if ext not in EXT_VIDEO:
            return jsonify(error=f"Formato no reconocido ({ext or 'sin extensión'}). Usa mp4, mkv, mov, avi o webm."), 400
        base = re.sub(r"[^\w.\- ]+", "_", Path(nombre).stem).strip() or "video"
        estado.videos.mkdir(parents=True, exist_ok=True)
        destino = estado.videos / f"{base}{ext}"
        k = 2
        while destino.exists():
            destino = estado.videos / f"{base} ({k}){ext}"
            k += 1
        f.save(destino)
        cap = cv2.VideoCapture(str(destino))
        ok = cap.isOpened() and cap.read()[0]
        cap.release()
        if not ok:
            destino.unlink(missing_ok=True)
            return jsonify(error="El archivo se subió pero no se puede leer como video"), 400
        return jsonify(ok=True, ruta=str(destino.resolve()), nombre=destino.name)

    @app.get("/api/videos")
    def api_videos():
        out = []
        if estado.videos.exists():
            for f in sorted(estado.videos.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
                if f.suffix.lower() not in EXT_VIDEO:
                    continue
                cap = cv2.VideoCapture(str(f))
                fps = cap.get(cv2.CAP_PROP_FPS) or 25
                n = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                cap.release()
                out.append({"nombre": f.name, "ruta": str(f.resolve()), "mb": round(f.stat().st_size / 1e6, 1),
                            "duracion": round(n / fps, 1) if n else None})
        return jsonify(videos=out)

    # ------------------------------------------------------------ pantalla y capturadoras
    @app.get("/api/pantallas")
    def api_pantallas():
        try:
            return jsonify(pantallas=listar_pantallas())
        except Exception as e:  # noqa: BLE001
            return jsonify(pantallas=[], error=f"No se puede capturar la pantalla en este equipo: {e}")

    @app.get("/api/pantalla.jpg")
    def api_pantalla_jpg():
        """Foto del monitor completo, reducida, para elegir la zona del reproductor."""
        # Con un solo monitor: unos segundos para pasar a la ventana del partido antes de la foto
        time.sleep(min(10.0, max(0.0, float(request.args.get("espera", 0)))))
        try:
            img = captura_pantalla(int(request.args.get("monitor", 1)))
        except Exception as e:  # noqa: BLE001
            return jsonify(error=str(e)), 400
        ancho = int(request.args.get("ancho", 1280))
        if img.shape[1] > ancho:
            img = cv2.resize(img, (ancho, int(round(img.shape[0] * ancho / img.shape[1]))), interpolation=cv2.INTER_AREA)
        return jpeg(img, 80)

    @app.get("/api/pantalla/sugerir")
    def api_pantalla_sugerir():
        """Busca sola la zona del reproductor (lo que se mueve en la pantalla)."""
        time.sleep(min(10.0, max(0.0, float(request.args.get("espera", 0)))))
        try:
            z = sugerir_zona(int(request.args.get("monitor", 1)))
        except Exception as e:  # noqa: BLE001
            return jsonify(error=str(e)), 400
        if z is None:
            return jsonify(error="No encontré el reproductor: ¿el video está en pausa o tapado? Marca la zona a mano."), 404
        return jsonify(z)

    @app.get("/api/dispositivos")
    def api_dispositivos():
        """Capturadoras y cámaras conectadas (se prueban los índices 0-5)."""
        estado.cerrar_vista_previa()            # libera la capturadora si la estaba usando la vista previa
        try:
            lista = listar_dispositivos()
        except Exception as e:  # noqa: BLE001
            return jsonify(dispositivos=[], error=str(e))
        estado.miniaturas = {d["indice"]: d.pop("cuadro") for d in lista}
        return jsonify(dispositivos=lista)

    @app.get("/api/dispositivo.jpg")
    def api_dispositivo_jpg():
        img = estado.miniaturas.get(int(request.args.get("i", 0)))
        if img is None:
            abort(404)
        img = cv2.resize(img, (480, int(round(img.shape[0] * 480 / img.shape[1]))), interpolation=cv2.INTER_AREA)
        return jpeg(img, 80)

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
        vivo = estado.info_fuente.get("tipo") == "vivo"
        # En vivo no hay "cuadro N": la calibración se hizo sobre el cuadro congelado que se ve en
        # pantalla, que se guarda como referencia para ubicar la cámara cuando empiece el análisis
        # (cada toma en vivo tiene su número negativo: -1, -2, … no coinciden con ningún cuadro real)
        previa = Calibracion.cargar(estado.ruta_calibracion) if d.get("agregar") and estado.ruta_calibracion.exists() else None
        if vivo:
            negativos = [int(k.get("cuadro") or 0) for k in (previa.claves if previa else []) if int(k.get("cuadro") or 0) < 0]
            cuadro = min(negativos, default=0) - 1
        else:
            cuadro = int(round(float(d.get("t", 0)) * fps))
        clave = {"cuadro": cuadro, "puntos": pares, "H_img2cancha": H.tolist()}
        if vivo:
            ref = estado.cuadro_cal if estado.cuadro_cal is not None else estado.ultimo_vivo
        else:
            ref = leer_cuadro(float(d.get("t", 0)), ancho=None)
        if ref is not None:
            w_cal = int(estado.info_fuente.get("ancho") or ref.shape[1])
            ref = cv2.resize(ref, (w_cal, int(round(ref.shape[0] * w_cal / ref.shape[1]))), interpolation=cv2.INTER_AREA)
            nombre_ref = f"calibracion_ref_{'vivo' + str(-cuadro) if vivo else cuadro}.jpg"
            cv2.imwrite(str(estado.carpeta / nombre_ref), ref, [cv2.IMWRITE_JPEG_QUALITY, 92])
            clave["imagen_ref"] = nombre_ref
        claves = [clave]
        if previa is not None:
            # Otra toma (después de un corte o una repetición): se suma a las claves existentes
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
            estado.fuente_activa = fuente
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
            grabacion = None
            if vivo and opciones.get("grabar_senal", True):
                grabacion = estado.videos / f"en_vivo_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
                estado.trabajo["grabacion"] = grabacion.name
            resumen = pipe.correr(fuente, salida, video_anotado=bool(opciones.get("grabar", True)) or not vivo,
                                  progreso=False, al_cuadro=al_cuadro, detener=lambda: estado.detener,
                                  grabar_entrada=grabacion)
            fuente.cerrar()
            if grabacion is not None and grabacion.exists():
                resumen["grabacion"] = grabacion.name
            estado.trabajo["estado"] = "calculando métricas"
            kits = {k: v for k, v in zip("AB", equipos.values())} if equipos else None
            nota = Path(spec).name if not vivo else f"en vivo ({spec})"
            video = spec if not vivo else (grabacion if grabacion is not None and grabacion.exists() else None)
            r = postprocesar(salida, cal.cancha, resumen, opciones.get("ataca_derecha") or None, kits,
                             opciones.get("titulo") or "Análisis del partido", nota, video)
            archivos = sorted(p.name for p in salida.iterdir() if p.is_file())
            estado.trabajo.update({"estado": "listo", "resumen": r["resumen"], "archivos": archivos,
                                   "segundos": round(time.time() - estado.trabajo["inicio"], 1)})
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            estado.trabajo.update({"estado": "error", "error": str(e)})
        finally:
            f, estado.fuente_activa = estado.fuente_activa, None
            if f is not None:
                try:
                    f.cerrar()
                except Exception:  # noqa: BLE001
                    pass

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

    # ------------------------------------------------------------ VAR
    @app.get("/api/var")
    def api_var():
        ruta = estado.resultados / "var_items.json"
        if not ruta.exists():
            return jsonify(items=[], resumen=None)
        d = json.loads(ruta.read_text(encoding="utf-8"))
        for it in d.get("items", []):
            if it.get("foto"):
                it["foto"] = "/resultados/" + it["foto"]
        proc = {}
        if (estado.resultados / "resumen_proceso.json").exists():
            proc = json.loads((estado.resultados / "resumen_proceso.json").read_text(encoding="utf-8"))
        d["equipos"] = proc.get("nombres_equipos", {"A": "Equipo A", "B": "Equipo B"})
        d["fps"] = float(proc.get("fps_video") or 25) / max(1, int(proc.get("paso") or 1))
        for it in d.get("items", []):
            if it.get("clip"):
                webm = Path(it["clip"]).with_suffix(".webm").name
                if (estado.resultados / webm).exists():          # para navegadores sin H.264
                    it["clip_webm"] = "/resultados/" + webm
                it["clip"] = "/resultados/" + it["clip"]
        d["duracion"] = None
        if (estado.resultados / "camara.csv").exists():
            import pandas as pd
            cam = pd.read_csv(estado.resultados / "camara.csv", usecols=["t"])
            d["duracion"] = [float(cam["t"].min()), float(cam["t"].max())] if len(cam) else None
        return jsonify(_json(d))

    def _partido_var():
        """El análisis cargado con su revisión VAR (en caché mientras no cambie el tracking)."""
        from ..partido import Partido
        clave = (estado.resultados / "tracking.csv").stat().st_mtime
        if getattr(estado, "var_cache", (None,))[0] != clave:
            p = Partido.video(estado.resultados)
            p.var()
            estado.var_cache = (clave, p)
        return estado.var_cache[1]

    def _video_analizado(proc: dict) -> str | None:
        """El video del último análisis: el archivo, o la grabación de la señal en vivo."""
        if proc.get("video") and Path(proc["video"]).exists():
            return proc["video"]
        if proc.get("grabacion") and (estado.videos / proc["grabacion"]).exists():
            return str(estado.videos / proc["grabacion"])
        spec = estado.fuente_spec
        return spec if spec and es_archivo(spec) and Path(spec).exists() else None

    def _item_manual(p, rev, titulo):
        from ..analisis.var import para_repeticion
        from ..reportes.var import CLASE
        from ..vision.var_imagen import imagenes_var
        proc = p.info.get("proceso", {})
        nombres_jug = {i: f"#{int(i)}" for i in p.tracking["id"].dropna().unique() if float(i).is_integer()}
        datos = para_repeticion(rev, p._posiciones, nombres_jug, p.cancha.largo, p.cancha.ancho)
        foto = None
        video = _video_analizado(proc)
        if video and (rev.get("linea_x") is not None or rev.get("x") is not None):
            rev["t_max"] = None
            fotos = imagenes_var([rev], p.camara, video, estado.resultados, int(proc.get("ancho_proceso") or 1280),
                                 p.cancha.largo, p.cancha.ancho, float(proc.get("grabacion_desfase_s") or 0.0),
                                 prefijo=f"var_manual_{int(time.time() * 1000)}")
            if fotos:
                foto = "/resultados/" + list(fotos.values())[0]
        ver = rev.get("veredicto", "")
        return {"titulo": titulo, "veredicto": ver, "clase": CLASE.get(ver, ""), "texto": rev.get("explicacion", ""),
                "adelantados": rev.get("adelantados", []), "datos": datos, "foto": foto}

    @app.get("/api/var/cuadro.jpg")
    def api_var_cuadro():
        """Un cuadro del video analizado (para elegir el instante exacto cuadro a cuadro)."""
        from ..vision.var_imagen import leer_cuadro as leer
        if not (estado.resultados / "resumen_proceso.json").exists():
            abort(404)
        proc = json.loads((estado.resultados / "resumen_proceso.json").read_text(encoding="utf-8"))
        video = _video_analizado(proc)
        if video is None:
            abort(404)
        f = leer(video, float(request.args.get("t", 0)) - float(proc.get("grabacion_desfase_s") or 0.0))
        if f is None:
            abort(404)
        if f.shape[1] > 1280:
            f = cv2.resize(f, (1280, int(round(f.shape[0] * 1280 / f.shape[1]))), interpolation=cv2.INTER_AREA)
        return jpeg(f, 85)

    @app.post("/api/var/instante")
    def api_var_instante():
        """Revisión manual: la línea del fuera de lugar en el cuadro que elige el usuario."""
        from ..analisis.var import parametros_para, revisar_instante
        d = request.json or {}
        if not (estado.resultados / "tracking.csv").exists():
            return jsonify(error="Primero analiza un video"), 400
        try:
            t, equipo = float(d.get("t", 0)), d.get("equipo", "A")
            p = _partido_var()
            proc = p.info.get("proceso", {})
            par = parametros_para("video", float(p.info.get("fps") or 25), p.cancha.largo, p.cancha.ancho,
                                  bool(proc.get("camara_fija")))
            direccion = p.direccion().get(equipo, 1)
            rev = revisar_instante(p._posiciones, t, equipo, direccion, par)
            video = _video_analizado(proc)
            if video and rev.get("receptor") is not None:
                from ..vision.var_imagen import afinar_con_cuerpo
                afinar_con_cuerpo([rev], p.tracking, p.camara, video, int(proc.get("ancho_proceso") or 1280),
                                  float(proc.get("grabacion_desfase_s") or 0.0), par.k_dudoso)
            return jsonify(_json(_item_manual(p, rev, f"Revisión manual · ataca {p.nombres.get(equipo, equipo)}")))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return jsonify(error=str(e)), 400

    @app.post("/api/var/falta")
    def api_var_falta():
        """Falta marcada a mano sobre el cuadro: ¿dentro del área? ¿cortó una ocasión manifiesta de gol?"""
        from ..analisis.var import parametros_para, revisar_falta, revisar_ocasion
        from ..cancha import aplicar_h
        d = request.json or {}
        if not (estado.resultados / "tracking.csv").exists():
            return jsonify(error="Primero analiza un video"), 400
        try:
            t, equipo = float(d.get("t", 0)), d.get("equipo", "A")
            p = _partido_var()
            proc = p.info.get("proceso", {})
            cam = p.camara.sort_values("t")
            i = int(np.argmin(np.abs(cam["t"].to_numpy(float) - t)))
            fila = cam.iloc[i]
            if abs(float(fila["t"]) - t) > 0.3 or not bool(fila.get("calibrado", True)):
                return jsonify(error="La cámara no está calibrada en ese cuadro: elige otro (una toma amplia)."), 400
            H = fila[[f"h{j}" for j in range(9)]].to_numpy(float).reshape(3, 3)
            k = int(proc.get("ancho_proceso") or 1280) / float(d.get("ancho_imagen") or 1280)
            x, y = (float(v) for v in aplicar_h(H, [[float(d["u"]) * k, float(d["v"]) * k]])[0])
            par = parametros_para("video", float(p.info.get("fps") or 25), p.cancha.largo, p.cancha.ancho,
                                  bool(proc.get("camara_fija")))
            lado = p.direccion().get(equipo, 1)
            ins = p._posiciones.instante(t)
            sig = float(np.hypot(par.error_posicion_m, 0.4))           # el clic agrega su propio error
            f = revisar_falta(x, y, lado, par, sig)
            oc = revisar_ocasion(ins, equipo, lado, x, y, par)
            f.update(t=round(t, 3), equipo=equipo, direccion=lado, ocasion={k_: v for k_, v in oc.items() if k_ != "jugadores"},
                     jugadores=oc["jugadores"])
            if oc["veredicto"] != "no es ocasión manifiesta":
                f["explicacion"] += (" " if oc["veredicto"] == "ocasión manifiesta" else " ¿Ocasión manifiesta? ") + oc["explicacion"]
            else:
                f["explicacion"] += " " + oc["explicacion"]
            return jsonify(_json(_item_manual(p, f, f"Falta marcada a mano · a favor de {p.nombres.get(equipo, equipo)}")))
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            return jsonify(error=str(e)), 400

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
            estado.datos_hechos[f"{fuente}:{partido_id}"] = dict(estado.datos)
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
        hecho = estado.datos_hechos.get(f"{fuente}:{pid}")
        if hecho and not d.get("regenerar") and (estado.reportes / f"{fuente}_{pid}.html").exists():
            estado.datos = dict(hecho)          # los datos abiertos no cambian: el mismo partido no se recalcula
            return jsonify(ok=True, reutilizado=True)
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
