"""Fuentes de video: archivo, capturadora HDMI / cámara, pantalla del PC o stream de red.

    archivo:      ruta a un .mp4, .mkv, .mov, …
    capturadora:  disp:0, disp:1, …            (la tarjeta HDMI-USB se ve como una cámara)
    pantalla:     pantalla                     (monitor 1 completo)
                  pantalla:2                   (monitor 2)
                  pantalla:x,y,w,h             (región del monitor 1, p. ej. el reproductor de una página)
                  pantalla:2:x,y,w,h           (región del monitor 2)
    red:          rtsp://…, http(s)://…/video.m3u8

Nota sobre la TV y las páginas con video: los decodificadores de cable y muchos servicios de
streaming cifran la imagen (HDCP / DRM) y la capturadora o la captura de pantalla reciben negro.
Esta herramienta no evade esa protección: úsala con video que puedas ver y analizar.
"""
from __future__ import annotations

import re
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np


class Fuente:
    fps: float = 25.0
    total: int | None = None
    en_vivo: bool = False

    def __iter__(self) -> Iterator[tuple[int, float, np.ndarray]]:
        raise NotImplementedError

    def cerrar(self):
        pass


class FuenteArchivo(Fuente):
    def __init__(self, ruta: str, inicio_s: float = 0.0, fin_s: float | None = None):
        self.cap = cv2.VideoCapture(str(ruta))
        if not self.cap.isOpened():
            raise FileNotFoundError(f"No se pudo abrir el video: {ruta}")
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 25.0
        n = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        self.inicio = int(round(inicio_s * self.fps))
        self.fin = int(round(fin_s * self.fps)) if fin_s else n
        self.total = max(0, self.fin - self.inicio) if n else None
        if self.inicio:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.inicio)

    @property
    def ancho(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)

    def __iter__(self):
        i = self.inicio
        while i < self.fin or self.total is None:
            ok, frame = self.cap.read()
            if not ok:
                break
            yield i, i / self.fps, frame
            i += 1

    def cerrar(self):
        self.cap.release()


class _Ritmo:
    """Entrega cuadros a un ritmo fijo y lleva el tiempo real desde el inicio."""

    def __init__(self, fps: float):
        self.dt = 1.0 / max(fps, 0.1)
        self.t0 = time.time()

    def esperar(self, t_ini: float):
        espera = self.dt - (time.time() - t_ini)
        if espera > 0:
            time.sleep(espera)


class Grabadora:
    """Graba la señal en vivo tal como llega, en tiempo real (H.264).

    Escribe a un ritmo fijo según el reloj: si la señal trae menos cuadros se repite el último y si
    trae más se saltan, así un minuto de partido dura un minuto en el archivo aunque el análisis
    vaya más lento.
    """

    def __init__(self, ruta: str | Path, fps: float = 25.0):
        self.ruta, self.fps = Path(ruta), float(fps)
        self.proc = None
        self.forma = None
        self.t0 = None
        self.escritos = 0
        self.lock = threading.Lock()

    def escribir(self, frame: np.ndarray, ahora: float | None = None):
        ahora = time.time() if ahora is None else ahora
        with self.lock:
            if self.proc is None:
                from .pipeline import _escritor_h264
                self.ruta.parent.mkdir(parents=True, exist_ok=True)
                self.forma, self.t0 = frame.shape, ahora
                self.proc = _escritor_h264(self.ruta, frame.shape, self.fps, rapido=True)
            if frame.shape != self.forma:
                frame = cv2.resize(frame, (self.forma[1], self.forma[0]))
            objetivo = int((ahora - self.t0) * self.fps) + 1
            veces = min(objetivo - self.escritos, int(2 * self.fps))     # tras un corte largo no inundar
            if veces <= 0:
                return
            datos = np.ascontiguousarray(frame).tobytes()
            try:
                for _ in range(veces):
                    self.proc.stdin.write(datos)
            except (BrokenPipeError, OSError):
                return
            self.escritos = max(objetivo, self.escritos + veces)

    @property
    def segundos(self) -> float:
        return self.escritos / self.fps

    def cerrar(self):
        with self.lock:
            if self.proc is not None:
                try:
                    self.proc.stdin.close()
                except OSError:
                    pass
                self.proc.wait()
                self.proc = None


class _FuenteViva(Fuente):
    """Fuente en vivo: un hilo captura sin parar y guarda solo el último cuadro. Si el análisis
    va más lento que la señal se saltan cuadros en vez de acumular retraso; la grabación (si se
    pidió) se hace en ese mismo hilo, a ritmo real, independiente del análisis."""
    en_vivo = True

    def _iniciar(self, fps: float):
        self.fps = fps
        self._ultimo = None
        self._seq = 0
        self._t0 = time.time()
        self._t_ultimo = 0.0
        self._activo = True
        self.error: Exception | None = None
        self.grabadora: Grabadora | None = None
        self._lock = threading.Lock()
        self._hilo = threading.Thread(target=self._correr_hilo, daemon=True)
        self._hilo.start()

    def _correr_hilo(self):
        try:
            self._capturar()
        except Exception as e:  # noqa: BLE001
            self.error = e
        finally:
            self._activo = False

    def _capturar(self):          # se implementa en cada fuente: llama a _entregar(frame) por cuadro
        raise NotImplementedError

    def _entregar(self, frame: np.ndarray):
        ahora = time.time()
        with self._lock:
            self._ultimo, self._t_ultimo = frame, ahora - self._t0
            self._seq += 1
        g = self.grabadora
        if g is not None:
            g.escribir(frame, ahora)

    def grabar(self, ruta: str | Path, fps: float = 25.0) -> Grabadora:
        """Empieza a grabar la señal en `ruta` (.mp4)."""
        self.grabadora = Grabadora(ruta, fps)
        return self.grabadora

    def terminar_grabacion(self) -> Path | None:
        g, self.grabadora = self.grabadora, None
        if g is None:
            return None
        g.cerrar()
        return g.ruta if g.escritos else None

    def ultimo_cuadro(self, espera_s: float = 3.0) -> np.ndarray | None:
        """El cuadro más reciente sin consumirlo (para vistas previas)."""
        t = time.time()
        while time.time() - t < espera_s:
            with self._lock:
                if self._ultimo is not None:
                    return self._ultimo.copy()
            if not self._activo:
                break
            time.sleep(0.02)
        if self.error is not None:
            raise RuntimeError(str(self.error))
        return None

    def __iter__(self):
        ritmo = _Ritmo(self.fps)
        i, visto, avisado = 0, 0, False
        while True:
            t_ini = time.time()
            with self._lock:
                frame, t, seq = self._ultimo, self._t_ultimo, self._seq
            if seq == visto:
                if not self._activo:
                    if self.error is not None:
                        raise RuntimeError(f"La captura se detuvo: {self.error}")
                    break
                time.sleep(0.005)
                continue
            visto = seq
            if not avisado and frame.mean() < 3:
                avisado = True
                print("Aviso: imagen negra. Si la fuente es un decodificador o app con DRM, "
                      "la señal está protegida (HDCP) y no se puede capturar.")
            yield i, t, frame
            i += 1
            ritmo.esperar(t_ini)

    def cerrar(self):
        self._activo = False
        if self._hilo.is_alive() and threading.current_thread() is not self._hilo:
            self._hilo.join(timeout=3)
        self.terminar_grabacion()


class FuenteDispositivo(_FuenteViva):
    """Capturadora HDMI-USB, cámara (índice 0, 1, …) o stream de red (rtsp://, http://)."""

    def __init__(self, origen: int | str = 0, ancho: int = 1920, alto: int = 1080, fps: float = 10.0):
        if isinstance(origen, int) and sys.platform.startswith("win"):
            self.cap = cv2.VideoCapture(origen, cv2.CAP_DSHOW)   # más rápido y fiable en Windows
        else:
            self.cap = cv2.VideoCapture(origen)
        if isinstance(origen, int):
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, ancho)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, alto)
        if not self.cap.isOpened():
            raise RuntimeError(f"No se pudo abrir la fuente de video {origen}")
        self._iniciar(fps)

    def _capturar(self):
        while self._activo:
            ok, frame = self.cap.read()       # al ritmo del dispositivo (30 o 60 cuadros/s)
            if not ok:
                break
            self._entregar(frame)

    def cerrar(self):
        super().cerrar()
        self.cap.release()


# ---------------------------------------------------------------------- pantalla
def _mss():
    try:
        import mss
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("Para capturar la pantalla instala 'mss': pip install mss") from e
    try:
        return (getattr(mss, "MSS", None) or mss.mss)()
    except Exception as e:  # noqa: BLE001  (sin pantalla: servidor, sesión remota sin escritorio)
        raise RuntimeError(f"No se pudo acceder a la pantalla: {e}") from e


def _foto(sct, zona) -> np.ndarray:
    return np.ascontiguousarray(np.array(sct.grab(zona))[..., :3])


def listar_pantallas() -> list[dict]:
    """Monitores del PC: [{indice, x, y, ancho, alto}] (el 1 es el principal)."""
    with _mss() as sct:
        return [{"indice": k, "x": m["left"], "y": m["top"], "ancho": m["width"], "alto": m["height"]}
                for k, m in enumerate(sct.monitors) if k >= 1]


def captura_pantalla(monitor: int = 1, region: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Un cuadro de la pantalla (BGR). `region` = (x, y, ancho, alto) relativa al monitor."""
    with _mss() as sct:
        return _foto(sct, _zona(sct, monitor, region))


def _zona(sct, monitor: int, region):
    if monitor < 1 or monitor >= len(sct.monitors):
        raise RuntimeError(f"No existe el monitor {monitor} (hay {len(sct.monitors) - 1})")
    mon = sct.monitors[monitor]
    if not region:
        return dict(mon)
    x, y, w, h = (int(v) for v in region)
    x, y = max(0, x), max(0, y)
    w, h = min(w, mon["width"] - x), min(h, mon["height"] - y)
    if w < 16 or h < 16:
        raise RuntimeError("La región de la pantalla es demasiado chica")
    return {"left": mon["left"] + x, "top": mon["top"] + y, "width": w, "height": h}


def _crecer_hasta_el_fondo(gris: np.ndarray, x1: int, y1: int, x2: int, y2: int, umbral: float = 3.0,
                           maximo: float = 0.35) -> tuple[int, int, int, int]:
    """Agranda el rectángulo hasta el borde del reproductor: mientras la fila (o columna) siguiente
    tenga contenido; se detiene en el fondo liso de la página o en el borde de la pantalla."""
    h, w = gris.shape
    lim_y, lim_x = int(maximo * (y2 - y1)) + 2, int(maximo * (x2 - x1)) + 2
    g = gris.astype(np.float32)
    k = 0
    while y1 > 0 and k < lim_y and g[y1 - 1, x1:x2].std() > umbral:
        y1, k = y1 - 1, k + 1
    k = 0
    while y2 < h and k < lim_y and g[y2, x1:x2].std() > umbral:
        y2, k = y2 + 1, k + 1
    k = 0
    while x1 > 0 and k < lim_x and g[y1:y2, x1 - 1].std() > umbral:
        x1, k = x1 - 1, k + 1
    k = 0
    while x2 < w and k < lim_x and g[y1:y2, x2].std() > umbral:
        x2, k = x2 + 1, k + 1
    return x1, y1, x2, y2


def sugerir_zona(monitor: int = 1, segundos: float = 1.5) -> dict | None:
    """Encuentra el reproductor de video en la pantalla: la zona que cambia mientras el resto de la
    página queda quieto (con una cámara fija se mueven solo los jugadores, así que se juntan sus
    movimientos y se suma la zona de césped), y después se ajusta al borde del reproductor, donde
    empieza el fondo liso de la página. Devuelve {x, y, ancho, alto, metodo} en píxeles del monitor."""
    from .deteccion import mascara_cesped
    with _mss() as sct:
        zona = _zona(sct, monitor, None)
        cuadros, t = [], time.time()
        while time.time() - t < segundos or len(cuadros) < 5:
            cuadros.append(_foto(sct, zona))
            time.sleep(0.12)
    esc = 640.0 / cuadros[0].shape[1]
    peq = [cv2.resize(c, None, fx=esc, fy=esc, interpolation=cv2.INTER_AREA) for c in cuadros]
    grises = [cv2.GaussianBlur(cv2.cvtColor(q, cv2.COLOR_BGR2GRAY), (3, 3), 0) for q in peq]
    mov = np.zeros_like(grises[0])
    for a, b in zip(grises, grises[1:]):
        mov = np.maximum(mov, cv2.absdiff(a, b))
    mascara = cv2.morphologyEx((mov > 10).astype(np.uint8), cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    _, campo = mascara_cesped(peq[-1])
    verde = (campo > 0).astype(np.uint8)
    caja, metodo = None, None
    if mascara.mean() > 0.002:
        # los movimientos sueltos (jugadores) se juntan en un solo grupo
        k = max(5, int(0.06 * mascara.shape[1]))
        junto = cv2.dilate(mascara, cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(junto)
        if n > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            ys_, xs_ = np.nonzero(mascara & (lab == i))
            if len(xs_):
                caja, metodo = [xs_.min(), ys_.min(), xs_.max() + 1, ys_.max() + 1], "movimiento"
    n, lab, stats, _ = cv2.connectedComponentsWithStats(verde)
    if n > 1:
        cand = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] > 0.01 * verde.size]
        if caja is not None:            # el césped que toca la zona en movimiento
            x1, y1, x2, y2 = caja
            cand = [i for i in cand if (lab[y1:y2, x1:x2] == i).any()]
        if cand:
            i = max(cand, key=lambda i_: stats[i_, cv2.CC_STAT_AREA])
            x, y, w, h = stats[i, :4]
            g = [x, y, x + w, y + h]
            caja = g if caja is None else [min(caja[0], g[0]), min(caja[1], g[1]), max(caja[2], g[2]), max(caja[3], g[3])]
            metodo = metodo or "césped"
    if caja is None:
        return None
    x1, y1, x2, y2 = _crecer_hasta_el_fondo(grises[-1], *(int(v) for v in caja))
    w, h = x2 - x1, y2 - y1
    if w * h < 0.02 * mascara.size or min(w, h) < 20:
        return None
    return {"x": int(round(x1 / esc)), "y": int(round(y1 / esc)), "ancho": int(round(w / esc)),
            "alto": int(round(h / esc)), "metodo": metodo}


class FuentePantalla(_FuenteViva):
    """Una región de la pantalla: el reproductor de una página web con el partido en vivo,
    una app de streaming o la ventana de cualquier reproductor."""

    def __init__(self, monitor: int = 1, region: tuple[int, int, int, int] | None = None, fps: float = 10.0,
                 fps_captura: float = 25.0):
        self.monitor, self.region, self.fps_captura = monitor, region, fps_captura
        with _mss() as sct:                 # valida monitor y región antes de empezar
            self.zona = _zona(sct, monitor, region)
        self._iniciar(fps)

    def _capturar(self):
        # mss debe usarse en el mismo hilo que lo crea (Windows): se crea dentro del hilo
        dt = 1.0 / max(self.fps, self.fps_captura, 0.1)
        with _mss() as sct:
            while self._activo:
                t_ini = time.time()
                self._entregar(_foto(sct, self.zona))
                espera = dt - (time.time() - t_ini)
                if espera > 0:
                    time.sleep(espera)


def interpretar_pantalla(spec: str) -> tuple[int, tuple[int, int, int, int] | None]:
    """'pantalla', 'pantalla:2', 'pantalla:x,y,w,h' o 'pantalla:2:x,y,w,h' -> (monitor, región)."""
    partes = spec.split(":")[1:]
    monitor, region = 1, None
    for p in partes:
        if "," in p:
            region = tuple(int(float(v)) for v in p.split(","))
            if len(region) != 4:
                raise ValueError(f"Región inválida: {p} (se espera x,y,ancho,alto)")
        elif p.strip():
            monitor = int(p)
    return monitor, region


# ---------------------------------------------------------------------- capturadoras
def nombres_dispositivos() -> list[str]:
    """Nombres de las cámaras y capturadoras en el orden en que las numera el sistema
    (lo mejor posible: en Windows y macOS se consultan con ffmpeg, en Linux con /sys)."""
    try:
        if sys.platform.startswith("linux"):
            out = []
            for d in sorted(Path("/sys/class/video4linux").glob("video*"), key=lambda p: int(re.sub(r"\D", "", p.name) or 0)):
                out.append((d / "name").read_text().strip() if (d / "name").exists() else d.name)
            return out
        from .pipeline import ruta_ffmpeg
        if sys.platform.startswith("win"):
            r = subprocess.run([ruta_ffmpeg(), "-hide_banner", "-list_devices", "true", "-f", "dshow", "-i", "dummy"],
                               capture_output=True, text=True, timeout=10, errors="ignore")
            return re.findall(r'"([^"]+)"\s*\(video\)', r.stderr)
        if sys.platform == "darwin":
            r = subprocess.run([ruta_ffmpeg(), "-hide_banner", "-f", "avfoundation", "-list_devices", "true", "-i", ""],
                               capture_output=True, text=True, timeout=10, errors="ignore")
            video = r.stderr.split("audio devices")[0]
            return re.findall(r"\]\s*\[\d+\]\s*(.+)", video)
    except Exception:  # noqa: BLE001
        pass
    return []


def listar_dispositivos(maximo: int = 6) -> list[dict]:
    """Prueba los índices 0…maximo-1 y devuelve los que entregan imagen:
    [{indice, nombre, ancho, alto, negro, cuadro}]."""
    nombres = nombres_dispositivos()
    out = []
    if sys.platform.startswith("linux") and not nombres:
        return out                                   # Linux sin /dev/video*: no hay nada que probar
    try:                                             # que OpenCV no llene la consola al probar índices vacíos
        nivel = cv2.utils.logging.getLogLevel()
        cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_SILENT)
    except AttributeError:
        nivel = None
    try:
        return _probar_dispositivos(nombres, maximo, out)
    finally:
        if nivel is not None:
            cv2.utils.logging.setLogLevel(nivel)


def _probar_dispositivos(nombres: list[str], maximo: int, out: list) -> list[dict]:
    for i in range(maximo):
        if sys.platform.startswith("linux") and nombres and i >= len(nombres):
            break
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW) if sys.platform.startswith("win") else cv2.VideoCapture(i)
        try:
            if not cap.isOpened():
                continue
            frame = None
            for _ in range(5):              # los primeros cuadros de algunas capturadoras vienen vacíos
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                    if f.mean() > 3:
                        break
            if frame is None:
                continue
            out.append({"indice": i, "nombre": nombres[i] if i < len(nombres) else f"Dispositivo {i}",
                        "ancho": int(frame.shape[1]), "alto": int(frame.shape[0]), "negro": bool(frame.mean() < 3),
                        "cuadro": frame})
        finally:
            cap.release()
    return out


# ---------------------------------------------------------------------- apertura
def es_archivo(spec: str) -> bool:
    return not (spec.startswith("pantalla") or spec.startswith("disp:") or "://" in spec)


def abrir_fuente(spec: str, **kw) -> Fuente:
    """Abre cualquier fuente a partir de su descripción (ver el inicio del módulo)."""
    if spec.startswith("pantalla"):
        monitor, region = interpretar_pantalla(spec)
        return FuentePantalla(monitor, region, fps=kw.get("fps", 10.0), fps_captura=kw.get("fps_captura", 25.0))
    if spec.startswith("disp:"):
        return FuenteDispositivo(int(spec.split(":")[1]), ancho=kw.get("ancho", 1920), alto=kw.get("alto", 1080),
                                 fps=kw.get("fps", 10.0))
    if "://" in spec:
        return FuenteDispositivo(spec, fps=kw.get("fps", 10.0))
    return FuenteArchivo(spec, kw.get("inicio_s", 0.0), kw.get("fin_s"))
