"""Fuentes de video: archivo, capturadora/cámara (índice), stream de red o región de la pantalla.

Nota sobre la TV: los decodificadores de cable y las apps con DRM cifran la salida
HDMI (HDCP) y las capturadoras reciben pantalla negra. Esta herramienta no evade
esa protección: úsala con video que tengas derecho a analizar (grabaciones propias,
partidos amateur, material del club, transmisiones sin protección).
"""
from __future__ import annotations

import sys
import threading
import time
from typing import Iterator

import cv2
import numpy as np


class Fuente:
    fps: float = 25.0
    total: int | None = None

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


class FuenteDispositivo(Fuente):
    """Capturadora HDMI-USB, cámara (índice 0, 1, …) o stream de red (rtsp://, http://).

    Un hilo lee continuamente y se guarda solo el último cuadro: si el análisis va más
    lento que el video, se saltan cuadros en vez de acumular retraso.
    """

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
        self.fps = fps
        self._ultimo = None
        self._activo = True
        self._lock = threading.Lock()
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    def _leer(self):
        while self._activo:
            ok, frame = self.cap.read()
            if not ok:
                self._activo = False
                break
            with self._lock:
                self._ultimo = frame

    def __iter__(self):
        i, t0, dt = 0, time.time(), 1.0 / self.fps
        avisado = False
        while self._activo or self._ultimo is not None:
            t_ini = time.time()
            with self._lock:
                frame, self._ultimo = self._ultimo, None
            if frame is None:
                if not self._activo:
                    break
                time.sleep(0.005)
                continue
            if frame.mean() < 3 and not avisado:
                avisado = True
                print("Aviso: imagen negra. Si la fuente es un decodificador o app con DRM, "
                      "la señal está protegida (HDCP) y no se puede capturar.")
            yield i, time.time() - t0, frame
            i += 1
            espera = dt - (time.time() - t_ini)
            if espera > 0:
                time.sleep(espera)

    def cerrar(self):
        self._activo = False
        self.cap.release()


class FuentePantalla(Fuente):
    """Región de la pantalla (p. ej. un reproductor de video en el PC)."""

    def __init__(self, region: tuple[int, int, int, int] | None = None, fps: float = 15.0):
        try:
            import mss
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("Para capturar la pantalla instala 'mss': pip install mss") from e
        self.sct = mss.mss()
        mon = self.sct.monitors[1]
        if region:
            x, y, w, h = region
            self.region = {"left": x, "top": y, "width": w, "height": h}
        else:
            self.region = mon
        self.fps = fps

    def __iter__(self):
        i, t0, dt = 0, time.time(), 1.0 / self.fps
        while True:
            t_ini = time.time()
            frame = np.array(self.sct.grab(self.region))[..., :3]
            yield i, time.time() - t0, np.ascontiguousarray(frame)
            i += 1
            espera = dt - (time.time() - t_ini)
            if espera > 0:
                time.sleep(espera)


def es_archivo(spec: str) -> bool:
    return not (spec.startswith("pantalla") or spec.startswith("disp:") or "://" in spec)


def abrir_fuente(spec: str, **kw) -> Fuente:
    """'pantalla', 'pantalla:x,y,w,h', 'disp:0', 'rtsp://…' o una ruta de archivo."""
    if spec.startswith("pantalla"):
        region = None
        if ":" in spec:
            region = tuple(int(v) for v in spec.split(":", 1)[1].split(","))
        return FuentePantalla(region, fps=kw.get("fps", 15.0))
    if spec.startswith("disp:"):
        return FuenteDispositivo(int(spec.split(":")[1]), fps=kw.get("fps", 10.0))
    if "://" in spec:
        return FuenteDispositivo(spec, fps=kw.get("fps", 10.0))
    return FuenteArchivo(spec, kw.get("inicio_s", 0.0), kw.get("fin_s"))
