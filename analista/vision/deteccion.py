"""Detección de jugadores y balón en un cuadro de video.

Dos detectores con la misma interfaz:
  - DetectorYOLO: red neuronal (Ultralytics). Es el recomendado para video real.
  - DetectorCesped: visión clásica (lo que no es césped dentro de la cancha).
    No necesita GPU ni modelos; sirve como respaldo y para pruebas.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class Detecciones:
    cajas: np.ndarray        # (N, 4) x1, y1, x2, y2 en píxeles
    confianza: np.ndarray    # (N,)
    es_balon: np.ndarray     # (N,) bool

    @staticmethod
    def vacias() -> "Detecciones":
        return Detecciones(np.zeros((0, 4)), np.zeros(0), np.zeros(0, bool))

    def jugadores(self) -> "Detecciones":
        m = ~self.es_balon
        return Detecciones(self.cajas[m], self.confianza[m], self.es_balon[m])

    def balones(self) -> "Detecciones":
        m = self.es_balon
        return Detecciones(self.cajas[m], self.confianza[m], self.es_balon[m])

    def __len__(self) -> int:
        return len(self.cajas)

    @property
    def pies(self) -> np.ndarray:
        """Punto de apoyo (centro del borde inferior), que es lo que se proyecta a la cancha."""
        c = self.cajas
        return np.stack([(c[:, 0] + c[:, 2]) / 2, c[:, 3]], 1) if len(c) else np.zeros((0, 2))

    @property
    def centros(self) -> np.ndarray:
        c = self.cajas
        return np.stack([(c[:, 0] + c[:, 2]) / 2, (c[:, 1] + c[:, 3]) / 2], 1) if len(c) else np.zeros((0, 2))


# ------------------------------------------------------------------ máscaras
def mascara_cesped(frame: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Devuelve (césped, campo): píxeles verdes y la región de juego rellena."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    cesped = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    cerrado = cv2.morphologyEx(cesped, cv2.MORPH_CLOSE, k)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cerrado, 8)
    campo = np.zeros_like(cesped)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        umbral = max(areas.max() * 0.15, cesped.size * 0.01)
        for i in np.where(areas >= umbral)[0] + 1:
            campo[lab == i] = 255
        # Rellenar huecos (jugadores, líneas) con la envolvente convexa de cada región
        contornos, _ = cv2.findContours(campo, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contornos:
            cv2.fillConvexPoly(campo, cv2.convexHull(c), 255)
    return cesped, campo


def mascara_lineas(frame: np.ndarray, campo: np.ndarray, excluir: np.ndarray | None = None) -> np.ndarray:
    """Líneas blancas delgadas dentro del campo (top-hat sobre luminancia)."""
    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    tophat = cv2.morphologyEx(gris, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    blanco = (hsv[..., 1] < 70) & (hsv[..., 2] > 140)
    m = ((tophat > 25) & blanco & (campo > 0)).astype(np.uint8) * 255
    if excluir is not None:
        m[excluir > 0] = 0
    return m


def mascara_cajas(forma, cajas: np.ndarray, holgura: float = 0.15) -> np.ndarray:
    m = np.zeros(forma[:2], np.uint8)
    h, w = forma[:2]
    for x1, y1, x2, y2 in np.asarray(cajas).reshape(-1, 4):
        dx, dy = (x2 - x1) * holgura, (y2 - y1) * holgura
        cv2.rectangle(m, (int(max(0, x1 - dx)), int(max(0, y1 - dy))),
                      (int(min(w - 1, x2 + dx)), int(min(h - 1, y2 + dy))), 255, -1)
    return m


# ------------------------------------------------------------------ detectores
class DetectorYOLO:
    """Personas (clase 0) y balón (clase 32) con un modelo COCO de Ultralytics.

    Para mejor resultado, entrena o descarga un modelo afinado para fútbol
    (jugador / arquero / árbitro / balón) y pásalo en `modelo`.
    """

    def __init__(self, modelo: str = "yolo11n.pt", imgsz: int = 1280, conf: float = 0.15,
                 clases_persona=(0,), clases_balon=(32,), dispositivo: str | None = None, conf_balon: float = 0.15):
        try:
            from ultralytics import YOLO  # import diferido: solo si se usa
        except ImportError as e:  # pragma: no cover
            raise RuntimeError("El detector YOLO necesita ultralytics: pip install ultralytics "
                               "(o usa el detector clásico: --detector cesped)") from e
        self.modelo = YOLO(modelo)
        self.imgsz, self.conf = imgsz, conf
        # Personas con confianza baja (jugadores lejanos); el balón exige más: con confianza baja la red
        # "ve" balones en medias, camisetas blancas y marcas de la cancha
        self.conf_balon = conf_balon
        self.clases_persona, self.clases_balon = set(clases_persona), set(clases_balon)
        self.dispositivo = dispositivo
        self.ultimas_crudas = Detecciones.vacias()

    def __call__(self, frame: np.ndarray, campo: np.ndarray | None = None) -> Detecciones:
        clases = sorted(self.clases_persona | self.clases_balon)
        r = self.modelo.predict(frame, imgsz=self.imgsz, conf=self.conf, classes=clases,
                                verbose=False, device=self.dispositivo)[0]
        if r.boxes is None or len(r.boxes) == 0:
            self.ultimas_crudas = Detecciones.vacias()
            return Detecciones.vacias()
        cajas = r.boxes.xyxy.cpu().numpy()
        conf = r.boxes.conf.cpu().numpy()
        cls = r.boxes.cls.cpu().numpy().astype(int)
        es_balon = np.isin(cls, list(self.clases_balon))
        ok = ~es_balon | (conf >= self.conf_balon)
        cajas, conf, es_balon = cajas[ok], conf[ok], es_balon[ok]
        det = Detecciones(cajas, conf, es_balon)
        self.ultimas_crudas = det          # antes de quitar público, bancos y gráficos
        return filtrar_en_campo(det, campo)


class DetectorCesped:
    """Detector clásico: manchas que no son césped dentro de la cancha."""

    def __init__(self, area_min: int = 40, alto_min: int = 10):
        self.area_min, self.alto_min = area_min, alto_min
        self.excluir_balon = None

    def __call__(self, frame: np.ndarray, campo: np.ndarray | None = None,
                 cesped: np.ndarray | None = None, alto_esperado=None) -> Detecciones:
        if cesped is None or campo is None:
            cesped, campo = mascara_cesped(frame)
        no_cesped = ((cesped == 0) & (campo > 0)).astype(np.uint8) * 255
        # La apertura elimina estructuras delgadas (las líneas) y deja cuerpos
        abierto = cv2.morphologyEx(no_cesped, cv2.MORPH_OPEN,
                                   cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        abierto = cv2.morphologyEx(abierto, cv2.MORPH_CLOSE,
                                   cv2.getStructuringElement(cv2.MORPH_RECT, (3, 7)))
        n, _, st, _ = cv2.connectedComponentsWithStats(abierto, 8)
        cajas, conf = [], []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if a < self.area_min or h < self.alto_min or h < 0.8 * w:
                # una mancha ancha puede ser 2-3 jugadores juntos: se parte abajo
                if not (a >= self.area_min and h >= self.alto_min and w <= 3.5 * h):
                    continue
            esperado = alto_esperado((x + w / 2, y + h)) if alto_esperado else None
            if esperado and not (0.45 * esperado <= h <= 2.2 * esperado):
                continue
            # Un jugador con brazos abiertos mide ~0.55 de su alto; solo se parte la mancha
            # si es claramente más ancha (dos o más jugadores pegados)
            ancho_tipico = (esperado or h) * 0.55
            partes = 1 if w < 1.7 * ancho_tipico else max(2, int(round(w / ancho_tipico)))
            for k in range(partes):
                x1 = x + k * w / partes
                cajas.append([x1, y, x1 + w / partes, y + h])
                conf.append(min(0.95, 0.5 + a / (w * h + 1) * 0.5) * (1.0 if partes == 1 else 0.7))
        det_j = Detecciones(np.array(cajas, float).reshape(-1, 4), np.array(conf, float),
                            np.zeros(len(cajas), bool))
        det_b = self._balon(frame, campo, det_j, alto_esperado, self.excluir_balon, cesped)
        return Detecciones(np.vstack([det_j.cajas, det_b.cajas]),
                           np.concatenate([det_j.confianza, det_b.confianza]),
                           np.concatenate([det_j.es_balon, det_b.es_balon]))

    def _balon(self, frame, campo, jugadores: Detecciones, alto_esperado,
               excluir: np.ndarray | None = None, cesped: np.ndarray | None = None) -> Detecciones:
        """Balón = mancha clara, pequeña y redonda, rodeada de césped, que no es línea,
        gráfico de TV ni parte de un jugador."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Blanco "absoluto" o mancha clara pequeña sobre el fondo (top-hat): un balón lejano
        # mide 3-6 px, sale borroso y teñido de verde, y no pasa un umbral fijo de blanco
        k = max(9, frame.shape[1] // 80) | 1
        tophat = cv2.morphologyEx(hsv[..., 2], cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        blanco = (((hsv[..., 1] < 60) & (hsv[..., 2] > 170)) | ((hsv[..., 1] < 110) & (tophat > 40))) & (campo > 0)
        blanco = blanco.astype(np.uint8) * 255
        if cesped is None:
            cesped = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255))
        H_img, W_img = frame.shape[:2]
        n, lab, st, cen = cv2.connectedComponentsWithStats(blanco, 8)
        cajas, conf = [], []
        for i in range(1, n):
            x, y, w, h, a = st[i]
            if not (3 <= a <= 400):
                continue
            # Anillo alrededor de la mancha: un balón en juego está rodeado de césped
            # (descarta letras de marcadores, medias, publicidad y gráficos)
            r = max(3, int(round(max(w, h) * 0.9)))
            ax1, ay1 = max(0, x - r), max(0, y - r)
            ax2, ay2 = min(W_img, x + w + r), min(H_img, y + h + r)
            anillo = cesped[ay1:ay2, ax1:ax2] > 0
            interior = np.zeros_like(anillo)
            interior[max(0, y - ay1 - 1):y - ay1 + h + 1, max(0, x - ax1 - 1):x - ax1 + w + 1] = True
            frac_cesped = float(anillo[~interior].mean()) if (~interior).any() else 0.0
            if frac_cesped < 0.45:
                continue
            lado_mayor, lado_menor = max(w, h), max(1, min(w, h))
            if lado_mayor > 1.6 * lado_menor + 1:        # alargado: línea, media, short
                continue
            relleno = a / float(w * h)
            if relleno < 0.5:
                continue
            cx, cy = cen[i]
            ajuste = 1.0
            if alto_esperado:
                diam = alto_esperado((cx, y + h)) * 0.115   # 0.22 m frente a ~1.9 m de jugador
                if not (0.45 * diam <= lado_mayor <= 2.2 * diam):
                    continue
                ajuste = 1.0 - min(0.6, abs(np.log(lado_mayor / diam)) * 0.5)
            if excluir is not None and excluir[int(cy), int(cx)] > 0:   # sobre una línea o punto marcado
                continue
            # Dentro del cuerpo de un jugador (camiseta, short o medias blancas)
            if any(x1 + 0.1 * (x2 - x1) <= cx <= x2 - 0.1 * (x2 - x1) and y1 <= cy <= y2 - 0.04 * (y2 - y1)
                   for x1, y1, x2, y2 in jugadores.cajas):
                continue
            cajas.append([x - 1, y - 1, x + w + 1, y + h + 1])
            conf.append(float(np.clip((0.15 + 0.5 * relleno * ajuste) * (0.6 + 0.4 * frac_cesped) + 0.1, 0, 0.95)))
        return Detecciones(np.array(cajas, float).reshape(-1, 4), np.array(conf, float),
                           np.ones(len(cajas), bool))


def filtrar_en_campo(det: Detecciones, campo: np.ndarray | None, tolerancia: int = 6) -> Detecciones:
    """Descarta detecciones cuyos pies caen fuera del campo (público, bancos, gráficos)."""
    if campo is None or len(det) == 0:
        return det
    h, w = campo.shape[:2]
    campo_dil = cv2.dilate(campo, np.ones((tolerancia * 2 + 1, tolerancia * 2 + 1), np.uint8))
    p = det.pies
    u = np.clip(p[:, 0].astype(int), 0, w - 1)
    v = np.clip(p[:, 1].astype(int), 0, h - 1)
    ok = campo_dil[v, u] > 0
    ok |= det.es_balon & (campo_dil[np.clip(det.centros[:, 1].astype(int), 0, h - 1), u] > 0)
    return Detecciones(det.cajas[ok], det.confianza[ok], det.es_balon[ok])
