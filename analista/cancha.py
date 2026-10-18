"""Modelo geométrico de la cancha y utilidades de homografía.

Convención de coordenadas de todo el paquete (igual que SkillCorner):
  - Origen en el centro de la cancha, unidades en metros.
  - x a lo largo de la cancha (de -L/2 a L/2), y a lo ancho (de -W/2 a W/2).
  - y positivo apunta hacia la banda lejana a la cámara principal (arriba en los mapas).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Cancha:
    largo: float = 105.0
    ancho: float = 68.0
    area_grande_fondo: float = 16.5
    area_grande_ancho: float = 40.32
    area_chica_fondo: float = 5.5
    area_chica_ancho: float = 18.32
    radio_circulo: float = 9.15
    distancia_penal: float = 11.0
    ancho_arco: float = 7.32

    @property
    def L2(self) -> float:
        return self.largo / 2

    @property
    def W2(self) -> float:
        return self.ancho / 2

    # ------------------------------------------------------------------ líneas
    def lineas(self, paso: float = 0.5) -> list[np.ndarray]:
        """Polilíneas (N x 2, metros) con todas las marcas de la cancha."""
        L2, W2 = self.L2, self.W2
        pol: list[np.ndarray] = []

        def seg(p, q):
            n = max(2, int(np.hypot(q[0] - p[0], q[1] - p[1]) / paso) + 1)
            t = np.linspace(0, 1, n)[:, None]
            return (1 - t) * np.array(p) + t * np.array(q)

        def arco(cx, cy, r, a0, a1):
            n = max(8, int(abs(a1 - a0) * r / paso))
            a = np.linspace(a0, a1, n)
            return np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], 1)

        pol += [seg((-L2, -W2), (L2, -W2)), seg((L2, -W2), (L2, W2)),
                seg((L2, W2), (-L2, W2)), seg((-L2, W2), (-L2, -W2)),
                seg((0, -W2), (0, W2))]
        pol.append(arco(0, 0, self.radio_circulo, 0, 2 * np.pi))
        for s in (-1, 1):  # s=-1 arco izquierdo, s=+1 derecho
            xg = s * L2
            ag, ac = self.area_grande_ancho / 2, self.area_chica_ancho / 2
            xa = xg - s * self.area_grande_fondo
            xc = xg - s * self.area_chica_fondo
            pol += [seg((xg, -ag), (xa, -ag)), seg((xa, -ag), (xa, ag)), seg((xa, ag), (xg, ag))]
            pol += [seg((xg, -ac), (xc, -ac)), seg((xc, -ac), (xc, ac)), seg((xc, ac), (xg, ac))]
            # Medialuna: parte del círculo del penal que queda fuera del área
            xp = xg - s * self.distancia_penal
            dx = self.area_grande_fondo - self.distancia_penal
            ang = np.arccos(dx / self.radio_circulo)
            if s < 0:
                pol.append(arco(xp, 0, self.radio_circulo, -ang, ang))
            else:
                pol.append(arco(xp, 0, self.radio_circulo, np.pi - ang, np.pi + ang))
        return pol

    # ------------------------------------------------------- puntos de referencia
    def puntos_referencia(self) -> dict[str, tuple[float, float]]:
        """Puntos fáciles de identificar en una imagen, para calibrar."""
        L2, W2 = self.L2, self.W2
        ag, ac = self.area_grande_ancho / 2, self.area_chica_ancho / 2
        r = self.radio_circulo
        p = {
            "centro": (0.0, 0.0),
            "mitad_banda_cercana": (0.0, -W2),
            "mitad_banda_lejana": (0.0, W2),
            "circulo_cercano": (0.0, -r),
            "circulo_lejano": (0.0, r),
            "circulo_izquierdo": (-r, 0.0),
            "circulo_derecho": (r, 0.0),
        }
        for lado, s in (("izq", -1), ("der", 1)):
            xg = s * L2
            xa = xg - s * self.area_grande_fondo
            xc = xg - s * self.area_chica_fondo
            p.update({
                f"esquina_{lado}_cercana": (xg, -W2),
                f"esquina_{lado}_lejana": (xg, W2),
                f"area_grande_{lado}_fondo_cercana": (xg, -ag),
                f"area_grande_{lado}_fondo_lejana": (xg, ag),
                f"area_grande_{lado}_frente_cercana": (xa, -ag),
                f"area_grande_{lado}_frente_lejana": (xa, ag),
                f"area_chica_{lado}_fondo_cercana": (xg, -ac),
                f"area_chica_{lado}_fondo_lejana": (xg, ac),
                f"area_chica_{lado}_frente_cercana": (xc, -ac),
                f"area_chica_{lado}_frente_lejana": (xc, ac),
                f"punto_penal_{lado}": (xg - s * self.distancia_penal, 0.0),
            })
        return p

    def dentro(self, xy: np.ndarray, margen: float = 3.0) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        return (np.abs(xy[:, 0]) <= self.L2 + margen) & (np.abs(xy[:, 1]) <= self.W2 + margen)


# ------------------------------------------------------------------- homografías
def aplicar_h(H: np.ndarray, pts) -> np.ndarray:
    """Aplica una homografía 3x3 a puntos (N x 2)."""
    pts = np.asarray(pts, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return pts
    hom = np.hstack([pts, np.ones((len(pts), 1))]) @ np.asarray(H, float).T
    w = hom[:, 2:3]
    w = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return hom[:, :2] / w


def h_desde_puntos(img_pts, cancha_pts) -> np.ndarray:
    """Homografía imagen -> cancha a partir de >= 4 correspondencias."""
    img_pts = np.asarray(img_pts, np.float64).reshape(-1, 2)
    cancha_pts = np.asarray(cancha_pts, np.float64).reshape(-1, 2)
    if len(img_pts) < 4:
        raise ValueError("Se necesitan al menos 4 puntos para calibrar")
    metodo = cv2.RANSAC if len(img_pts) > 4 else 0
    H, _ = cv2.findHomography(img_pts, cancha_pts, metodo, 1.0)
    if H is None:
        raise ValueError("No se pudo estimar la homografía: revisa que los puntos no sean colineales")
    return H / H[2, 2]


# ------------------------------------------------------------------ dibujo 2D
@dataclass
class LienzoCancha:
    """Imagen cenital de la cancha con su transformación metros -> píxeles."""
    cancha: Cancha = field(default_factory=Cancha)
    ppm: float = 6.0          # píxeles por metro
    margen: float = 4.0       # metros alrededor de la cancha
    color_fondo: tuple = (60, 120, 60)
    color_linea: tuple = (235, 235, 235)

    def __post_init__(self):
        c = self.cancha
        self.ancho_px = int(round((c.largo + 2 * self.margen) * self.ppm))
        self.alto_px = int(round((c.ancho + 2 * self.margen) * self.ppm))

    def a_px(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float).reshape(-1, 2)
        u = (xy[:, 0] + self.cancha.L2 + self.margen) * self.ppm
        v = (self.cancha.W2 + self.margen - xy[:, 1]) * self.ppm  # y lejana arriba
        return np.stack([u, v], 1)

    def de_px(self, uv) -> np.ndarray:
        uv = np.asarray(uv, float).reshape(-1, 2)
        x = uv[:, 0] / self.ppm - self.cancha.L2 - self.margen
        y = self.cancha.W2 + self.margen - uv[:, 1] / self.ppm
        return np.stack([x, y], 1)

    def base(self) -> np.ndarray:
        img = np.full((self.alto_px, self.ancho_px, 3), self.color_fondo, np.uint8)
        grosor = max(1, int(round(self.ppm * 0.15)))
        for pl in self.cancha.lineas():
            pts = np.round(self.a_px(pl)).astype(np.int32)
            cv2.polylines(img, [pts], False, self.color_linea, grosor, cv2.LINE_AA)
        for s in (-1, 1):
            xp = s * (self.cancha.L2 - self.cancha.distancia_penal)
            cv2.circle(img, tuple(np.round(self.a_px([xp, 0])[0]).astype(int)), grosor + 1,
                       self.color_linea, -1, cv2.LINE_AA)
        return img


def mascara_lineas_proyectadas(H_cancha2img: np.ndarray, forma: tuple, cancha: Cancha,
                               grosor: int = 2) -> np.ndarray:
    """Dibuja las líneas de la cancha proyectadas en la imagen (para alinear/validar)."""
    h, w = forma[:2]
    m = np.zeros((h, w), np.uint8)
    H = np.asarray(H_cancha2img, float)
    # Una homografía vale igual multiplicada por -1: se fija el signo para que un punto
    # visible (abajo al centro de la imagen) quede "delante" de la cámara (w > 0)
    q = np.linalg.inv(H) @ np.array([w / 2, h * 0.9, 1.0])
    H = H * (np.sign(q[2]) or 1.0)
    for pl in cancha.lineas(paso=0.5):
        hom = np.hstack([pl, np.ones((len(pl), 1))]) @ H.T
        # Puntos detrás de la cámara (w <= 0) no existen en la imagen: se corta la polilínea
        delante = hom[:, 2] > 1e-6
        pts = hom[:, :2] / np.where(delante, hom[:, 2], 1)[:, None]
        ok = delante & (np.abs(pts) < 2e4).all(1)
        for tramo in np.split(np.arange(len(pts)), np.where(~ok)[0]):
            tramo = [i for i in tramo if ok[i]]
            if len(tramo) >= 2:
                cv2.polylines(m, [np.round(pts[tramo]).astype(np.int32)], False, 255, grosor, cv2.LINE_AA)
    return m
