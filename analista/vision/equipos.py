"""Separación de equipos por color de camiseta (sin etiquetas, k-means).

Se toma la zona del torso de cada jugador, se descartan los píxeles de césped y
se resume el color en espacio Lab. Con las primeras muestras se ajustan los grupos;
lo que queda lejos de ambos equipos (árbitros, arqueros) se marca como "otro".
"""
from __future__ import annotations

import cv2
import numpy as np
from sklearn.cluster import KMeans

PESOS_LAB = np.array([0.6, 1.0, 1.0])   # menos peso a la luminancia: la luz cambia mucho la L


def _hex_a_lab(hex_color: str) -> np.ndarray:
    h = hex_color.lstrip("#")
    rgb = np.array([[[int(h[i:i + 2], 16) for i in (0, 2, 4)]]], np.uint8)
    return cv2.cvtColor(rgb[..., ::-1].copy(), cv2.COLOR_BGR2LAB).reshape(3).astype(float)


def _lab_a_hex(lab) -> str:
    lab = np.clip(np.asarray(lab, float), 0, 255).astype(np.uint8).reshape(1, 1, 3)
    b, g, r = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR).reshape(3)
    return f"#{r:02x}{g:02x}{b:02x}"


class ClasificadorEquipos:
    def __init__(self, muestras_ajuste: int = 400, colores: dict[str, str] | None = None,
                 factor_otro: float = 2.6):
        self.muestras_ajuste = muestras_ajuste
        self.colores_dados = colores or {}
        self.factor_otro = factor_otro
        self.buffer: list[np.ndarray] = []
        self.centros: np.ndarray | None = None   # fila 0 = A, fila 1 = B (Lab)
        self.umbral = None

    @staticmethod
    def caracteristica(frame: np.ndarray, caja, cesped: np.ndarray | None) -> np.ndarray | None:
        x1, y1, x2, y2 = [int(round(v)) for v in caja]
        h, w = frame.shape[:2]
        alto, ancho = y2 - y1, x2 - x1
        if alto < 8 or ancho < 3:
            return None
        # Torso: 15%-50% del alto, 20%-80% del ancho
        ty1, ty2 = y1 + int(0.15 * alto), y1 + int(0.5 * alto)
        tx1, tx2 = x1 + int(0.2 * ancho), x2 - int(0.2 * ancho)
        ty1, ty2 = max(0, ty1), min(h, max(ty2, ty1 + 2))
        tx1, tx2 = max(0, tx1), min(w, max(tx2, tx1 + 2))
        recorte = frame[ty1:ty2, tx1:tx2]
        if recorte.size == 0:
            return None
        lab = cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(float)
        if cesped is not None:
            m = cesped[ty1:ty2, tx1:tx2].reshape(-1) == 0
            if m.sum() >= 6:
                lab = lab[m]
        if len(lab) < 6:
            return None
        return np.median(lab, axis=0)

    @property
    def listo(self) -> bool:
        return self.centros is not None

    def agregar(self, feats: list):
        if self.listo:
            return
        self.buffer += [f for f in feats if f is not None]
        if len(self.buffer) >= self.muestras_ajuste:
            self._ajustar()

    def forzar_ajuste(self):
        if not self.listo and len(self.buffer) >= 20:
            self._ajustar()

    def _ajustar(self):
        X = np.array(self.buffer)
        km = KMeans(n_clusters=3, n_init=10, random_state=0).fit(X * PESOS_LAB)
        tamanos = np.bincount(km.labels_, minlength=3)
        # Los dos grupos más grandes son los equipos; el tercero absorbe árbitros/arqueros
        orden = np.argsort(-tamanos)[:2]
        centros = km.cluster_centers_[orden] / PESOS_LAB
        if self.colores_dados:
            nombres = list(self.colores_dados)
            ref = _hex_a_lab(self.colores_dados[nombres[0]])
            if np.linalg.norm(centros[1] - ref) < np.linalg.norm(centros[0] - ref):
                centros = centros[::-1]
        self.centros = centros
        dmin = self._distancias(X).min(1)
        self.umbral = max(18.0, np.percentile(dmin, 80) * self.factor_otro)

    def _distancias(self, X):
        return np.linalg.norm((np.asarray(X)[:, None, :] - self.centros[None]) * PESOS_LAB, axis=2)

    def predecir(self, feats: list) -> list:
        if not self.listo:
            return [None] * len(feats)
        out = []
        for f in feats:
            if f is None:
                out.append(None)
                continue
            d = self._distancias(f[None])[0]
            k = int(np.argmin(d))
            out.append("otro" if d[k] > self.umbral else ("A", "B")[k])
        return out

    def colores_hex(self) -> dict[str, str]:
        if not self.listo:
            return {}
        return {"A": _lab_a_hex(self.centros[0]), "B": _lab_a_hex(self.centros[1])}
