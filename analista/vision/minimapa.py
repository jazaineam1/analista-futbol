"""Minimapa (radar táctico) y anotaciones sobre el cuadro de video."""
from __future__ import annotations

import cv2
import numpy as np

from ..cancha import LienzoCancha

COLOR_OTRO = (0, 215, 255)   # árbitros / sin equipo (BGR amarillo)
COLOR_BALON = (255, 255, 255)


def hex_a_bgr(h: str) -> tuple:
    h = h.lstrip("#")
    return (int(h[4:6], 16), int(h[2:4], 16), int(h[0:2], 16))


def color_contraste(bgr) -> tuple:
    b, g, r = bgr
    return (20, 20, 20) if (0.299 * r + 0.587 * g + 0.114 * b) > 150 else (245, 245, 245)


class Minimapa:
    def __init__(self, lienzo: LienzoCancha, colores: dict[str, tuple]):
        self.lienzo = lienzo
        self.fondo = lienzo.base()
        self.colores = colores

    def dibujar(self, jugadores: list[dict], balon: tuple | None, huella: np.ndarray | None,
                fuera: list[dict] | None = None) -> np.ndarray:
        img = self.fondo.copy()
        if huella is not None:
            capa = img.copy()
            pts = np.round(self.lienzo.a_px(huella)).astype(np.int32)
            cv2.fillPoly(capa, [pts], (255, 255, 255))
            img = cv2.addWeighted(capa, 0.16, img, 0.84, 0)
            cv2.polylines(img, [pts], True, (255, 255, 255), 1, cv2.LINE_AA)
        r = max(4, int(self.lienzo.ppm * 0.9))
        for j in fuera or []:   # extrapolados fuera de cuadro: huecos
            c = self.colores.get(j.get("equipo"), COLOR_OTRO)
            u, v = np.round(self.lienzo.a_px([j["x"], j["y"]])[0]).astype(int)
            cv2.circle(img, (u, v), r, c, 1, cv2.LINE_AA)
        for j in jugadores:
            c = self.colores.get(j.get("equipo"), COLOR_OTRO)
            u, v = np.round(self.lienzo.a_px([j["x"], j["y"]])[0]).astype(int)
            cv2.circle(img, (u, v), r, c, -1, cv2.LINE_AA)
            cv2.circle(img, (u, v), r, (30, 30, 30), 1, cv2.LINE_AA)
        if balon is not None:
            u, v = np.round(self.lienzo.a_px([balon])[0]).astype(int)
            cv2.circle(img, (u, v), max(3, r // 2), COLOR_BALON, -1, cv2.LINE_AA)
            cv2.circle(img, (u, v), max(3, r // 2), (0, 0, 0), 1, cv2.LINE_AA)
        return img


def esquina_libre(campo: np.ndarray, ancho_rel: float = 0.32, alto_rel: float = 0.30) -> str:
    """Esquina del cuadro con menos cancha, para poner el minimapa sin tapar el juego."""
    h, w = campo.shape[:2]
    mw, mh = int(ancho_rel * w), int(alto_rel * h)
    zonas = {"ii": campo[h - mh:, :mw], "id": campo[h - mh:, w - mw:], "si": campo[:mh, :mw], "sd": campo[:mh, w - mw:]}
    return min(zonas, key=lambda k: (float((zonas[k] > 0).mean()), k != "id"))


def anotar_cuadro(frame: np.ndarray, jugadores: list[dict], balon_uv, colores: dict, estado_txt: str,
                  minimapa: np.ndarray | None, esquina: str = "id") -> np.ndarray:
    out = frame.copy()
    for j in jugadores:
        c = colores.get(j.get("equipo"), COLOR_OTRO)
        u, v, ancho = j["u"], j["v"], j.get("ancho_px", 20)
        cv2.ellipse(out, (int(u), int(v)), (max(6, int(ancho * 0.7)), max(3, int(ancho * 0.25))), 0, -45, 235,
                    c, 2, cv2.LINE_AA)
        etiqueta = str(j["id"])
        (tw, th), _ = cv2.getTextSize(etiqueta, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        x0, y0 = int(u - tw / 2 - 3), int(v + 6)
        cv2.rectangle(out, (x0, y0), (x0 + tw + 6, y0 + th + 6), c, -1)
        cv2.putText(out, etiqueta, (x0 + 3, y0 + th + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color_contraste(c), 1,
                    cv2.LINE_AA)
    if balon_uv is not None:
        u, v = int(balon_uv[0]), int(balon_uv[1])
        cv2.drawMarker(out, (u, v - 14), (0, 220, 255), cv2.MARKER_TRIANGLE_DOWN, 14, 2, cv2.LINE_AA)
    h, w = out.shape[:2]
    arriba_der = esquina == "sd" and minimapa is not None
    ty = h - 46 if arriba_der else 10
    cv2.rectangle(out, (w - 330, ty), (w - 10, ty + 26), (20, 20, 20), -1)
    cv2.putText(out, estado_txt, (w - 322, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    if minimapa is not None:
        escala = min(0.32 * w / minimapa.shape[1], 0.30 * h / minimapa.shape[0])
        mm = cv2.resize(minimapa, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
        mh, mw = mm.shape[:2]
        x0 = 12 if esquina in ("ii", "si") else w - mw - 12
        y0 = 12 if esquina in ("si", "sd") else h - mh - 12
        roi = out[y0:y0 + mh, x0:x0 + mw]
        out[y0:y0 + mh, x0:x0 + mw] = cv2.addWeighted(mm, 0.88, roi, 0.12, 0)
    return out
