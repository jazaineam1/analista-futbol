"""Archivo de calibración: puntos marcados, homografías clave y corrección de lente.

Formato JSON:
{
  "cancha": {"largo": 105, "ancho": 68},
  "resolucion": [1280, 720],              # imagen sobre la que se marcaron los puntos
  "camara_fija": false,                   # true para cámara táctica/trípode/dron quieto
  "lente": {"K": [[...]], "dist": [...]}, # opcional (gran angular / ojo de pez)
  "claves": [{"cuadro": 0, "puntos": [{"nombre": "...", "cancha": [x, y], "imagen": [u, v]}],
              "H_img2cancha": [[...], [...], [...]]}]
}
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from ..cancha import Cancha, h_desde_puntos


@dataclass
class Calibracion:
    cancha: Cancha = field(default_factory=Cancha)
    resolucion: tuple[int, int] | None = None
    camara_fija: bool = False
    lente: dict | None = None
    claves: list[dict] = field(default_factory=list)
    carpeta: Path | None = None      # donde está el .json (para las imágenes de referencia)

    # ------------------------------------------------------------ E/S
    @staticmethod
    def cargar(ruta: str | Path) -> "Calibracion":
        d = json.loads(Path(ruta).read_text(encoding="utf-8"))
        c = d.get("cancha", {})
        cal = Calibracion(Cancha(largo=c.get("largo", 105.0), ancho=c.get("ancho", 68.0)),
                          tuple(d["resolucion"]) if d.get("resolucion") else None,
                          bool(d.get("camara_fija", False)), d.get("lente"), d.get("claves", []),
                          Path(ruta).resolve().parent)
        for k in cal.claves:  # recalcular H si solo hay puntos
            if not k.get("H_img2cancha") and len(k.get("puntos", [])) >= 4:
                k["H_img2cancha"] = h_desde_puntos([p["imagen"] for p in k["puntos"]],
                                                   [p["cancha"] for p in k["puntos"]]).tolist()
        return cal

    def guardar(self, ruta: str | Path):
        Path(ruta).write_text(json.dumps({
            "cancha": {"largo": self.cancha.largo, "ancho": self.cancha.ancho},
            "resolucion": list(self.resolucion) if self.resolucion else None,
            "camara_fija": self.camara_fija, "lente": self.lente, "claves": self.claves,
        }, indent=2, ensure_ascii=False), encoding="utf-8")

    # ------------------------------------------------------------ uso
    def homografias(self, ancho: int, alto: int) -> dict[int, np.ndarray]:
        """Homografías clave ajustadas a la resolución de procesamiento."""
        sx = sy = 1.0
        if self.resolucion:
            sx, sy = self.resolucion[0] / ancho, self.resolucion[1] / alto
        S = np.diag([sx, sy, 1.0])  # proc -> original
        out = {}
        for k in self.claves:
            if k.get("H_img2cancha") is None:
                continue
            H = np.asarray(k["H_img2cancha"], float) @ S
            out[int(k.get("cuadro") or 0)] = H / H[2, 2]
        return out

    def referencias(self, ancho: int, alto: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Imágenes de referencia guardadas con la calibración (en vivo: el cuadro congelado que
        se calibró), reescaladas a (ancho, alto), con su homografía: [(imagen BGR, H)]."""
        out = []
        Hs = self.homografias(ancho, alto)
        for k in self.claves:
            nombre = k.get("imagen_ref")
            if not nombre or self.carpeta is None:
                continue
            img = cv2.imread(str(self.carpeta / nombre))
            H = Hs.get(int(k.get("cuadro") if k.get("cuadro") is not None else 0))
            if img is None or H is None:
                continue
            out.append((cv2.resize(img, (ancho, alto), interpolation=cv2.INTER_AREA), H))
        return out

    def corrector_lente(self, ancho: int, alto: int):
        """Función que corrige la distorsión del lente a la resolución dada (o None)."""
        if not self.lente:
            return None
        K = np.asarray(self.lente["K"], float).copy()
        D = np.asarray(self.lente["dist"], float).ravel()
        if self.resolucion:
            s = ancho / self.resolucion[0]
            K[:2] *= s
        if self.lente.get("modelo") == "fisheye":
            m1, m2 = cv2.fisheye.initUndistortRectifyMap(K, D[:4], np.eye(3), K, (ancho, alto), cv2.CV_16SC2)
        else:
            m1, m2 = cv2.initUndistortRectifyMap(K, D, None, K, (ancho, alto), cv2.CV_16SC2)
        return lambda img: cv2.remap(img, m1, m2, cv2.INTER_LINEAR)
