"""Calibración del clip real de SoccerTrack (cámara fija con lente gran angular).

1. Intrínsecos y distorsión del lente con las fotos del tablero de ajedrez del dataset.
   Las fotos del tablero cubren solo una esquina de la imagen, así que la distorsión se afina
   después con los puntos de la cancha (focal, k1 y k2 que mejor dejan rectas las líneas).
2. Puntos de la cancha anotados en CVAT (etiqueta "(X,Y)" en metros desde la esquina) ->
   metros con origen en el centro, sin distorsión, y homografía robusta (RANSAC: el archivo
   trae un par de etiquetas equivocadas).

    python demo/soccertrack_calibracion.py --dataset ruta/SoccerTrack --lado right -o demo/clips/real/calibracion.json
"""
from __future__ import annotations

import argparse
import glob
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

from analista.cancha import Cancha
from analista.vision.calibracion import Calibracion


def intrinsecos(carpeta: Path, patrones=((10, 7), (9, 6), (9, 5))):
    fotos = sorted(glob.glob(str(carpeta / "*.jpg")))
    mejor = None
    for patron in patrones:
        obj = np.zeros((patron[0] * patron[1], 3), np.float32)
        obj[:, :2] = np.mgrid[0:patron[0], 0:patron[1]].T.reshape(-1, 2)
        objs, imgs, forma = [], [], None
        for f in fotos:
            g = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            forma = g.shape[::-1]
            esc = 0.5
            chico = cv2.resize(g, None, fx=esc, fy=esc)
            ok, c = cv2.findChessboardCornersSB(chico, patron)
            if ok:
                c = cv2.cornerSubPix(g, (c / esc).astype(np.float32), (11, 11), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3))
                objs.append(obj)
                imgs.append(c)
        print(f"Tablero {patron}: encontrado en {len(objs)} de {len(fotos)} fotos")
        if len(objs) >= 8 and (mejor is None or len(objs) > mejor[0]):
            mejor = (len(objs), objs, imgs, forma, patron)
    if mejor:
        n, objs, imgs, forma, patron = mejor
        err, K, D, _, _ = cv2.calibrateCamera(objs, imgs, forma, None, None)
        print(f"Intrínsecos con el tablero {patron} ({n} fotos): error de reproyección {err:.2f} px")
        return K, D.ravel()
    npy = carpeta / ".camera_matrix.npy"
    print("No se detectó el tablero; uso los intrínsecos que trae el dataset")
    return np.load(npy), np.load(carpeta / ".distortion_coefficients.npy").ravel()


def puntos_cvat(xml: Path, largo=105.0, ancho=68.0):
    out = []
    for p in ET.parse(xml).getroot().iter("points"):
        m = re.match(r"\(([\d.]+),([\d.]+)\)", p.get("label", ""))
        if not m:
            continue
        X, Y = float(m.group(1)), float(m.group(2))
        u, v = (float(s) for s in p.get("points").split(";")[0].split(","))
        out.append(((u, v), (X - largo / 2, ancho / 2 - Y)))
    return out


def _errores(K, D, img, can, umbral=1.0):
    und = cv2.undistortPoints(img.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    H, inl = cv2.findHomography(und, can, cv2.RANSAC, umbral)
    inl = inl.ravel().astype(bool)
    for _ in range(3):
        H, _ = cv2.findHomography(und[inl], can[inl], 0)
        e = np.linalg.norm(cv2.perspectiveTransform(und.reshape(-1, 1, 2), H).reshape(-1, 2) - can, axis=1)
        inl = e < umbral
    return e, H, und, inl


def afinar_lente(K, D, img, can):
    """Focal y distorsión radial (k1, k2) que minimizan el error de la cancha (80% de puntos más fieles).
    El centro óptico se fija en el centro de la imagen: el tablero, siempre en la misma esquina, lo sesga."""
    from scipy.optimize import minimize
    K = K.copy()
    K[0, 2], K[1, 2] = 3072 / 2, 1728 / 2

    def coste(p):
        Kp = K.copy()
        Kp[0, 0] = Kp[1, 1] = p[0]
        e = _errores(Kp, np.array([p[1], p[2], 0, 0, 0]), img, can)[0]
        return float(np.mean(np.minimum(np.sort(e)[:int(0.8 * len(e))], 3) ** 2))
    mejor = None
    for f in (0.9 * K[0, 0], K[0, 0], 1.1 * K[0, 0]):
        for k1 in (-0.45, -0.35, -0.25):
            r = minimize(coste, [f, k1, 0.12], method="Nelder-Mead", options={"xatol": 1e-4, "fatol": 1e-6, "maxiter": 600})
            if mejor is None or r.fun < mejor.fun:
                mejor = r
    K = K.copy()
    K[0, 0] = K[1, 1] = mejor.x[0]
    return K, np.array([mejor.x[1], mejor.x[2], 0.0, 0.0, 0.0])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="carpeta clonada de github.com/keisuke198619/SoccerTrack")
    ap.add_argument("--lado", default="right", choices=["left", "right"])
    ap.add_argument("-o", "--salida", default="calibracion.json")
    a = ap.parse_args()
    raw = Path(a.dataset) / "data" / "raw"
    K, D = intrinsecos(raw / f"{a.lado}_checkerboard_images")
    pares = puntos_cvat(raw / f"{a.lado}_keypoints.xml")
    img = np.array([p[0] for p in pares], np.float64)
    can = np.array([p[1] for p in pares], np.float64)
    K, D = afinar_lente(K, D, img, can)
    print(f"Lente afinado con la cancha: focal {K[0, 0]:.0f} px · k1 {D[0]:.3f} · k2 {D[1]:.3f}")
    err, H, und, inl = _errores(K, D, img, can)
    print(f"{inl.sum()} de {len(pares)} puntos usados · error medio {err[inl].mean():.2f} m · descartados "
          f"(etiquetas dudosas del dataset): " + ", ".join(f"({can[k][0]:.1f}, {can[k][1]:.1f}) {err[k]:.1f} m" for k in np.where(~inl)[0]))
    cal = Calibracion(Cancha(105.0, 68.0), (3072, 1728), True,
                      {"K": K.tolist(), "dist": D.tolist(), "modelo": "estandar"},
                      [{"cuadro": 0, "puntos": [{"imagen": und[k].round(2).tolist(), "cancha": can[k].round(2).tolist()}
                                               for k in np.where(inl)[0]], "H_img2cancha": (H / H[2, 2]).tolist()}])
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    cal.guardar(a.salida)
    print(f"Calibración: {a.salida}")


if __name__ == "__main__":
    main()
