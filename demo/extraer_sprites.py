"""Recortes de jugadores reales (sprites) para la transmisión sintética.

Toma cuadros del video real de SoccerTrack, detecta personas con YOLO, se queda con los
cuerpos completos, quita el fondo con GrabCut y clasifica el color de la camiseta.

    python demo/extraer_sprites.py --videos left.mp4 right.mp4 -o demo/sprites --modelo modelos/yolo11s.pt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def kit(rgba: np.ndarray) -> str | None:
    """Color de la camiseta: negro, blanco, azul o amarillo (None si no es claro)."""
    h, w = rgba.shape[:2]
    torso = rgba[int(0.18 * h):int(0.48 * h), int(0.2 * w):int(0.8 * w)]
    m = torso[..., 3] > 0
    if m.sum() < 20:
        return None
    hsv = cv2.cvtColor(np.ascontiguousarray(torso[..., :3]), cv2.COLOR_BGR2HSV)[m]
    H, S, V = (float(np.median(hsv[:, k])) for k in range(3))
    if V < 100:
        return "negro"
    if S < 110 and V > 180:
        return "blanco"
    if 90 <= H <= 130 and S >= 130:
        return "azul"
    if 15 <= H <= 40 and S > 90:
        return "amarillo"
    return None


def recortar(frame, caja, margen=0.12) -> np.ndarray | None:
    x1, y1, x2, y2 = caja
    w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - margen * w)), max(0, int(y1 - margen * h * 0.5))
    X2, Y2 = min(frame.shape[1], int(x2 + margen * w)), min(frame.shape[0], int(y2 + margen * h * 0.3))
    rec = frame[Y1:Y2, X1:X2].copy()
    if rec.shape[0] < 40 or rec.shape[1] < 12:
        return None
    mask = np.full(rec.shape[:2], cv2.GC_PR_BGD, np.uint8)
    rx1, ry1 = int(x1 - X1), int(y1 - Y1)
    rx2, ry2 = int(x2 - X1), int(y2 - Y1)
    mask[ry1:ry2, rx1:rx2] = cv2.GC_PR_FGD
    # el centro del cuerpo es seguro, el césped que rodea es fondo seguro
    cx = (rx1 + rx2) // 2
    mask[ry1 + (ry2 - ry1) // 5: ry2 - (ry2 - ry1) // 4, max(cx - 2, 0):cx + 2] = cv2.GC_FGD
    hsv = cv2.cvtColor(rec, cv2.COLOR_BGR2HSV)
    verde = cv2.inRange(hsv, (30, 40, 40), (90, 255, 255)) > 0
    mask[verde & (mask != cv2.GC_FGD)] = cv2.GC_BGD
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(rec, mask, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return None
    alfa = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
    alfa = cv2.morphologyEx(alfa, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    # componente principal (dilatada para no perder la cabeza si el cuello queda fino)
    n, lab, st, _ = cv2.connectedComponentsWithStats(cv2.dilate(alfa, np.ones((5, 5), np.uint8)))
    if n < 2:
        return None
    k = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    alfa = np.where((lab == k) & (alfa > 0), 255, 0).astype(np.uint8)
    ys, xs = np.where(alfa > 0)
    if len(ys) < 150:
        return None
    rgba = np.dstack([rec, cv2.GaussianBlur(alfa, (3, 3), 0)])[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    hh, ww = rgba.shape[:2]
    if not (1.6 <= hh / max(ww, 1) <= 4.5):      # cuerpo completo, de pie o corriendo
        return None
    # con cabeza: las filas de arriba son más angostas que los hombros
    anchos = (rgba[..., 3] > 0).sum(1)
    if anchos[:max(2, int(0.08 * hh))].mean() > 0.6 * anchos.max():
        return None
    return rgba


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("-o", "--salida", default="demo/sprites")
    ap.add_argument("--modelo", default="yolo11s.pt")
    ap.add_argument("--cuadros", type=int, default=12, help="cuadros por video")
    ap.add_argument("--por-kit", type=int, default=40)
    a = ap.parse_args()
    from ultralytics import YOLO
    modelo = YOLO(a.modelo)
    out = Path(a.salida)
    out.mkdir(parents=True, exist_ok=True)
    indice, cuenta = [], {}
    for video in a.videos:
        cap = cv2.VideoCapture(video)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx in np.linspace(0, total - 1, a.cuadros).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok:
                continue
            corte = int(0.32 * frame.shape[0])          # cielo, árboles y tribuna lejana
            r = modelo.predict(frame[corte:], imgsz=1600, conf=0.35, classes=[0], verbose=False)[0]
            for caja in r.boxes.xyxy.cpu().numpy():
                caja = caja + np.array([0, corte, 0, corte])
                if caja[3] - caja[1] < 60:                # demasiado lejos: queda borroso
                    continue
                rgba = recortar(frame, caja)
                if rgba is None:
                    continue
                k = kit(rgba)
                if k is None or cuenta.get(k, 0) >= a.por_kit:
                    continue
                cuenta[k] = cuenta.get(k, 0) + 1
                nombre = f"{k}_{cuenta[k]:03d}.png"
                cv2.imwrite(str(out / nombre), rgba)
                indice.append({"archivo": nombre, "kit": k, "alto": int(rgba.shape[0]), "ancho": int(rgba.shape[1]),
                               "video": Path(video).name, "cuadro": int(idx)})
        cap.release()
    (out / "indice.json").write_text(json.dumps(indice, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{len(indice)} sprites en {out}: " + ", ".join(f"{k} {v}" for k, v in sorted(cuenta.items())))


if __name__ == "__main__":
    main()
