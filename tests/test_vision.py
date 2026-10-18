"""Visión de punta a punta con un video sintético y el detector clásico (sin red neuronal)."""
import json

import cv2
import numpy as np
import pandas as pd

from analista.cancha import Cancha, aplicar_h, h_desde_puntos
from analista.vision.calibracion import Calibracion
from analista.vision.deteccion import mascara_cesped
from analista.vision.fuentes import FuenteArchivo
from analista.vision.pipeline import ConfigPipeline, Pipeline

ESQ_IMG = [[60, 690], [1220, 690], [1010, 150], [270, 150]]


def _video(ruta, n=40, w=1280, h=720):
    c = Cancha()
    H_c2i = np.linalg.inv(h_desde_puntos(ESQ_IMG, [[-c.L2, -c.W2], [c.L2, -c.W2], [c.L2, c.W2], [-c.L2, c.W2]]))
    escr = cv2.VideoWriter(str(ruta), cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))
    jugadores = [(-20 + 10 * i, -10 + 5 * (i % 3), (220, 80, 30) if i < 4 else (240, 240, 240)) for i in range(8)]
    for k in range(n):
        img = np.full((h, w, 3), (40, 40, 40), np.uint8)
        poly = aplicar_h(H_c2i, [[-c.L2 - 3, -c.W2 - 3], [c.L2 + 3, -c.W2 - 3], [c.L2 + 3, c.W2 + 3], [-c.L2 - 3, c.W2 + 3]])
        cv2.fillPoly(img, [poly.astype(np.int32)], (60, 150, 60))
        for pl in c.lineas(paso=0.5):
            cv2.polylines(img, [aplicar_h(H_c2i, pl).astype(np.int32)], False, (235, 235, 235), 2)
        for x, y, col in jugadores:
            u, v = aplicar_h(H_c2i, [[x + 0.1 * k, y]])[0]
            cv2.rectangle(img, (int(u - 5), int(v - 26)), (int(u + 5), int(v)), col, -1)
            cv2.rectangle(img, (int(u - 4), int(v - 33)), (int(u + 4), int(v - 26)), (150, 170, 200), -1)
        escr.write(img)
    escr.release()


def test_mascara_cesped():
    img = np.full((200, 300, 3), (50, 160, 60), np.uint8)
    img[:60] = (200, 200, 200)
    cesped, campo = mascara_cesped(img)
    assert cesped[100:].mean() > 200 and cesped[:50].mean() < 30


def test_calibracion_guardar_y_cargar(tmp_path):
    c = Cancha()
    pares = [{"imagen": i, "cancha": k} for i, k in zip(ESQ_IMG, [[-c.L2, -c.W2], [c.L2, -c.W2], [c.L2, c.W2], [-c.L2, c.W2]])]
    cal = Calibracion(c, (1280, 720), True, None, [{"cuadro": 0, "puntos": pares}])
    cal.guardar(tmp_path / "cal.json")
    cal2 = Calibracion.cargar(tmp_path / "cal.json")
    H = cal2.homografias(640, 360)[0]
    assert np.allclose(aplicar_h(H, [[30, 345]]), [[-c.L2, -c.W2]], atol=0.5)


def test_pipeline_detector_clasico(tmp_path):
    ruta = tmp_path / "sint.mp4"
    _video(ruta)
    c = Cancha()
    pares = [{"imagen": i, "cancha": k} for i, k in zip(ESQ_IMG, [[-c.L2, -c.W2], [c.L2, -c.W2], [c.L2, c.W2], [-c.L2, c.W2]])]
    cal = Calibracion(c, (1280, 720), True, None, [{"cuadro": 0, "puntos": pares}])
    for k in cal.claves:
        k["H_img2cancha"] = h_desde_puntos([p["imagen"] for p in k["puntos"]], [p["cancha"] for p in k["puntos"]]).tolist()
    pipe = Pipeline(cal, ConfigPipeline(detector="cesped", ancho_proceso=1280, minimapa=True))
    fuente = FuenteArchivo(str(ruta))
    resumen = pipe.correr(fuente, tmp_path / "res", video_anotado=False, progreso=False)
    fuente.cerrar()
    tr = pd.read_csv(tmp_path / "res" / "tracking.csv")
    assert resumen["porcentaje_calibrado"] == 100
    assert tr["id"].nunique() >= 6
    # las posiciones caen cerca de las reales (pie del rectángulo)
    primero = tr[tr["cuadro"] == tr["cuadro"].min()]
    reales = np.array([[-20 + 10 * i, -10 + 5 * (i % 3)] for i in range(8)])
    d = np.min(np.linalg.norm(primero[["x", "y"]].to_numpy()[:, None] - reales[None], axis=2), axis=1)
    assert np.median(d) < 2.0
    assert json.loads((tmp_path / "res" / "resumen_proceso.json").read_text())["camara_fija"] is True


def test_cli_calibrar_y_verificar(tmp_path):
    from analista.cli import main
    ruta = tmp_path / "sint.mp4"
    _video(ruta, n=12)
    c = Cancha()
    esq = [[-c.L2, -c.W2], [c.L2, -c.W2], [c.L2, c.W2], [-c.L2, c.W2]]
    puntos = ";".join(f"{u},{v}={x},{y}" for (u, v), (x, y) in zip(ESQ_IMG, esq))
    assert main(["calibrar", str(ruta), "--puntos", puntos, "--camara-fija", "-o", str(tmp_path / "cal.json")]) == 0
    cal = json.loads((tmp_path / "cal.json").read_text())
    assert cal["camara_fija"] and len(cal["claves"][0]["puntos"]) == 4
    for t in ("0", "0.5"):
        assert main(["verificar", str(ruta), "-c", str(tmp_path / "cal.json"), "-t", t, "-o", str(tmp_path / "v.jpg")]) == 0
        assert (tmp_path / "v.jpg").stat().st_size > 1000
    assert main(["analizar", str(ruta), "-c", str(tmp_path / "no.json"), "-o", str(tmp_path / "x")]) == 2
