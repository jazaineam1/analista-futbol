"""Fuentes de video: descripción de la pantalla, grabación en tiempo real y endpoints del panel."""
import io
import os
import time

import cv2
import numpy as np
import pytest

from analista.vision import fuentes as fu


def test_interpretar_pantalla_y_tipo():
    assert fu.interpretar_pantalla("pantalla") == (1, None)
    assert fu.interpretar_pantalla("pantalla:2") == (2, None)
    assert fu.interpretar_pantalla("pantalla:10,20,640,360") == (1, (10, 20, 640, 360))
    assert fu.interpretar_pantalla("pantalla:2:10,20,640,360") == (2, (10, 20, 640, 360))
    with pytest.raises(ValueError):
        fu.interpretar_pantalla("pantalla:1:10,20")
    assert fu.es_archivo("C:/videos/partido.mp4") and not fu.es_archivo("disp:0") and not fu.es_archivo("rtsp://x")
    assert not fu.es_archivo("pantalla:1")


def test_grabadora_en_tiempo_real(tmp_path):
    """Aunque lleguen pocos cuadros, el archivo dura lo mismo que la captura."""
    g = fu.Grabadora(tmp_path / "vivo.mp4", fps=10)
    t0 = time.time()
    for k in range(5):                       # 5 cuadros a lo largo de 2 s
        g.escribir(np.full((120, 160, 3), 40 * k, np.uint8), t0 + 0.5 * k)
    g.cerrar()
    cap = cv2.VideoCapture(str(tmp_path / "vivo.mp4"))
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    assert abs(n / fps - 2.1) < 0.25


@pytest.fixture
def cliente(tmp_path):
    from analista.panel.servidor import Estado, crear_app
    app = crear_app(Estado(tmp_path))
    app.testing = True
    return app.test_client()


def _mp4(ruta, n=12):
    from analista.vision.pipeline import _escritor_h264
    p = _escritor_h264(ruta, (90, 160, 3), 10)
    for k in range(n):
        p.stdin.write(np.full((90, 160, 3), 20 * k % 255, np.uint8).tobytes())
    p.stdin.close()
    p.wait()


def test_subir_y_listar_videos(cliente, tmp_path):
    origen = tmp_path / "clip.mp4"
    _mp4(origen)
    r = cliente.post("/api/subir", data={"video": (io.BytesIO(origen.read_bytes()), "Mi partido.mp4")},
                     content_type="multipart/form-data")
    assert r.status_code == 200 and r.json["nombre"] == "Mi partido.mp4"
    r2 = cliente.post("/api/subir", data={"video": (io.BytesIO(origen.read_bytes()), "Mi partido.mp4")},
                      content_type="multipart/form-data")
    assert r2.json["nombre"] == "Mi partido (2).mp4"          # no se pisa el anterior
    malo = cliente.post("/api/subir", data={"video": (io.BytesIO(b"hola"), "notas.txt")}, content_type="multipart/form-data")
    assert malo.status_code == 400
    roto = cliente.post("/api/subir", data={"video": (io.BytesIO(b"no es video"), "x.mp4")}, content_type="multipart/form-data")
    assert roto.status_code == 400
    vids = cliente.get("/api/videos").json["videos"]
    assert len(vids) == 2 and vids[0]["duracion"] == pytest.approx(1.2, abs=0.2)
    assert cliente.post("/api/fuente", json={"spec": r.json["ruta"]}).json["tipo"] == "archivo"


def test_pantallas_y_dispositivos_sin_equipo(cliente):
    """Sin pantalla ni capturadora el panel responde con un mensaje, no con un error 500."""
    d = cliente.get("/api/pantallas").json
    assert "pantallas" in d
    if not os.environ.get("DISPLAY") and not os.name == "nt":
        assert d["pantallas"] == [] and d["error"]
    assert "dispositivos" in cliente.get("/api/dispositivos").json
    assert cliente.get("/api/var").json["items"] == []
    assert cliente.post("/api/var/instante", json={"t": 1, "equipo": "A"}).status_code == 400


@pytest.mark.skipif(not os.environ.get("DISPLAY"), reason="necesita una pantalla (Xvfb)")
def test_captura_de_pantalla_real():
    pantallas = fu.listar_pantallas()
    assert pantallas
    img = fu.captura_pantalla(1, (0, 0, 64, 48))
    assert img.shape == (48, 64, 3)
    f = fu.FuentePantalla(1, (0, 0, 64, 48), fps=10)
    cuadros = []
    for i, t, c in f:
        cuadros.append(t)
        if len(cuadros) >= 3:
            break
    f.cerrar()
    assert len(cuadros) == 3 and cuadros[-1] > cuadros[0]
