"""VAR: fuera de lugar, línea de gol, balón fuera, faltas en el área y detección de pases."""
import numpy as np
import pandas as pd
import pytest

from analista.analisis import var as av

P = av.ParametrosVar(error_posicion_m=0.1, error_estimado_m=1.0, error_cuerpo_m=0.0, error_tiempo_s=0.0, error_balon_m=0.1)


def instante(atacantes, defensores, balon=(0.0, 0.0), arquero=True):
    """A ataca hacia +x. atacantes/defensores: [(id, x, y)]."""
    filas = [{"id": i, "equipo": "A", "rol": "campo", "x": x, "y": y, "vx": 0.0, "vy": 0.0, "visible": True} for i, x, y in atacantes]
    filas += [{"id": i, "equipo": "B", "rol": "campo", "x": x, "y": y, "vx": 0.0, "vy": 0.0, "visible": True} for i, x, y in defensores]
    if arquero:
        filas.append({"id": "gk", "equipo": "B", "rol": "arquero", "x": 50.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "visible": True})
    return av.Instante(0.0, pd.DataFrame(filas), balon)


def test_fuera_de_lugar_claro_habilitado_y_dudoso():
    ins = instante([("pas", 0, 0), ("r", 31, 5)], [("d1", 30, 0), ("d2", 25, 10)])
    r = av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")
    assert r["veredicto"] == "fuera de lugar" and r["margen_m"] == pytest.approx(1.0) and r["linea_x"] == 30
    ins = instante([("pas", 0, 0), ("r", 28, 5)], [("d1", 30, 0), ("d2", 25, 10)])
    assert av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")["veredicto"] == "habilitado"
    ins = instante([("pas", 0, 0), ("r", 30.05, 5)], [("d1", 30, 0), ("d2", 25, 10)])
    assert av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")["veredicto"] == "dudoso"


def test_reglas_campo_propio_balon_y_saques():
    # en su propio campo nunca está adelantado
    ins = instante([("pas", -30, 0), ("r", -5, 5)], [("d1", -10, 0), ("d2", -12, 10)])
    assert av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")["veredicto"] == "habilitado"
    # el balón más adelantado que el penúltimo defensor define la línea
    ins = instante([("pas", 35, 0), ("r", 34, 5)], [("d1", 30, 0), ("d2", 25, 10)], balon=(35.0, 0.0))
    r = av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")
    assert r["linea_por"] == "balón" and r["veredicto"] == "habilitado"
    # saque de banda: no hay fuera de lugar
    ins = instante([("pas", 0, 0), ("r", 40, 5)], [("d1", 30, 0), ("d2", 25, 10)])
    assert av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r", reanudacion="saque de banda")["veredicto"] == "no aplica"


def test_direccion_y_arquero_supuesto():
    # B ataca hacia -x: los defensores de A están del lado negativo
    filas = [{"id": "r", "equipo": "B", "rol": "campo", "x": -31.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "visible": True},
             {"id": "d", "equipo": "A", "rol": "campo", "x": -30.0, "y": 3.0, "vx": 0.0, "vy": 0.0, "visible": True}]
    ins = av.Instante(0.0, pd.DataFrame(filas), (0.0, 0.0))
    r = av.revisar_fuera_de_lugar(ins, "B", -1, P, None, "r")
    assert r["veredicto"] == "fuera de lugar" and any("arquero" in a for a in r["avisos"])


def test_receptor_fuera_de_camara_y_cobertura():
    ins = instante([("pas", 0, 0), ("r", 31, 5)], [("d1", 30, 0), ("d2", 25, 10)], arquero=False)
    ins.zona_visible = np.array([[-10, -34], [32, -34], [32, 34], [-10, 34]], float)   # no se ve detrás del receptor
    r = av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")
    assert r["veredicto"] == "dudoso" and any("defensor fuera de imagen" in a for a in r["avisos"])
    # un jugador fuera de cámara (posición estimada) tiene más incertidumbre
    ins = instante([("pas", 0, 0), ("r", 31, 5)], [("d1", 30, 0), ("d2", 25, 10)])
    ins.jugadores.loc[ins.jugadores["id"] == "r", "visible"] = False
    assert av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")["veredicto"] == "dudoso"


def test_sin_intervencion_si_el_pase_no_llega():
    ins = instante([("pas", 0, 0), ("r", 33, 5)], [("d1", 30, 0), ("d2", 25, 10)])
    r = av.revisar_fuera_de_lugar(ins, "A", 1, P, "pas", "r")
    av.marcar_intervencion(r, "unsuccessful")
    assert r["veredicto"] == "sin intervención" and r["posicion"] == "fuera de lugar"


def _tracking(n=30, fps=10):
    """Pase de A1 (en x=0) a A2 que corre; el balón sale en t=1,0 s a 15 m/s."""
    filas, bal = [], []
    for k in range(n):
        t = k / fps
        filas += [{"cuadro": k, "t": t, "id": 1, "equipo": "A", "rol": "campo", "x": 0.0, "y": 0.0},
                  {"cuadro": k, "t": t, "id": 2, "equipo": "A", "rol": "campo", "x": 20.0 + 2 * t, "y": 0.0},
                  {"cuadro": k, "t": t, "id": 3, "equipo": "B", "rol": "campo", "x": 23.0, "y": 5.0},
                  {"cuadro": k, "t": t, "id": 4, "equipo": "B", "rol": "campo", "x": 21.0, "y": -5.0},
                  {"cuadro": k, "t": t, "id": 5, "equipo": "B", "rol": "arquero", "x": 50.0, "y": 0.0}]
        bx = 0.3 if t <= 1.0 else 0.3 + 15 * (t - 1.0)
        bal.append({"cuadro": k, "t": t, "x": min(bx, 26.0), "y": 0.0})
    return pd.DataFrame(filas), pd.DataFrame(bal)


def test_posiciones_interpolan_y_momento_del_pase():
    tr, ba = _tracking()
    pos = av.Posiciones(tr, ba)
    ins = pos.instante(1.05)
    a2 = ins.jugadores[ins.jugadores["id"] == 2].iloc[0]
    assert a2["x"] == pytest.approx(22.1, abs=1e-6) and a2["vx"] == pytest.approx(2.0)
    t = av.momento_del_pase(pos, 1.0, 1)
    assert abs(t - 1.0) < 0.12
    # en ese instante A2 (x≈22) está detrás del penúltimo defensor (x=23): habilitado
    r = av.revisar_fuera_de_lugar(pos.instante(t), "A", 1, P, 1, 2)
    assert r["veredicto"] == "habilitado"
    # un segundo después (cuando recibe) ya estaría adelantado: por eso importa el instante
    r2 = av.revisar_fuera_de_lugar(pos.instante(t + 1.0), "A", 1, P, 1, 2)
    assert r2["margen_m"] > 0


def test_linea_de_gol_y_balon_fuera():
    t = np.arange(0, 3, 0.1)
    bal = pd.DataFrame({"cuadro": range(len(t)), "t": t, "x": 45 + 4 * t, "y": 1.0})       # entra al arco
    pos = av.Posiciones(pd.DataFrame({"cuadro": [0], "t": [0.0], "id": [1], "equipo": ["A"], "rol": ["campo"],
                                      "x": [0.0], "y": [0.0]}), bal)
    g = av.revisar_gol(pos, 2.0, 1, P)
    assert g["veredicto"] == "gol" and g["margen_m"] > 0.5
    bal2 = bal.assign(x=np.minimum(45 + 4 * t, 52.4))                                         # se queda sobre la línea
    g2 = av.revisar_gol(av.Posiciones(pos.df.reset_index().rename(columns={"index": "cuadro"}).assign(t=0.0), bal2), 2.0, 1, P)
    assert g2["veredicto"] in ("no entró", "dudoso")
    # balón que sale 0,6 m por la banda y vuelve sin que se pare el juego
    tt = np.arange(0, 4, 0.1)
    y = 33 + 1.6 * np.exp(-(((tt - 2) / 0.3) ** 2))            # llega a 34,6: 0,5 m afuera, unos 0,3 s
    bal3 = pd.DataFrame({"cuadro": range(len(tt)), "t": tt, "x": 10 + 4 * tt, "y": y})
    sal = av.salidas_del_balon(av.Posiciones(pos.df.reset_index().rename(columns={"index": "cuadro"}).assign(t=0.0), bal3), P)
    assert len(sal) == 1 and sal[0]["linea"] == "de banda" and sal[0]["veredicto"] == "salió"


def test_falta_dentro_y_fuera_del_area():
    dentro, d = av.en_area(40.0, 5.0, 1, av.ParametrosVar())
    assert dentro and d == pytest.approx(4.0)
    assert av.revisar_falta(40.0, 5.0, 1, av.ParametrosVar(), 0.3)["veredicto"] == "dentro del área"
    assert av.revisar_falta(34.0, 0.0, 1, av.ParametrosVar(), 0.3)["veredicto"] == "fuera del área"
    assert av.revisar_falta(36.05, 0.0, 1, av.ParametrosVar(), 0.3)["veredicto"] == "dudoso"
    f = av.revisar_falta(34.0, 0.0, 1, av.ParametrosVar(), 0.3, sancion="penal")
    assert "revisarlo" in f["explicacion"]


def test_revision_completa_y_repeticion():
    tr, ba = _tracking(40)
    p = av.ParametrosVar(error_posicion_m=0.1, error_cuerpo_m=0.0, error_tiempo_s=0.0)
    pases = pd.DataFrame([{"t_toque": 1.0, "equipo": "A", "pasador": 1, "receptor": 2}])
    rev, pos = av.revisar_tracking(tr, ba, p, {"A": 1, "B": -1}, pases=pases)
    assert rev.resumen()["pases_revisados"] == 1
    datos = av.para_repeticion(rev.fueras_de_lugar[0], pos, {2: "Delantero"})
    assert datos["cuadros"] and datos["revision"]["receptor"] == "2" and datos["nombres"]["2"] == "Delantero"
    manual = av.revisar_instante(pos, 2.0, "A", 1, p)
    assert manual["manual"] and manual["receptor"] == 2


def test_rebote_en_fuera_de_lugar():
    # A remata desde x=20; "r" está en 40 (adelantado: penúltimo defensor en 35) y juega el rebote tras la atajada
    ins = instante([("tir", 20, 0), ("r", 40, 3)], [("d1", 35, 0), ("d2", 30, 10)], balon=(20.0, 0.0))
    sig = [{"id": "gk", "equipo": "B", "rol": "arquero", "duracion": 0.2}, {"id": "r", "equipo": "A", "rol": "campo", "duracion": 1.0}]
    r = av.revisar_tiro(None, ins, "A", 1, P, "tir", sig)
    assert r["motivo"] == "rebote" and r["veredicto"] == "fuera de lugar" and r["margen_m"] == pytest.approx(5.0)
    # si un defensor la juega a propósito (control largo) se reinicia: ya no es fuera de lugar
    sig = [{"id": "d1", "equipo": "B", "rol": "campo", "duracion": 1.0}, {"id": "r", "equipo": "A", "rol": "campo", "duracion": 1.0}]
    assert av.revisar_tiro(None, ins, "A", 1, P, "tir", sig)["motivo"] != "rebote"
    # nadie adelantado: nada que revisar
    ins = instante([("tir", 20, 0), ("r", 30, 3)], [("d1", 35, 0), ("d2", 33, 10)], balon=(20.0, 0.0))
    assert av.revisar_tiro(None, ins, "A", 1, P, "tir", []) is None


def test_interferencia_con_el_arquero():
    # "r" adelantado, parado entre el balón y el arquero (en x=50): le tapa la visión
    ins = instante([("tir", 25, 0), ("r", 45, 0.3)], [("d1", 38, 5), ("d2", 30, 10)], balon=(25.0, 0.0))
    r = av.revisar_tiro(None, ins, "A", 1, P, "tir", [])
    assert r["motivo"] == "interferencia" and r["veredicto"] == "revisar" and r["vision"][1] == [50.0, 0.0]
    # adelantado pero lejos de la trayectoria y del arquero: sin intervención
    ins = instante([("tir", 25, 0), ("r", 45, 25)], [("d1", 38, 5), ("d2", 30, 10)], balon=(25.0, 0.0))
    assert av.revisar_tiro(None, ins, "A", 1, P, "tir", [])["veredicto"] == "sin intervención"


def test_ocasion_manifiesta_de_gol():
    # solo frente al arco a 18 m, con el balón, nadie entre él y el arco salvo el arquero
    filas = [{"id": "v", "equipo": "A", "rol": "campo", "x": 34.5, "y": 0.0, "vx": 6.0, "vy": 0.0, "visible": True},
             {"id": "inf", "equipo": "B", "rol": "campo", "x": 33.5, "y": 0.8, "vx": 6.0, "vy": 0.0, "visible": True},
             {"id": "d", "equipo": "B", "rol": "campo", "x": 20.0, "y": 10.0, "vx": 0.0, "vy": 0.0, "visible": True},
             {"id": "gk", "equipo": "B", "rol": "arquero", "x": 51.0, "y": 0.0, "vx": 0.0, "vy": 0.0, "visible": True}]
    ins = av.Instante(0.0, pd.DataFrame(filas), (35.0, 0.0))
    r = av.revisar_ocasion(ins, "A", 1, 34.5, 0.0, P, "v", "inf")
    assert r["veredicto"] == "ocasión manifiesta" and r["criterios"]["defensores_entre"] == 0
    assert "roja" in r["explicacion"]
    # con un defensor de campo cerrando el camino al arco ya no es manifiesta
    filas.append({"id": "c", "equipo": "B", "rol": "campo", "x": 44.0, "y": 1.0, "vx": 0.0, "vy": 0.0, "visible": True})
    r = av.revisar_ocasion(av.Instante(0.0, pd.DataFrame(filas), (35.0, 0.0)), "A", 1, 34.5, 0.0, P, "v", "inf")
    assert r["veredicto"] != "ocasión manifiesta"
    # lejos del arco
    r = av.revisar_ocasion(ins, "A", 1, -5.0, 0.0, P, "v", "inf")
    assert r["veredicto"] == "no es ocasión manifiesta"


def test_punto_mas_adelantado_del_cuerpo():
    """Una silueta sobre césped: el extremo hacia el ataque cae en el pie adelantado, no en el centro."""
    from analista.vision.var_imagen import extremo_cuerpo
    img = np.zeros((200, 300, 3), np.uint8)
    img[:] = (40, 150, 40)                                      # césped (BGR)
    img[60:140, 145:155] = (200, 60, 30)                       # torso y piernas
    img[125:140, 155:175] = (200, 60, 30)                      # pie adelantado hacia la derecha
    img[75:85, 110:145] = (200, 60, 30)                        # brazo extendido hacia la izquierda (no cuenta)
    H = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1.0]])     # 10 px = 1 m
    caja = (150.0, 140.0, 80.0, 50.0)                           # u, v (pie), alto, ancho (como la caja del detector)
    derecha = extremo_cuerpo(img, caja, H, 1)
    izquierda = extremo_cuerpo(img, caja, H, -1)
    assert derecha == pytest.approx(17.4, abs=0.2)             # el pie llega hasta x = 174 px
    assert izquierda > 13.5                                    # el brazo (hasta x = 110 px) no cuenta


def test_clip_var_como_en_la_tv(tmp_path):
    """Un clip con cámara lenta, imagen congelada y las líneas animadas sobre el video."""
    import cv2

    from analista.vision.pipeline import _escritor_h264
    from analista.vision.var_imagen import clip_var
    video = tmp_path / "partido.mp4"
    esc = _escritor_h264(video, (180, 320, 3), 25)
    for k in range(75):                                  # 3 s de "césped" con un jugador que avanza
        f = np.full((180, 320, 3), (40, 150, 40), np.uint8)
        cv2.rectangle(f, (100 + k, 80), (108 + k, 120), (200, 60, 30), -1)
        esc.stdin.write(f.tobytes())
    esc.stdin.close()
    esc.wait()
    H = np.array([[0.2, 0, -32], [0, -0.2, 18], [0, 0, 1.0]])    # 320x180 px -> 64x36 m, origen al centro
    rev = {"tipo": "fuera de lugar", "t": 1.5, "veredicto": "fuera de lugar", "linea_x": 3.0, "receptor_xy": [4.0, -2.0],
           "defensor_xy": [3.0, 1.0], "margen_m": 1.0, "sigma_m": 0.3, "direccion": 1}
    ruta = clip_var(video, lambda t: H, rev, tmp_path / "var.mp4", 64.0, 36.0, antes_s=1.0, despues_s=0.5, congelado_s=2.0)
    assert ruta is not None and ruta.exists()
    cap = cv2.VideoCapture(str(ruta))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, n - 30)             # ya congelado y con las líneas
    ok, f = cap.read()
    cap.release()
    # 25 cuadros en cámara lenta (x2) + 50 congelados + 13 que siguen
    assert ok and n >= 2 * 25 + 50 + 10
    azul = (np.abs(f.astype(int) - np.array([208, 151, 21])).sum(axis=2) < 90).sum()
    assert azul > 30                                      # la línea del defensor está dibujada
