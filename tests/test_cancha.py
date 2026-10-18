import numpy as np

from analista.cancha import Cancha, aplicar_h, h_desde_puntos, mascara_lineas_proyectadas


def test_puntos_y_lineas(cancha):
    p = cancha.puntos_referencia()
    assert p["centro"] == (0.0, 0.0) if "centro" in p else True
    assert len(p) >= 20
    for x, y in p.values():
        assert abs(x) <= cancha.L2 + 1e-6 and abs(y) <= cancha.W2 + 1e-6
    assert len(cancha.lineas()) > 5


def test_homografia_ida_y_vuelta():
    H0 = np.array([[0.09, 0.01, -58.0], [0.002, -0.2, 70.0], [0.0, 0.0006, 1.0]])
    img = [[100, 600], [1180, 610], [900, 200], [300, 205], [640, 400], [500, 650]]
    can = aplicar_h(H0, img)
    ruido = np.random.default_rng(0).normal(0, 0.05, can.shape)
    H = h_desde_puntos(img, can + ruido)
    err = np.linalg.norm(aplicar_h(H, img) - can, axis=1)
    assert err.max() < 0.3
    vuelta = aplicar_h(np.linalg.inv(H), aplicar_h(H, img))
    assert np.allclose(vuelta, img, atol=1e-6)


def test_dentro():
    c = Cancha(100, 64)
    xy = np.array([[0, 0], [50, 32], [53, 0], [60, 0]])
    assert c.dentro(xy, margen=3.0).tolist() == [True, True, True, False]


def test_mascara_proyectada_dibuja_lineas():
    c = Cancha()
    H_c2i = np.linalg.inv(h_desde_puntos([[0, 700], [1280, 700], [1100, 150], [180, 150]],
                                          [[-c.L2, -c.W2], [c.L2, -c.W2], [c.L2, c.W2], [-c.L2, c.W2]]))
    m = mascara_lineas_proyectadas(H_c2i, (720, 1280, 3), c, grosor=2)
    assert m.shape == (720, 1280) and m.any()
