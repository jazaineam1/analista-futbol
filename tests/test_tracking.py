import numpy as np

from analista.analisis import tracking as at


def test_metricas_de_tracking(tracking_sintetico):
    tr, bal = tracking_sintetico
    p = at.ParametrosMetricas()
    calc = at.calcular_todo(tr, bal, None, p)
    assert calc["direccion"] == {"A": 1, "B": -1}
    r = calc["resumen"]
    assert r["posesion_pct"]["A"] > 90
    assert 0 < r["forma"]["A"]["ancho"] <= 68
    assert len(calc["pases"]) >= 3


def test_fisico_y_velocidad(tracking_sintetico):
    tr, _ = tracking_sintetico
    f = at.fisico(tr, at.ParametrosMetricas())
    a1 = f[f["id"] == "A1"].iloc[0]
    assert 8 < a1["distancia_m"] < 12               # 0,5 m/s durante 20 s
    assert a1["vel_max"] < 1.5


def test_control_de_cancha(tracking_sintetico):
    tr, bal = tracking_sintetico
    j = tr[tr["cuadro"] == 0]
    xs, ys, P = at.control_de_cancha(j, (bal["x"].iloc[0], bal["y"].iloc[0]), 105, 68)
    assert P.shape == (len(ys), len(xs))
    assert np.nanmin(P) >= 0 and np.nanmax(P) <= 1
    # cerca del arco propio de A domina A, cerca del de B domina B
    assert P[len(ys) // 2, 1] > 0.5 and P[len(ys) // 2, -2] < 0.5


def test_formacion(tracking_sintetico):
    tr, bal = tracking_sintetico
    calc = at.calcular_todo(tr, bal, None, at.ParametrosMetricas())
    f = at.formacion(tr, "B", calc["direccion"], calc["poseedor"])
    assert f.get("posiciones") and len(f["posiciones"]) == 10
