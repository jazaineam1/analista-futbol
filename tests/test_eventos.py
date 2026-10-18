import numpy as np

from analista.analisis import eventos as ae
from analista.analisis import xg


def test_conversion_statsbomb(eventos, cancha):
    assert {"pase", "tiro", "presion", "cambio"} <= set(eventos["tipo"])
    assert eventos["equipo"].isin(["A", "B"]).all()
    # (120, 80) -> esquina (L/2, -W/2); (60, 40) -> centro
    from analista.datos.statsbomb import a_metros
    assert np.allclose(a_metros([60, 40], cancha), (0, 0))
    assert np.allclose(a_metros([120, 80], cancha), (cancha.L2, -cancha.W2))


def test_tiros_goles_y_xg(eventos):
    t = ae.tiros(eventos)
    assert len(t) == 4
    assert t["xg_modelo"].between(0, 1).all()
    g = ae.goles(eventos)
    assert g["equipo"].value_counts().to_dict() == {"A": 2, "B": 1}
    r = ae.resumen(eventos)
    assert r["por_equipo"]["A"]["goles"] == 2 and r["por_equipo"]["B"]["goles"] == 1
    assert abs(r["por_equipo"]["A"]["xg"] - 1.0) < 1e-6


def test_red_de_pases_hasta_el_cambio(eventos):
    nodos, aristas = ae.red_de_pases(eventos, "A", min_pases=1)
    assert len(nodos) > 3 and (aristas["n"] >= 1).all()
    pases_A = eventos[(eventos["tipo"] == "pase") & (eventos["equipo"] == "A") & (eventos["exito"] == True)]  # noqa: E712
    assert aristas["n"].sum() < len(pases_A)          # se corta en el primer cambio (minuto 30)


def test_ppda_posesion_y_flujo(eventos):
    p = ae.posesion(eventos)
    assert abs(sum(p.values()) - 100) < 1
    assert all(v is None or v > 0 for v in ae.ppda(eventos).values())
    f = ae.flujo_xg(eventos)
    assert f.groupby("equipo")["xg"].max().round(2).to_dict() == {"A": 1.0, "B": 0.45}


def test_modelo_xg_razonable():
    cerca = xg.xg_punto(47.0, 0.0)
    lejos = xg.xg_punto(25.0, 0.0)
    angulo = xg.xg_punto(50.0, 25.0)
    assert 0 < lejos < cerca < 1 and angulo < cerca
    m = xg.cargar_modelo()
    assert {"basico", "contexto"} <= set(m["modelos"]) and m["penal"] == 0.78
