import pytest

from analista import __version__
from analista.cli import construir_parser


def test_cli_parser():
    ap = construir_parser()
    a = ap.parse_args(["datos", "statsbomb", "partidos", "43", "106"])
    assert a.proveedor == "statsbomb" and a.ids == ["43", "106"]
    with pytest.raises(SystemExit):
        ap.parse_args(["--version"])
    assert __version__.count(".") == 2


@pytest.fixture
def cliente(tmp_path):
    from analista.panel.servidor import Estado, crear_app
    app = crear_app(Estado(tmp_path))
    app.testing = True
    return app.test_client()


def test_panel_paginas(cliente):
    r = cliente.get("/")
    assert r.status_code == 200 and b"Analista" in r.data
    assert cliente.get("/api/estado").json["calibracion"] is False
    c = cliente.get("/api/cancha?largo=100&ancho=64").json
    assert c["largo"] == 100 and len(c["puntos"]) > 10


def test_panel_calibracion(cliente):
    pares = [{"imagen": [60, 690], "cancha": [-52.5, -34]}, {"imagen": [1220, 690], "cancha": [52.5, -34]},
             {"imagen": [1010, 150], "cancha": [52.5, 34]}, {"imagen": [270, 150], "cancha": [-52.5, 34]}]
    r = cliente.post("/api/calibracion/previsualizar", json={"pares": pares})
    assert r.status_code == 200 and len(r.json["lineas"]) > 5 and max(r.json["errores"]) < 0.01
    r = cliente.post("/api/calibracion", json={"pares": pares, "camara_fija": True})
    assert r.json["ok"] and cliente.get("/api/estado").json["calibracion"] is True
    # puntos en línea recta: aviso
    recta = [{"imagen": [100 * i, 300 + i], "cancha": [10.0 * i, 0.0]} for i in range(4)]
    assert cliente.post("/api/calibracion/previsualizar", json={"pares": recta}).json.get("avisos")


def test_panel_sin_fuente(cliente):
    assert cliente.post("/api/analizar", json={}).status_code == 400
    assert cliente.post("/api/fuente", json={"spec": "/no/existe.mp4"}).status_code == 400


def test_panel_varias_tomas(cliente, tmp_path):
    import json
    pares = [{"imagen": [60, 690], "cancha": [-52.5, -34]}, {"imagen": [1220, 690], "cancha": [52.5, -34]},
             {"imagen": [1010, 150], "cancha": [52.5, 34]}, {"imagen": [270, 150], "cancha": [-52.5, 34]}]
    cliente.post("/api/calibracion", json={"pares": pares, "t": 0})
    r = cliente.post("/api/calibracion", json={"pares": pares, "t": 0, "agregar": True}).json
    assert r["tomas"] == 1                      # mismo cuadro: se reemplaza
    cal = json.loads(cliente.get("/api/calibracion").data)
    assert len(cal["claves"]) == 1


def test_panel_resultados_guardados(tmp_path):
    """Tras reiniciar el panel, la pestaña Resultados lee el resumen guardado (con NaN convertidos)."""
    from analista.panel.servidor import Estado, crear_app
    res = tmp_path / "resultados"
    res.mkdir()
    (res / "metricas.json").write_text('{"posesion_pct": {"A": 60.0, "B": NaN}, "eventos": [{"tipo": "gol"}]}')
    (res / "tracking.csv").write_text("cuadro,t\n")
    app = crear_app(Estado(tmp_path))
    d = app.test_client().get("/api/resultados").json
    assert d["resumen"]["posesion_pct"] == {"A": 60.0, "B": None}
    assert d["archivos"] == ["metricas.json", "tracking.csv"]


def test_panel_var_sin_analisis(cliente):
    """Las herramientas del VAR responden con un mensaje claro si todavía no hay análisis."""
    assert cliente.get("/api/var").json["items"] == []
    assert cliente.get("/api/var/cuadro.jpg?t=1").status_code == 404
    r = cliente.post("/api/var/instante", json={"t": 1.0, "equipo": "A"})
    assert r.status_code == 400 and "analiza" in r.json["error"]
    r = cliente.post("/api/var/falta", json={"t": 1.0, "equipo": "A", "u": 10, "v": 10})
    assert r.status_code == 400 and "analiza" in r.json["error"]
