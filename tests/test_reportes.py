from analista.partido import Partido


def test_reporte_de_eventos(eventos, tmp_path):
    info = {"fuente": "prueba", "equipos": {"A": {"nombre": "Local"}, "B": {"nombre": "Visita"}},
            "marcador": {"A": 2, "B": 1}, "cancha": {"largo": 105, "ancho": 68}}
    p = Partido(info, eventos=eventos)
    ruta = p.reporte(tmp_path / "r.html")
    html = ruta.read_text(encoding="utf-8")
    assert "<svg" in html and "Local" in html and "xG" in html
    assert p.resumen()["por_equipo"]["A"]["goles"] == 2


def test_reporte_de_tracking(tracking_sintetico, tmp_path):
    tr, bal = tracking_sintetico
    info = {"fuente": "prueba", "equipos": {"A": {"nombre": "Rojo", "color": "#c00"}, "B": {"nombre": "Azul", "color": "#00c"}},
            "marcador": {"A": 0, "B": 0}, "cancha": {"largo": 105, "ancho": 68}, "fps": 10}
    p = Partido(info, tracking=tr.assign(nombre=tr["id"]), balon=bal)
    ruta = p.reporte(tmp_path / "t.html")
    assert "<svg" in ruta.read_text(encoding="utf-8")
    assert p.formacion("A")["posiciones"]


def test_datos_del_reporte_son_json_estandar(tracking_sintetico, tmp_path):
    """Los datos que trae el reporte se pueden leer con JSON estricto (sin NaN)."""
    import json
    import re
    tr, bal = tracking_sintetico
    tr = tr.copy()
    tr.loc[tr.index[:50], "x"] = float("nan")          # huecos, como en datos reales
    info = {"fuente": "prueba", "equipos": {"A": {"nombre": "Rojo"}, "B": {"nombre": "Azul"}},
            "marcador": {"A": 0, "B": 0}, "cancha": {"largo": 105, "ancho": 68}, "fps": 10}
    html = Partido(info, tracking=tr.assign(nombre=tr["id"]), balon=bal).reporte(tmp_path / "t.html").read_text(encoding="utf-8")
    datos = re.search(r'<script type="application/json" id="datos">(.*?)</script>', html, re.S).group(1)

    def no_nan(c):
        raise ValueError(f"constante no estándar: {c}")
    json.loads(datos.replace("<\\/", "</"), parse_constant=no_nan)
