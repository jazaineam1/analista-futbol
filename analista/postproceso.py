"""Después de procesar un video: métricas, tablas y reporte HTML (lo usan la CLI y el panel)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .analisis.tracking import ParametrosMetricas, calcular_todo
from .cancha import Cancha
from .reportes.video import generar_reporte


def leer_csv(ruta: Path) -> pd.DataFrame:
    return pd.read_csv(ruta) if ruta.exists() and ruta.stat().st_size > 5 else pd.DataFrame()


def var_de_resultados(salida: Path, video: str | Path | None, resumen: dict):
    """Revisión VAR del análisis: var.json y las imágenes con las líneas sobre el video."""
    from .partido import Partido
    from .util import json_limpio
    from .vision.var_imagen import clips_var, imagenes_var
    p = Partido.video(salida)
    rev = p.var()
    fotos, clips = {}, {}
    for viejo in list(salida.glob("var_*.jpg")) + list(salida.glob("var_clip_*.mp4")):
        viejo.unlink(missing_ok=True)                  # no mezclar con los de un análisis anterior
    if video and Path(video).exists():
        args = (p.camara, video, salida, int(resumen.get("ancho_proceso") or 1280), p.cancha.largo, p.cancha.ancho,
                float(resumen.get("grabacion_desfase_s") or 0.0))
        fotos = imagenes_var(rev.destacadas(), *args)
        clips = clips_var(rev.destacadas(), *args, maximo=int(resumen.get("clips_var_max", 12)))
    ligeras = [{k: v for k, v in r.items() if k != "jugadores"} for r in rev.destacadas()]
    from .reportes.var import items_var
    nombres_jug = {i: f"#{int(i)}" for i in p.tracking["id"].dropna().unique() if float(i).is_integer()} if len(p.tracking) else {}
    items = items_var(rev, p._posiciones, nombres_jug, p.nombres, p.cancha.largo, p.cancha.ancho, fotos=fotos, clips=clips)
    (salida / "var_items.json").write_text(json.dumps(json_limpio({"items": items, "resumen": rev.resumen()}),
                                                      ensure_ascii=False), encoding="utf-8")
    (salida / "var.json").write_text(json.dumps(json_limpio({
        "resumen": rev.resumen(), "destacadas": ligeras, "fotos": {str(k): v for k, v in fotos.items()},
        "clips": {str(k): v for k, v in clips.items()},
        "fuera_de_lugar": [{k: r.get(k) for k in ("t", "equipo", "pasador", "receptor", "veredicto", "margen_m", "sigma_m",
                                                  "linea_por", "explicacion")} for r in rev.fueras_de_lugar]}),
        indent=1, ensure_ascii=False), encoding="utf-8")
    return rev, p._posiciones, fotos, clips


def postprocesar(salida: Path, cancha: Cancha, resumen: dict, ataca_derecha: str | None = None,
                 colores_kit: dict | None = None, titulo: str = "Análisis del partido", nota: str = "",
                 video: str | Path | None = None) -> dict:
    salida = Path(salida)
    df = leer_csv(salida / "tracking.csv")
    balon = leer_csv(salida / "balon.csv")
    cam = leer_csv(salida / "camara.csv")
    p = ParametrosMetricas(largo=cancha.largo, ancho=cancha.ancho, ataca_derecha=ataca_derecha)
    calc = calcular_todo(df, balon, cam, p)
    for nombre in ("forma", "toques", "pases", "eventos", "fisico", "poseedor"):
        tabla = calc[nombre]
        if isinstance(tabla, pd.DataFrame):
            tabla.to_csv(salida / f"{nombre}.csv", index=False)
    (salida / "metricas.json").write_text(json.dumps(calc["resumen"], indent=2, ensure_ascii=False, default=str),
                                          encoding="utf-8")
    if colores_kit:
        resumen["colores_kit"] = colores_kit
    resumen["titulo"], resumen["nota"] = titulo, nota          # para rehacer el reporte con `analista reporte`
    if video and Path(video).exists():
        resumen["video"] = str(Path(video).resolve())         # el VAR vuelve al video (cuerpo, imágenes, clips)
    (salida / "resumen_proceso.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
    var = None
    try:
        var = var_de_resultados(salida, video, resumen)
    except Exception as e:  # noqa: BLE001  (el análisis no se pierde si falla la revisión)
        import traceback
        traceback.print_exc()
        print(f"Aviso: no se pudo hacer la revisión VAR: {e}")
    ruta = generar_reporte(salida, df, calc, resumen, cancha, titulo, nota, var=var)
    return {"reporte": str(ruta), "resumen": calc["resumen"]}
