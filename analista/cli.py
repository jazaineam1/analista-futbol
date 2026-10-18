"""Línea de comandos: `analista <comando>`.

    analista panel                                    panel web local (recomendado)
    analista analizar partido.mp4 -c cal.json         procesar un video
    analista vivo disp:0 -c cal.json                  seguir un partido en vivo (capturadora HDMI)
    analista calibrar partido.mp4 --puntos "…"        crear una calibración sin el panel
    analista verificar partido.mp4 -c cal.json        dibujar la cancha calibrada sobre un cuadro
    analista explicar partido.mp4 -c cal.json -t 20   página con las etapas de la detección
    analista reporte resultados/                      rehacer el reporte de un análisis
    analista datos statsbomb competiciones            explorar datos abiertos
    analista datos statsbomb reporte 3869685          reporte de un partido (eventos)
    analista datos skillcorner reporte 1886347        reporte de un partido (tracking)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__


def _equipos(txt: str | None) -> dict:
    """'Azul=#1f6fd1,Blanco=#f2f2f2' -> {'Azul': '#1f6fd1', 'Blanco': '#f2f2f2'}"""
    out = {}
    for parte in (txt or "").split(","):
        if "=" in parte:
            n, c = parte.split("=", 1)
            out[n.strip()] = c.strip()
    return out


def _config(a):
    from .vision.pipeline import ConfigPipeline
    return ConfigPipeline(detector=a.detector, modelo_yolo=a.modelo, imgsz=a.imgsz, conf=a.conf, ancho_proceso=a.ancho,
                          paso=a.paso, equipos=_equipos(a.equipos),
                          camara_fija=True if a.camara_fija else None)


def _opciones_video(p):
    p.add_argument("-c", "--calibracion", required=True, help="calibracion.json (del panel o de `analista calibrar`)")
    p.add_argument("--detector", default="yolo", choices=["yolo", "cesped"], help="red neuronal o detector clásico")
    p.add_argument("--modelo", default="yolo11n.pt", help="pesos YOLO (se descargan solos la primera vez)")
    p.add_argument("--imgsz", type=int, default=1280, help="tamaño de entrada de la red")
    p.add_argument("--conf", type=float, default=None,
                   help="confianza mínima de la red; por defecto 0,03 a ≥1600 px y 0,05 a menos")
    p.add_argument("--ancho", type=int, default=1280, help="ancho de procesamiento del video")
    p.add_argument("--paso", type=int, default=1, help="procesar 1 de cada N cuadros")
    p.add_argument("--equipos", help="nombres y colores: 'Azul=#1f6fd1,Blanco=#f2f2f2'")
    p.add_argument("--camara-fija", action="store_true", help="la cámara no se mueve (fuerza el modo fijo)")


def cmd_panel(a):
    from .panel.servidor import main
    main(a.carpeta, a.puerto, a.ancho, not a.sin_navegador, a.modelo, a.host)


def cmd_analizar(a, vivo: bool = False):
    from .postproceso import postprocesar
    from .vision.calibracion import Calibracion
    from .vision.fuentes import abrir_fuente
    from .vision.pipeline import Pipeline
    cal = Calibracion.cargar(a.calibracion)
    cfg = _config(a)
    fuente = abrir_fuente(a.fuente, inicio_s=getattr(a, "inicio", 0.0) or 0.0, fin_s=getattr(a, "fin", None),
                          fps=getattr(a, "fps", 10.0))
    pipe = Pipeline(cal, cfg)
    print(f"Analizando {a.fuente} -> {a.salida}")
    resumen = pipe.correr(fuente, Path(a.salida), video_anotado=not a.sin_video, mostrar=vivo or a.mostrar,
                          max_cuadros=getattr(a, "max_cuadros", None))
    fuente.cerrar()
    kits = dict(zip("AB", cfg.equipos.values())) if cfg.equipos else None
    r = postprocesar(Path(a.salida), cal.cancha, resumen, a.ataca_derecha, kits, a.titulo, Path(a.fuente).name)
    pos = r["resumen"].get("posesion_pct")
    print(f"Listo: {resumen['cuadros_procesados']} cuadros en {resumen['segundos_proceso']:.0f} s "
          f"({resumen['cuadros_por_segundo']} cuadros/s) · reporte: {r['reporte']}")
    if pos:
        print(f"Posesión A {pos['A']}% · B {pos['B']}%")


def cmd_calibrar(a):
    """Puntos 'u,v=x,y;u,v=x,y;…' (píxeles del video original = metros con origen en el centro)."""
    import numpy as np

    from .cancha import Cancha, aplicar_h, h_desde_puntos
    from .vision.calibracion import Calibracion
    from .vision.fuentes import FuenteArchivo
    pares = []
    for parte in a.puntos.split(";"):
        if not parte.strip():
            continue
        img, can = parte.split("=")
        pares.append({"imagen": [float(v) for v in img.split(",")], "cancha": [float(v) for v in can.split(",")]})
    if len(pares) < 4:
        sys.exit("Hacen falta al menos 4 pares de puntos")
    f = FuenteArchivo(a.video)
    fps = f.fps
    _, _, frame = next(iter(f))
    f.cerrar()
    H = h_desde_puntos([p["imagen"] for p in pares], [p["cancha"] for p in pares])
    err = np.linalg.norm(aplicar_h(H, [p["imagen"] for p in pares]) - np.array([p["cancha"] for p in pares]), axis=1)
    cal = Calibracion(Cancha(a.largo, a.ancho_cancha), (frame.shape[1], frame.shape[0]), a.camara_fija, None,
                      [{"cuadro": int(round(a.t * fps)), "puntos": pares, "H_img2cancha": H.tolist()}])
    cal.guardar(a.salida)
    print(f"Calibración guardada en {a.salida} · error medio {err.mean():.2f} m (máx {err.max():.2f} m)")


def cmd_verificar(a):
    import cv2
    import numpy as np

    from .cancha import mascara_lineas_proyectadas
    from .vision.calibracion import Calibracion
    cal = Calibracion.cargar(a.calibracion)
    cap = cv2.VideoCapture(a.video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cuadro = int(round(a.t * fps)) if a.t is not None else int(cal.claves[0].get("cuadro") or 0)
    cap.set(cv2.CAP_PROP_POS_FRAMES, cuadro)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        sys.exit("No se pudo leer el cuadro")
    corr = cal.corrector_lente(frame.shape[1], frame.shape[0])
    if corr is not None:
        frame = corr(frame)
    Hs = cal.homografias(frame.shape[1], frame.shape[0])
    H = Hs[cuadro] if cuadro in Hs else next(iter(Hs.values()))
    m = mascara_lineas_proyectadas(np.linalg.inv(H), frame.shape, cal.cancha, grosor=max(2, frame.shape[1] // 600))
    frame[m > 0] = (40, 40, 255)
    cv2.imwrite(a.salida, frame)
    print(f"Cuadro {cuadro} con la cancha calibrada en rojo: {a.salida}")
    if cuadro not in Hs and not cal.camara_fija:
        print(f"Aviso: la cámara se mueve y la calibración es del cuadro {next(iter(Hs))}; en otros cuadros las "
              "líneas no tienen por qué calzar (la app sigue el movimiento al analizar).")


def cmd_explicar(a):
    from .vision.calibracion import Calibracion
    from .vision.diagnostico import explicar_video, generar_pagina
    cal = Calibracion.cargar(a.calibracion)
    cfg = _config(a)
    tiempos = [float(t) for t in a.tiempos.split(",")]
    salida = Path(a.salida)
    nombres = dict(zip("AB", cfg.equipos.keys())) if cfg.equipos else {"A": "Equipo A", "B": "Equipo B"}
    momentos = explicar_video(a.video, cal, cfg, tiempos, salida.parent / "img", salida.stem, nombres,
                              progreso=lambda n, t, o: print(f"\r  {t:6.1f} s -> {o:.1f} s", end="", flush=True))
    print()
    for m in momentos:
        m["etiqueta"] = f"{int(m['t'] // 60)}:{m['t'] % 60:04.1f}"
    clip = {"nombre": Path(a.video).stem, "descripcion": "", "nombres": nombres, "video": None, "momentos": momentos}
    generar_pagina(salida, [clip], a.titulo, "Cada etapa de la detección sobre el cuadro elegido, en el orden en que la app la ejecuta.")
    print(f"Página: {salida}")


def cmd_reporte(a):
    from .partido import Partido
    p = Partido.video(a.carpeta)
    ruta = p.reporte(Path(a.salida) if a.salida else Path(a.carpeta) / "reporte.html", a.titulo)
    print(f"Reporte: {ruta}")


def _tabla(df, n=None):
    import pandas as pd
    with pd.option_context("display.max_rows", n or 500, "display.width", 160, "display.max_colwidth", 40):
        print(df.to_string(index=False))


def cmd_datos(a):
    if a.proveedor == "statsbomb":
        from .datos import statsbomb as sb
        if a.accion == "competiciones":
            _tabla(sb.competiciones())
        elif a.accion == "partidos":
            if len(a.ids) < 2:
                sys.exit("Uso: analista datos statsbomb partidos COMPETICION_ID TEMPORADA_ID")
            _tabla(sb.partidos(int(a.ids[0]), int(a.ids[1]))[
                ["partido_id", "fecha", "local", "goles_local", "goles_visitante", "visitante", "fase"]])
        elif a.accion == "reporte":
            from .partido import Partido
            for pid in a.ids:
                p = Partido.statsbomb(int(pid), a.competicion, a.temporada)
                ruta = p.reporte(Path(a.salida or ".") / f"statsbomb_{pid}.html")
                print(f"{p.nombres['A']} {p.info['marcador']['A']}-{p.info['marcador']['B']} {p.nombres['B']}: {ruta}")
        elif a.accion == "eventos":
            for pid in a.ids:
                ruta = Path(a.salida or ".") / f"statsbomb_{pid}_eventos.csv"
                df = sb.eventos(int(pid), nombres_cortos=True).drop(columns=["contexto"])
                df.to_csv(ruta, index=False)
                print(f"{len(df)} eventos: {ruta}")
    else:
        from .datos import skillcorner as sc
        if a.accion in ("competiciones", "partidos"):
            _tabla(sc.partidos())
        elif a.accion == "reporte":
            from .partido import Partido
            for pid in a.ids:
                p = Partido.skillcorner(int(pid))
                ruta = p.reporte(Path(a.salida or ".") / f"skillcorner_{pid}.html")
                print(f"{p.nombres['A']} - {p.nombres['B']}: {ruta}")
        elif a.accion == "eventos":
            for pid in a.ids:
                d = sc.cargar(int(pid))
                ruta = Path(a.salida or ".") / f"skillcorner_{pid}_tracking.csv"
                d["tracking"].to_csv(ruta, index=False)
                print(f"{len(d['tracking'])} filas de tracking: {ruta}")


def construir_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="analista", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", action="version", version=f"analista {__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="comando")

    p = sub.add_parser("panel", help="abrir el panel web local")
    p.add_argument("--carpeta", default="proyecto", help="donde se guardan calibración, resultados y reportes")
    p.add_argument("--puerto", type=int, default=8765)
    p.add_argument("--ancho", type=int, default=1280, help="ancho de la vista previa y de la calibración")
    p.add_argument("--modelo", default=None, help="pesos YOLO a usar (por defecto yolo11n.pt)")
    p.add_argument("--host", default="127.0.0.1", help="0.0.0.0 para abrirlo desde otro equipo de la red")
    p.add_argument("--sin-navegador", action="store_true")
    p.set_defaults(func=cmd_panel)

    p = sub.add_parser("analizar", help="procesar un video (archivo)")
    p.add_argument("fuente", help="archivo de video")
    _opciones_video(p)
    p.add_argument("-o", "--salida", default="resultados")
    p.add_argument("--inicio", type=float, default=0.0, help="segundo de inicio")
    p.add_argument("--fin", type=float, default=None, help="segundo final")
    p.add_argument("--max-cuadros", type=int, default=None)
    p.add_argument("--titulo", default="Análisis del partido")
    p.add_argument("--ataca-derecha", choices=["A", "B"], default=None, help="qué equipo ataca hacia +x al inicio")
    p.add_argument("--sin-video", action="store_true", help="no grabar el video anotado")
    p.add_argument("--mostrar", action="store_true", help="ventana con el video anotado")
    p.set_defaults(func=cmd_analizar)

    p = sub.add_parser("vivo", help="seguir un partido en vivo: disp:0 (capturadora), pantalla o rtsp://")
    p.add_argument("fuente")
    _opciones_video(p)
    p.add_argument("-o", "--salida", default="vivo")
    p.add_argument("--fps", type=float, default=10.0, help="cuadros por segundo a procesar")
    p.add_argument("--titulo", default="Partido en vivo")
    p.add_argument("--ataca-derecha", choices=["A", "B"], default=None)
    p.add_argument("--sin-video", action="store_true")
    p.add_argument("--mostrar", action="store_true", default=True)
    p.set_defaults(func=lambda a: cmd_analizar(a, vivo=True))

    p = sub.add_parser("calibrar", help="crear calibracion.json a partir de pares de puntos")
    p.add_argument("video")
    p.add_argument("--puntos", required=True, help="'u,v=x,y;…' píxel del video = metros (origen en el centro, y hacia arriba)")
    p.add_argument("-t", type=float, default=0.0, help="segundo del cuadro calibrado")
    p.add_argument("--largo", type=float, default=105.0)
    p.add_argument("--ancho-cancha", type=float, default=68.0)
    p.add_argument("--camara-fija", action="store_true")
    p.add_argument("-o", "--salida", default="calibracion.json")
    p.set_defaults(func=cmd_calibrar)

    p = sub.add_parser("verificar", help="dibujar la cancha calibrada sobre un cuadro")
    p.add_argument("video")
    p.add_argument("-c", "--calibracion", required=True)
    p.add_argument("-t", type=float, default=None)
    p.add_argument("-o", "--salida", default="verificacion.jpg")
    p.set_defaults(func=cmd_verificar)

    p = sub.add_parser("explicar", help="página con las 10 etapas de la detección en uno o más instantes")
    p.add_argument("video")
    _opciones_video(p)
    p.add_argument("-t", "--tiempos", required=True, help="segundos separados por comas: 20.2,23.2")
    p.add_argument("-o", "--salida", default="explicacion/pagina.html")
    p.add_argument("--titulo", default="Cómo detecta la app")
    p.set_defaults(func=cmd_explicar)

    p = sub.add_parser("reporte", help="rehacer el reporte HTML de una carpeta de resultados")
    p.add_argument("carpeta")
    p.add_argument("-o", "--salida", default=None)
    p.add_argument("--titulo", default=None)
    p.set_defaults(func=cmd_reporte)

    p = sub.add_parser("datos", help="datos abiertos: StatsBomb (eventos) y SkillCorner (tracking)")
    p.add_argument("proveedor", choices=["statsbomb", "skillcorner"])
    p.add_argument("accion", choices=["competiciones", "partidos", "reporte", "eventos"],
                   help="competiciones | partidos COMP TEMP | reporte ID… | eventos ID… (exporta CSV)")
    p.add_argument("ids", nargs="*")
    p.add_argument("-o", "--salida", default=None, help="carpeta de salida")
    p.add_argument("--competicion", type=int, default=None, help="StatsBomb: id de competición (para fecha y fase)")
    p.add_argument("--temporada", type=int, default=None, help="StatsBomb: id de temporada")
    p.set_defaults(func=cmd_datos)
    return ap


def main(argv=None):
    a = construir_parser().parse_args(argv)
    try:
        a.func(a)
    except KeyboardInterrupt:
        print("\nInterrumpido")
        return 130
    except FileNotFoundError as e:
        print(f"Error: no existe el archivo {e.filename or e}", file=sys.stderr)
        return 2
    except RuntimeError as e:          # p. ej. falta ultralytics, video ilegible
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
