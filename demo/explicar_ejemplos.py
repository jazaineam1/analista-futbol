"""Página "Cómo detecta" con los tres videos de ejemplo (la del sitio de documentación).

    python demo/explicar_ejemplos.py -o docs/deteccion --modelo modelos/yolo11n.pt

Usa los videos de demo/clips (ver generar_transmision.py y soccertrack_calibracion.py) y,
si existen, los videos anotados de demo/salidas para la sección final de cada clip.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from analista.vision.calibracion import Calibracion
from analista.vision.diagnostico import explicar_video, generar_pagina
from analista.vision.pipeline import ConfigPipeline, ruta_ffmpeg

EQUIPOS = {"Auckland FC": "#1f6fd1", "Newcastle Jets": "#f2f2f2"}

EJEMPLOS = [
    {"clave": "completo", "nombre": "Plano completo", "video": "demo/clips/completo/video.mp4",
     "calibracion": "demo/clips/completo/calibracion.json", "ancho": 1920, "imgsz": 1920, "equipos": EQUIPOS,
     "momentos": [(19.4, "Antes del centro · 0:19"), (22.6, "Remate · 0:22")],
     "descripcion": "Cámara fija que ve toda la cancha (transmisión sintética hecha con el tracking real del gol de "
                    "N. Moreno, Auckland FC – Newcastle Jets, A-League 2024/25). Los jugadores miden unos 25 píxeles: "
                    "por eso se procesa a 1920 px."},
    {"clave": "tv", "nombre": "Cámara de TV", "video": "demo/clips/tv/video.mp4",
     "calibracion": "demo/clips/tv/calibracion.json", "ancho": 1280, "imgsz": 1280, "equipos": EQUIPOS,
     "momentos": [(3.0, "Medio campo · 0:03"), (22.6, "Remate · 0:22")],
     "descripcion": "La cámara principal de la transmisión sigue la jugada con paneo y zoom (movimiento real de "
                    "la cámara de SkillCorner en esa jugada). Se calibra a mano solo el primer cuadro; el resto lo "
                    "sigue la app."},
    {"clave": "real", "nombre": "Video real", "video": "demo/clips/real/soccertrack.mp4",
     "calibracion": "demo/clips/real/calibracion.json", "ancho": 1920, "imgsz": 1920,
     "equipos": {"Azul": "#1f6fd1", "Blanco": "#f2f2f2"},
     "momentos": [(6.0, "Juego · 0:06")],
     "descripcion": "Partido universitario real grabado con una cámara fija gran angular (dataset SoccerTrack, "
                    "Universidad de Tsukuba). La distorsión del lente se corrige antes de detectar. El juez de línea "
                    "y el arquero aparecen como 'otro'."},
]


def webm(mp4: Path, destino: Path, ancho: int = 1280):
    ff = ruta_ffmpeg()
    destino.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", f"scale={ancho}:-2", "-c:v", "libx264",
                    "-crf", "26", "-preset", "medium", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an",
                    str(destino.with_suffix(".mp4"))], check=True)
    subprocess.run([ff, "-y", "-loglevel", "error", "-i", str(mp4), "-vf", f"scale={ancho}:-2", "-c:v", "libvpx-vp9",
                    "-b:v", "0", "-crf", "38", "-deadline", "good", "-cpu-used", "4", "-an", str(destino.with_suffix(".webm"))],
                   check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--salida", default="docs/deteccion")
    ap.add_argument("--modelo", default="modelos/yolo11n.pt")
    ap.add_argument("--solo", nargs="*", help="claves de los ejemplos a regenerar")
    ap.add_argument("--artefacto", action="store_true", help="página sin <html>/<head> (para publicar como artefacto)")
    a = ap.parse_args()
    salida = Path(a.salida)
    clips = []
    for ej in EJEMPLOS:
        if a.solo and ej["clave"] not in a.solo:
            continue
        print(f"== {ej['nombre']}")
        cal = Calibracion.cargar(ej["calibracion"])
        cfg = ConfigPipeline(detector="yolo", modelo_yolo=a.modelo, imgsz=ej["imgsz"], ancho_proceso=ej["ancho"],
                             equipos=ej["equipos"])
        nombres = dict(zip("AB", ej["equipos"]))
        momentos = explicar_video(ej["video"], cal, cfg, [t for t, _ in ej["momentos"]], salida / "img", ej["clave"], nombres,
                                  progreso=lambda n, t, o: print(f"\r  {t:5.1f} s -> {o:.1f} s", end="", flush=True))
        print()
        etiquetas = dict(ej["momentos"])
        for m in momentos:
            m["etiqueta"] = etiquetas.get(m["t"], f"{m['t']:.1f} s")
        anotado = Path("demo/salidas") / ej["clave"] / "video_anotado.mp4"
        video_rel = None
        if anotado.exists():
            webm(anotado, salida / "video" / f"{ej['clave']}.mp4")
            video_rel = f"video/{ej['clave']}.mp4"
        clips.append({"nombre": ej["nombre"], "descripcion": ej["descripcion"], "nombres": nombres,
                      "video": video_rel, "momentos": momentos})
    intro = ("La app no usa datos externos para ver el partido: todo sale de la imagen. Elige un video y un momento, "
             "y avanza por las 10 etapas en el orden en que la app las ejecuta, cuadro tras cuadro.")
    ruta = generar_pagina(salida / "index.html", clips, "Cómo detecta la app", intro, artefacto=a.artefacto,
                          inicio="../index.html")
    print(f"Página: {ruta}")


if __name__ == "__main__":
    main()
