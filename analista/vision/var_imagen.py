"""Imagen VAR sobre el video: las líneas del fuera de lugar dibujadas en el plano de la cancha.

Como en la TV, las líneas van "pintadas" sobre el césped en perspectiva: se proyectan con la
homografía de la cámara de ese cuadro (la misma que ubica a los jugadores en metros).
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

AZUL = (208, 151, 21)       # BGR
ROJO = (43, 54, 224)
VERDE = (87, 154, 29)
AMBAR = (0, 133, 201)
GRIS = (124, 132, 123)
COLOR_VEREDICTO = {"fuera de lugar": ROJO, "habilitado": VERDE, "dudoso": AMBAR, "gol": VERDE, "no entró": ROJO,
                   "salió": ROJO, "sin intervención": GRIS, "revisar": AMBAR, "dentro del área": ROJO,
                   "fuera del área": VERDE, "ocasión manifiesta": ROJO, "gol válido": VERDE}
TEXTO_VEREDICTO = {"fuera de lugar": "FUERA DE LUGAR", "habilitado": "HABILITADO", "dudoso": "DUDOSO",
                   "sin intervención": "ADELANTADO SIN INTERVENIR", "gol": "GOL", "no entró": "NO ENTRÓ",
                   "salió": "BALÓN FUERA", "no revisable": "NO REVISABLE", "revisar": "REVISAR",
                   "dentro del área": "DENTRO DEL ÁREA", "fuera del área": "FUERA DEL ÁREA",
                   "ocasión manifiesta": "OCASIÓN MANIFIESTA", "gol válido": "GOL VÁLIDO"}


def _fuente(tam: int):
    from PIL import ImageFont
    candidatas = ["DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
                  "C:/Windows/Fonts/segoeuib.ttf", "C:/Windows/Fonts/arialbd.ttf",
                  "/System/Library/Fonts/Supplemental/Arial Bold.ttf", "/Library/Fonts/Arial Bold.ttf"]
    for c in candidatas:
        try:
            return ImageFont.truetype(c, tam)
        except OSError:
            continue
    return ImageFont.load_default(size=tam)


def _texto(img: np.ndarray, partes: list[tuple[str, tuple]], x: int, y: int, tam: int) -> np.ndarray:
    """Chips de texto (fondo de color, letra blanca) con tildes: se dibujan con Pillow."""
    from PIL import Image, ImageDraw
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    dib = ImageDraw.Draw(pil, "RGBA")
    f = _fuente(tam)
    pad = max(4, tam // 3)
    for txt, color in partes:
        x0, y0, x1, y1 = dib.textbbox((x + pad, y + pad), txt, font=f)
        b, g, r = color[:3]
        alfa = color[3] if len(color) > 3 else 255
        dib.rounded_rectangle((x, y, x1 + pad, y1 + pad), radius=pad, fill=(r, g, b, alfa))
        dib.text((x + pad, y + pad), txt, font=f, fill=(255, 255, 255, 255))
        x = x1 + pad + max(6, tam // 3)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def _proyectar(Hc2i: np.ndarray, pts: np.ndarray, forma) -> np.ndarray | None:
    """Puntos de la cancha -> imagen, descartando los que quedan detrás de la cámara."""
    h, w = forma[:2]
    hom = np.column_stack([pts, np.ones(len(pts))]) @ Hc2i.T
    signo = np.sign(np.median(hom[:, 2])) or 1.0
    ok = hom[:, 2] * signo > 1e-9
    if ok.sum() < min(2, len(pts)):
        return None
    uv = hom[ok, :2] / hom[ok, 2:3]
    uv = np.clip(uv, -4 * w, 5 * w)
    return uv


def _parcial(ys: np.ndarray, frac: float) -> np.ndarray:
    """La parte de la línea ya trazada (de una banda a la otra) según el avance 0-1."""
    n = max(2, int(round(len(ys) * min(1.0, max(0.0, frac)))))
    return ys[:n]


def foco_imagen(H_img2cancha: np.ndarray, rev: dict, forma) -> tuple[float, float] | None:
    """Punto de la imagen donde mirar (entre el receptor y la línea, o el balón)."""
    Hc2i = np.linalg.inv(np.asarray(H_img2cancha, float))
    if rev.get("tipo") == "fuera de lugar" and rev.get("linea_x") is not None:
        xr, yr = rev.get("receptor_xy") or rev.get("defensor_xy") or (rev["linea_x"], 0.0)
        pt = ((float(rev["linea_x"]) + xr) / 2, yr)
    elif rev.get("balon"):
        pt = tuple(rev["balon"])
    elif rev.get("x") is not None:
        pt = (rev["x"], rev["y"])
    else:
        return None
    uv = _proyectar(Hc2i, np.array([pt]), forma)
    return None if uv is None or not len(uv) else (float(uv[0][0]), float(uv[0][1]))


def dibujar_var(frame: np.ndarray, H_img2cancha: np.ndarray, rev: dict, largo: float = 105.0,
                ancho: float = 68.0, avance: dict | None = None, rotulo: bool = True) -> np.ndarray:
    """Dibuja la revisión sobre el cuadro. H_img2cancha debe corresponder a la resolución de `frame`.
    avance: cuánto de cada parte está dibujado (0-1) para animarlo: {"def", "ata", "banda", "veredicto"}."""
    av = {"def": 1.0, "ata": 1.0, "banda": 1.0, "veredicto": 1.0, **(avance or {})}
    img = frame.copy()
    h, w = img.shape[:2]
    W2 = ancho / 2
    Hc2i = np.linalg.inv(np.asarray(H_img2cancha, float))
    grosor = max(2, w // 420)
    ys = np.linspace(-W2, W2, 120)
    ver = rev.get("veredicto", "")
    color_v = COLOR_VEREDICTO.get(ver, GRIS)
    if rev.get("tipo") == "fuera de lugar" and rev.get("linea_x") is not None:
        xl = float(rev["linea_x"])
        xr = float(rev["receptor_xy"][0]) if rev.get("receptor_xy") else None
        if xr is not None and av["banda"] > 0:
            borde = np.vstack([np.column_stack([np.full_like(ys, xl), ys]), np.column_stack([np.full_like(ys, xr), ys[::-1]])])
            poli = _proyectar(Hc2i, borde, img.shape)
            if poli is not None and len(poli) >= 3:
                capa = img.copy()
                cv2.fillPoly(capa, [poli.astype(np.int32)], color_v, cv2.LINE_AA)
                img = cv2.addWeighted(capa, 0.3 * av["banda"], img, 1 - 0.3 * av["banda"], 0)
        if rev.get("vision") and av["ata"] > 0:
            (bx, by), (gx, gy) = rev["vision"]
            f = av["ata"]
            seg = _proyectar(Hc2i, np.array([[bx, by], [bx + (gx - bx) * f, by + (gy - by) * f]]), img.shape)
            if seg is not None and len(seg) == 2:
                p0, p1 = seg.astype(int)
                n = max(2, int(np.hypot(*(p1 - p0)) / 12))
                for k in range(0, n, 2):           # punteada
                    a = p0 + (p1 - p0) * k / n
                    b = p0 + (p1 - p0) * min(1, (k + 1) / n)
                    cv2.line(img, tuple(int(v) for v in a), tuple(int(v) for v in b), (255, 255, 255), grosor, cv2.LINE_AA)
        if xr is not None and av["ata"] > 0:
            linea = _proyectar(Hc2i, np.column_stack([np.full_like(ys, xr), ys])[: len(_parcial(ys, av["ata"]))], img.shape)
            if linea is not None:
                cv2.polylines(img, [linea.astype(np.int32)], False, ROJO, grosor, cv2.LINE_AA)
        if av["def"] > 0:
            linea = _proyectar(Hc2i, np.column_stack([np.full_like(ys, xl), ys])[: len(_parcial(ys, av["def"]))], img.shape)
            if linea is not None:
                cv2.polylines(img, [linea.astype(np.int32)], False, AZUL, grosor, cv2.LINE_AA)
        # marcas en los pies del receptor y del defensor
        for xy, col, f in ((rev.get("receptor_xy"), ROJO, av["ata"]), (rev.get("defensor_xy"), AZUL, av["def"])):
            if not xy or f <= 0:
                continue
            c = _proyectar(Hc2i, np.array([[xy[0], xy[1]], [xy[0] + 0.8, xy[1]], [xy[0], xy[1] + 0.8]]), img.shape)
            if c is None or len(c) < 3:
                continue
            ax = max(6, int(np.hypot(*(c[1] - c[0]))))
            ay = max(3, int(np.hypot(*(c[2] - c[0]))))
            cv2.ellipse(img, (int(c[0][0]), int(c[0][1])), (ax, ay), 0, 0, int(360 * min(1, f)), col, grosor, cv2.LINE_AA)
    elif rev.get("tipo") in ("falta en el área", "ocasión manifiesta") and rev.get("x") is not None:
        lado = rev.get("direccion") or (1 if rev["x"] >= 0 else -1)
        x0 = lado * largo / 2
        borde = np.array([[x0, -20.16], [x0 - lado * 16.5, -20.16], [x0 - lado * 16.5, 20.16], [x0, 20.16]])
        tramos = np.vstack([np.linspace(borde[k], borde[k + 1], 40) for k in range(3)])
        parte = tramos[: max(2, int(len(tramos) * min(1, av["def"])))]
        linea = _proyectar(Hc2i, parte, img.shape)
        if linea is not None:
            cv2.polylines(img, [linea.astype(np.int32)], False, AMBAR, grosor + 1, cv2.LINE_AA)
        c = _proyectar(Hc2i, np.array([[rev["x"], rev["y"]]]), img.shape)
        if c is not None and len(c) and av["ata"] > 0:
            r = max(8, w // 90)
            cv2.circle(img, (int(c[0][0]), int(c[0][1])), int(r * (1 + 2 * (1 - av["ata"]))), color_v, grosor, cv2.LINE_AA)
    elif rev.get("balon"):
        bx, by = rev["balon"]
        if rev.get("tipo") in ("línea de gol", "balón fuera") and av["def"] > 0:
            lado = 1 if bx >= 0 else -1
            if rev.get("tipo") == "línea de gol" or rev.get("linea") == "de meta":
                pts = np.column_stack([np.full_like(ys, lado * largo / 2), ys])
            else:
                lado_y = 1 if by >= 0 else -1
                xs = np.linspace(-largo / 2, largo / 2, 160)
                pts = np.column_stack([xs, np.full_like(xs, lado_y * W2)])
            pts = pts[: max(2, int(len(pts) * min(1, av["def"])))]
            linea = _proyectar(Hc2i, pts, img.shape)
            if linea is not None:
                cv2.polylines(img, [linea.astype(np.int32)], False, AZUL, grosor, cv2.LINE_AA)
        c = _proyectar(Hc2i, np.array([[bx, by]]), img.shape)
        if c is not None and len(c) and av["ata"] > 0:
            r = max(8, w // 90)
            cv2.circle(img, (int(c[0][0]), int(c[0][1])), int(r * (1 + 2 * (1 - av["ata"]))), color_v, grosor, cv2.LINE_AA)
    if not rotulo or av["veredicto"] <= 0:
        return img
    # abajo a la izquierda (arriba suele estar el marcador de la transmisión)
    tam = max(14, h // 30)
    partes = [("VAR", (30, 30, 30, 235)), (TEXTO_VEREDICTO.get(ver, ver.upper()), color_v + (255,))]
    if rev.get("margen_m") is not None:
        m = rev["margen_m"] * min(1.0, av["veredicto"] * 1.4)
        txt = f"{'+' if m > 0 else '−' if m < 0 else '±'}{abs(m):.2f} m".replace(".", ",")
        if rev.get("sigma_m") is not None:
            txt += f"  ±{rev['sigma_m']:.2f}".replace(".", ",")
        partes.append((txt, (30, 30, 30, 215)))
    return _texto(img, partes, max(10, w // 80), h - int(tam * 1.9) - max(10, h // 40), tam)


def leer_cuadro(video: str | Path, t: float) -> np.ndarray | None:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(round(t * fps))))
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def imagenes_var(destacadas: list[dict], camara, video: str | Path, carpeta: Path, ancho_proceso: int,
                 largo: float, ancho: float, desfase_s: float = 0.0, maximo: int = 40, prefijo: str = "var") -> dict:
    """Guarda var_N.jpg para cada jugada (cuadro del video con las líneas) -> {t: nombre}."""
    import pandas as pd
    out = {}
    if camara is None or not len(camara) or not Path(video).exists():
        return out
    cam = camara.sort_values("t")
    tt = cam["t"].to_numpy(float)
    for k, r in enumerate(destacadas[:maximo]):
        if r.get("tipo") not in ("fuera de lugar", "línea de gol", "balón fuera"):
            continue
        t = float(r.get("t_max") or r.get("t") or 0.0)
        i = int(np.argmin(np.abs(tt - t)))
        if abs(tt[i] - t) > 0.3:
            continue
        fila = cam.iloc[i]
        if not bool(fila.get("calibrado", True)) or pd.isna(fila.get("h0")):
            continue
        frame = leer_cuadro(video, t - desfase_s)
        if frame is None:
            continue
        H = fila[[f"h{j}" for j in range(9)]].to_numpy(float).reshape(3, 3)
        s = ancho_proceso / frame.shape[1]
        H = H @ np.diag([s, s, 1.0])                       # píxel del video original -> píxel procesado -> cancha
        try:
            img = dibujar_var(frame, H, r, largo, ancho)
        except Exception as e:  # noqa: BLE001
            print(f"Aviso: no se pudo dibujar la jugada {k}: {e}", file=sys.stderr)
            continue
        nombre = f"{prefijo}_{k:02d}.jpg"
        cv2.imwrite(str(Path(carpeta) / nombre), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        out[round(float(r.get("t") or 0), 2)] = nombre
    return out



# ---------------------------------------------------------------------- clip animado (como en la TV)
def _suave(x: float) -> float:
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


def _zoom(img: np.ndarray, foco, z: float) -> np.ndarray:
    if z <= 1.001 or foco is None:
        return img
    h, w = img.shape[:2]
    cw, ch = w / z, h / z
    cx = min(max(foco[0], cw / 2), w - cw / 2)
    cy = min(max(foco[1], ch / 2), h - ch / 2)
    M = np.array([[z, 0, -(cx - cw / 2) * z], [0, z, -(cy - ch / 2) * z]], float)
    return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR)


def _sello(img: np.ndarray, texto: str) -> np.ndarray:
    """Rótulo arriba a la derecha (a la izquierda suele estar el marcador de la transmisión)."""
    h, w = img.shape[:2]
    tam = max(12, h // 40)
    f = _fuente(tam)
    pad = max(4, tam // 3)
    ancho_txt = int(f.getlength("VAR") + f.getlength(texto) + 4 * pad + max(6, tam // 3))
    return _texto(img, [("VAR", (12, 13, 12, 240)), (texto, (60, 60, 60, 220))], w - ancho_txt - max(10, w // 60),
                  max(10, h // 40), tam)


TITULO_CLIP = {"fuera de lugar": "REVISIÓN DE FUERA DE LUGAR", "línea de gol": "LÍNEA DE GOL", "balón fuera": "BALÓN FUERA",
               "falta en el área": "¿DENTRO DEL ÁREA?", "ocasión manifiesta": "OCASIÓN MANIFIESTA"}


def clip_var(video: str | Path, H_por_t, rev: dict, ruta: Path, largo: float, ancho: float, desfase_s: float = 0.0,
             antes_s: float = 2.0, despues_s: float = 1.2, congelado_s: float = 3.8, zoom_max: float = 1.45) -> Path | None:
    """Clip MP4 de la jugada como lo muestra el VAR en la TV: repetición en cámara lenta hasta el
    instante revisado, imagen congelada con zoom, las líneas que se trazan en perspectiva sobre el
    césped, el margen y el veredicto, y la jugada que sigue. H_por_t(t) -> homografía en píxeles del
    video original (o None si la cámara no está calibrada en ese instante)."""
    from .pipeline import _escritor_h264
    t_ev = float(rev.get("t_max") or rev.get("t") or 0.0)
    H = H_por_t(t_ev)
    if H is None:
        return None
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return None
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    i_ev = int(round((t_ev - desfase_s) * fps))
    i0 = max(0, i_ev - int(round(antes_s * fps)))
    i1 = i_ev + int(round(despues_s * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
    cuadros = []
    for _ in range(i0, i1 + 1):
        ok, f = cap.read()
        if not ok:
            break
        cuadros.append(f)
    cap.release()
    k_ev = i_ev - i0
    if k_ev < 0 or k_ev >= len(cuadros):
        return None
    titulo = TITULO_CLIP.get(rev.get("tipo"), "REVISIÓN")
    if rev.get("motivo") == "rebote":
        titulo = "FUERA DE LUGAR EN EL REBOTE"
    elif rev.get("motivo") == "interferencia":
        titulo = "INTERFERENCIA CON EL ARQUERO"
    base = cuadros[k_ev]
    foco = foco_imagen(H, rev, base.shape)
    esc = None
    try:
        ruta = Path(ruta)
        esc = _escritor_h264(ruta, base.shape, fps)

        def escribir(img):
            esc.stdin.write(np.ascontiguousarray(img).tobytes())
        # 1) repetición en cámara lenta (cada cuadro dos veces)
        for f in cuadros[:k_ev]:
            img = _sello(f, "REPETICIÓN")
            escribir(img)
            escribir(img)
        # 2) imagen congelada: zoom, líneas, margen y veredicto
        n = int(round(congelado_s * fps))
        for k in range(n):
            tau = k / fps
            av = {"def": _suave((tau - 0.5) / 0.7), "ata": _suave((tau - 1.2) / 0.7),
                  "banda": _suave((tau - 1.8) / 0.5), "veredicto": _suave((tau - 2.2) / 0.4)}
            img = dibujar_var(base, H, rev, largo, ancho, av, rotulo=False)
            img = _zoom(img, foco, 1 + (zoom_max - 1) * _suave(tau / 0.8))
            if tau < 0.16:                                            # destello al congelar
                img = cv2.addWeighted(img, 1 - 0.45 * (1 - tau / 0.16), np.full_like(img, 255), 0.45 * (1 - tau / 0.16), 0)
            img = _sello(img, titulo)
            if av["veredicto"] > 0:
                img = _rotulo(img, rev, av["veredicto"])
            escribir(img)
        # 3) la jugada sigue (vuelve el zoom) con el veredicto puesto
        for j, f in enumerate(cuadros[k_ev:]):
            z = 1 + (zoom_max - 1) * (1 - _suave(j / max(1, 0.4 * fps)))
            fz = _zoom(f, foco, z)
            escribir(_rotulo(_sello(fz, titulo), rev, 1.0))
    except (BrokenPipeError, OSError) as e:
        print(f"Aviso: no se pudo grabar el clip VAR: {e}", file=sys.stderr)
        return None
    finally:
        if esc is not None:
            esc.stdin.close()
            esc.wait()
    return ruta


def _rotulo(img: np.ndarray, rev: dict, avance: float) -> np.ndarray:
    """El rótulo del veredicto abajo a la izquierda (con el margen contando hasta su valor)."""
    h, w = img.shape[:2]
    ver = rev.get("veredicto", "")
    color_v = COLOR_VEREDICTO.get(ver, GRIS)
    tam = max(14, h // 30)
    partes = [(TEXTO_VEREDICTO.get(ver, ver.upper()), color_v + (255,))]
    if rev.get("margen_m") is not None:
        m = rev["margen_m"] * min(1.0, avance * 1.4)
        txt = f"{'+' if m > 0 else '−' if m < 0 else '±'}{abs(m):.2f} m".replace(".", ",")
        if rev.get("sigma_m") is not None:
            txt += f"  ±{rev['sigma_m']:.2f}".replace(".", ",")
        partes.append((txt, (30, 30, 30, 215)))
    return _texto(img, partes, max(10, w // 80), h - int(tam * 1.9) - max(10, h // 40), tam)


def clips_var(destacadas: list[dict], camara, video: str | Path, carpeta: Path, ancho_proceso: int, largo: float,
              ancho: float, desfase_s: float = 0.0, maximo: int = 12, prefijo: str = "var_clip") -> dict:
    """Clips animados de las jugadas más importantes -> {t: nombre.mp4}."""
    import pandas as pd
    out = {}
    if camara is None or not len(camara) or not Path(video).exists():
        return out
    cam = camara.sort_values("t")
    tt = cam["t"].to_numpy(float)
    cap = cv2.VideoCapture(str(video))
    w_video = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or ancho_proceso)
    cap.release()
    s = ancho_proceso / max(1, w_video)

    def H_por_t(t):
        i = int(np.argmin(np.abs(tt - t)))
        if abs(tt[i] - t) > 0.3:
            return None
        fila = cam.iloc[i]
        if not bool(fila.get("calibrado", True)) or pd.isna(fila.get("h0")):
            return None
        return fila[[f"h{j}" for j in range(9)]].to_numpy(float).reshape(3, 3) @ np.diag([s, s, 1.0])

    def prioridad(r):
        v = r.get("veredicto")
        return (0 if r.get("tipo") == "línea de gol" else 1 if v in ("fuera de lugar", "ocasión manifiesta", "revisar") else
                2 if v == "dudoso" or r.get("en_gol") else 3)
    # los habilitados por varios metros no necesitan repetición: no hay nada que mirar
    candidatas = [r for r in destacadas if r.get("tipo") in ("fuera de lugar", "línea de gol", "balón fuera",
                                                             "falta en el área", "ocasión manifiesta")
                  and r.get("veredicto") not in ("no revisable", "sin receptor", "sin intervención")
                  and not (r.get("veredicto") == "habilitado" and (r.get("margen_m") or 0) < -3.0)]
    for k, r in enumerate(sorted(candidatas, key=prioridad)[:maximo]):
        nombre = f"{prefijo}_{k:02d}.mp4"
        try:
            ok = clip_var(video, H_por_t, r, Path(carpeta) / nombre, largo, ancho, desfase_s)
        except Exception as e:  # noqa: BLE001
            print(f"Aviso: no se pudo armar el clip de la jugada {k}: {e}", file=sys.stderr)
            ok = None
        if ok is not None:
            out[round(float(r.get("t") or 0), 2)] = nombre
    return out


# ---------------------------------------------------------------------- punto más adelantado del cuerpo
def extremo_cuerpo(frame: np.ndarray, caja: tuple[float, float, float, float], H_img2cancha: np.ndarray,
                   direccion: int) -> float | None:
    """Coordenada x (metros) de la parte del cuerpo más cercana a la línea de meta rival, como mide el
    VAR (pie, rodilla, cadera, hombro o cabeza; los brazos no cuentan).

    caja = (u, v, alto, ancho) en píxeles de `frame` (u, v: centro del borde inferior). La silueta es
    lo que no es césped dentro de la caja. Cada punto se baja en vertical hasta la altura de los pies
    (como la línea que traza el VAR en el suelo) y se proyecta a la cancha. Se descartan los brazos
    con una regla simple: en la franja del torso solo cuenta la parte central."""
    u, v, alto, ancho = caja
    if not np.isfinite([u, v, alto, ancho]).all() or alto < 12 or ancho < 4:
        return None
    h, w = frame.shape[:2]
    x1, x2 = int(max(0, u - 0.65 * ancho)), int(min(w, u + 0.65 * ancho))
    y1, y2 = int(max(0, v - alto)), int(min(h, v + 2))
    if x2 - x1 < 4 or y2 - y1 < 8:
        return None
    rec = frame[y1:y2, x1:x2]
    hsv = cv2.cvtColor(rec, cv2.COLOR_BGR2HSV)
    cesped = cv2.inRange(hsv, (30, 35, 35), (95, 255, 255)) > 0
    linea = (hsv[..., 1] < 60) & (hsv[..., 2] > 170)
    cuerpo = (~cesped).astype(np.uint8)
    cuerpo = cv2.morphologyEx(cuerpo, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(cuerpo, 8)
    if n <= 1:
        return None
    # el componente que contiene el centro de la caja (el jugador, no una línea pintada ni otro)
    cx, cy = int(u - x1), int((y2 - y1) * 0.55)
    k = lab[min(cy, lab.shape[0] - 1), min(max(cx, 0), lab.shape[1] - 1)]
    if k == 0:
        k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    sil = (lab == k) & ~(linea & (np.arange(rec.shape[0])[:, None] > 0.85 * rec.shape[0]))
    filas = np.arange(rec.shape[0])[:, None] / max(1, rec.shape[0])
    cols = (np.arange(rec.shape[1])[None, :] + x1 - u) / max(1.0, ancho)
    torso = (filas > 0.17) & (filas < 0.5)
    sil &= ~torso | (np.abs(cols) < 0.22)            # brazos fuera
    ys_, xs_ = np.nonzero(sil)
    if len(xs_) < 10:
        return None
    pts = np.column_stack([xs_ + x1, np.full(len(xs_), float(v))])        # bajados a la altura de los pies
    hom = np.column_stack([pts, np.ones(len(pts))]) @ np.asarray(H_img2cancha, float).T
    xy = hom[:, :2] / hom[:, 2:3]
    a = xy[:, 0] * direccion
    return float(np.percentile(a, 98) * direccion)                       # 98 %: sin píxeles sueltos


def afinar_con_cuerpo(revisiones: list[dict], tracking, camara, video: str | Path, ancho_proceso: int,
                      desfase_s: float = 0.0, k_dudoso: float = 2.0) -> int:
    """Rehace el margen de cada fuera de lugar con el punto más adelantado del cuerpo del receptor y
    del penúltimo defensor (en vez del centro de los pies). Devuelve cuántas jugadas se afinaron."""
    import pandas as pd

    from ..analisis.var import id_texto
    if camara is None or not len(camara) or not Path(video).exists() or tracking is None or "u" not in tracking:
        return 0
    cam = camara.sort_values("t")
    tt = cam["t"].to_numpy(float)
    hechos = 0
    for r in revisiones:
        if r.get("tipo") != "fuera de lugar" or r.get("receptor") is None or r.get("margen_m") is None \
                or r.get("veredicto") in ("no revisable", "sin receptor", "no aplica"):
            continue
        t = float(r["t"])
        i = int(np.argmin(np.abs(tt - t)))
        fila = cam.iloc[i]
        if abs(tt[i] - t) > 0.2 or not bool(fila.get("calibrado", True)) or pd.isna(fila.get("h0")):
            continue
        frame = leer_cuadro(video, t - desfase_s)
        if frame is None:
            continue
        s = frame.shape[1] / ancho_proceso
        Hp = fila[[f"h{j}" for j in range(9)]].to_numpy(float).reshape(3, 3)
        H = Hp @ np.diag([1 / s, 1 / s, 1.0])                 # píxel del video original -> cancha
        cuadro = int(fila["cuadro"])
        filas_c = tracking[tracking["cuadro"] == cuadro]
        d = int(r.get("direccion") or 1)

        def caja(pid):
            if pid is None:
                return None
            f = filas_c[filas_c["id"].map(id_texto) == id_texto(pid)]
            if not len(f):
                return None
            f = f.iloc[0]
            return (f["u"] * s, f["v"] * s, f["alto_px"] * s, f["ancho_px"] * s)
        c_rec = caja(r["receptor"])
        if c_rec is None:
            continue
        x_rec = extremo_cuerpo(frame, c_rec, H, d)
        if r.get("linea_por") == "balón":
            x_def = float(r["linea_x"])
        else:
            c_def = caja(r.get("defensor"))
            x_def = extremo_cuerpo(frame, c_def, H, d) if c_def is not None else None
        if x_rec is None or x_def is None:
            continue
        # tapados entre sí: la silueta mezcla a dos jugadores y el extremo no es confiable
        if any("tapado" in a for a in r.get("avisos", [])):
            continue
        margen = (x_rec - x_def) * d
        if abs(margen - r["margen_m"]) > 1.5:
            continue                        # la silueta no corresponde (otro jugador pegado): se deja el centro
        r["margen_centro_m"] = r["margen_m"]
        r["margen_m"] = round(margen, 2)
        r["cuerpo"] = {"receptor_x": round(x_rec, 2), "defensor_x": round(x_def, 2)}
        r["linea_x"] = round(x_def, 2) if r.get("linea_por") != "balón" else r["linea_x"]
        if r.get("receptor_xy"):
            r["receptor_xy"] = [round(x_rec, 2), r["receptor_xy"][1]]
        sg = float(r.get("sigma_m") or 0.3)
        if r.get("veredicto") in ("fuera de lugar", "habilitado", "dudoso"):
            r["veredicto"] = "fuera de lugar" if margen > k_dudoso * sg else "habilitado" if margen < -k_dudoso * sg else "dudoso"
            del_ = "del balón" if r.get("linea_por") == "balón" else "del penúltimo defensor"
            m_, s_ = f"{abs(margen):.2f}".replace(".", ","), f"{sg:.2f}".replace(".", ",")
            r["explicacion"] = (f"Adelantado {m_} m respecto {del_} (±{s_} m)." if r["veredicto"] == "fuera de lugar" else
                                f"Habilitado: {m_} m detrás {del_} (±{s_} m)." if r["veredicto"] == "habilitado" else
                                f"Dudoso: {m_} m {'adelantado respecto' if margen > 0 else 'detrás'} {del_}, dentro del error (±{s_} m).")
        r["explicacion"] += (f" Medido con la parte del cuerpo más adelantada (con el centro de los pies daba "
                             f"{r['margen_centro_m']:+.2f} m)".replace(".", ",") + ".")
        hechos += 1
    return hechos
