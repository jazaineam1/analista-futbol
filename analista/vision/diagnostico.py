"""Vista paso a paso: cómo la app pasa de un cuadro de video a datos tácticos.

Corre el pipeline real (no una simulación) con el modo diagnóstico activo y, en los
instantes pedidos, dibuja lo que hizo cada etapa: césped, líneas, calibración, personas,
equipos, balón, proyección a metros, seguimiento y lectura táctica.
"""
from __future__ import annotations

import html
import json
from dataclasses import dataclass, field
from itertools import combinations  # noqa: F401  (usado por formacion vía analisis)
from pathlib import Path

import cv2
import numpy as np
from scipy.spatial import ConvexHull

from ..analisis.tracking import ParametrosMetricas, direccion_ataque
from ..cancha import aplicar_h, mascara_lineas_proyectadas
from ..reportes.base import SERIE, SvgCancha
from .deteccion import mascara_cajas, mascara_lineas

# Colores de identidad (BGR) para dibujar sobre el video: los mismos del reporte
BGR = {"A": (214, 120, 42), "B": (52, 104, 235), "otro": (0, 200, 255), None: (170, 170, 170)}


@dataclass
class Etapa:
    clave: str
    corto: str           # nombre corto para la barra de etapas
    titulo: str
    que: str             # qué hace
    como: str            # cómo lo hace
    imagen: np.ndarray | None = None
    svg: str | None = None
    datos: list = field(default_factory=list)   # [(etiqueta, valor), ...]


def _e(s) -> str:
    return html.escape(str(s))


def _fmt_t(t: float) -> str:
    return f"{int(t // 60)}:{t % 60:04.1f}"


def _oscurecer(img, mantener: np.ndarray | None = None, factor: float = 0.35):
    out = (img.astype(np.float32) * factor).astype(np.uint8)
    if mantener is not None:
        out[mantener] = img[mantener]
    return out


def _texto(img, txt, org, escala=0.55, color=(255, 255, 255), fondo=(20, 20, 20)):
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, escala, 1)
    x, y = int(org[0]), int(org[1])
    cv2.rectangle(img, (x - 3, y - th - 5), (x + tw + 3, y + 4), fondo, -1)
    cv2.putText(img, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, escala, color, 1, cv2.LINE_AA)


def _grosor(img) -> int:
    return max(1, img.shape[1] // 700)


def _zona_interes(d, w, h, frac: float = 0.42):
    """Recorte (x0, y0, x1, y1) alrededor de la jugada: el balón o el grupo más denso de jugadores."""
    centro = None
    if d.get("balon") is not None:
        centro = (d["balon"]["u"], d["balon"]["v"])
    else:
        pies = d["jug"].pies
        if len(pies):
            dens = [(np.linalg.norm(pies - p, axis=1) < 0.15 * w).sum() for p in pies]
            centro = tuple(pies[int(np.argmax(dens))])
    if centro is None:
        return None
    cw, ch = int(w * frac), int(h * frac)
    x0 = int(np.clip(centro[0] - cw / 2, 0, w - cw))
    y0 = int(np.clip(centro[1] - ch * 0.55, 0, h - ch))
    return x0, y0, x0 + cw, y0 + ch


def _con_zoom(img, roi):
    """Amplía la zona de interés a todo el ancho y deja el cuadro completo como miniatura."""
    if roi is None:
        return img
    h, w = img.shape[:2]
    x0, y0, x1, y1 = roi
    grande = cv2.resize(img[y0:y1, x0:x1], (w, h), interpolation=cv2.INTER_LINEAR)
    mw = int(w * 0.24)
    mh = int(h * mw / w)
    mini = cv2.resize(img, (mw, mh), interpolation=cv2.INTER_AREA)
    s = mw / w
    cv2.rectangle(mini, (int(x0 * s), int(y0 * s)), (int(x1 * s), int(y1 * s)), (0, 230, 255), 2)
    grande[10:10 + mh + 4, 10:10 + mw + 4] = (255, 255, 255)
    grande[12:12 + mh, 12:12 + mw] = mini
    _texto(grande, f"zoom x{w / (x1 - x0):.1f}", (20, 12 + mh + 26), 0.6)
    return grande


# ====================================================================== etapas
def construir_etapas(pipe, res, nombres: dict | None = None) -> list[Etapa]:
    d = pipe.diag
    if d is None:
        raise RuntimeError("El pipeline no guardó el diagnóstico de este cuadro")
    nombres = nombres or {"A": "Equipo A", "B": "Equipo B"}
    frame = d["frame"]
    h, w = frame.shape[:2]
    g = _grosor(frame)
    cancha = pipe.cancha
    estado = d["estado"]
    H = estado.H_img2cancha
    etapas: list[Etapa] = []
    # En un plano general los jugadores miden 20-40 px: las etapas de detalle se muestran ampliadas
    altos = d["jug"].cajas[:, 3] - d["jug"].cajas[:, 1] if len(d["jug"]) else np.array([h])
    roi = _zona_interes(d, w, h) if np.median(altos) < 0.055 * h else None
    nota_zoom = " La imagen amplía la zona de la jugada; el recuadro amarillo de la miniatura indica dónde." if roi else ""

    # 1. cuadro original ----------------------------------------------------------
    etapas.append(Etapa(
        "original", "Cuadro", "El cuadro de video",
        "Es lo que entra a la app: un cuadro del archivo, de la capturadora o de la pantalla. "
        "Todo lo que sigue se calcula sobre esta imagen, cuadro tras cuadro.",
        "El video se reescala al ancho de análisis. Si la cámara tiene lente gran angular, "
        "antes se corrige la distorsión para que las líneas rectas vuelvan a ser rectas.",
        imagen=frame.copy(),
        datos=[("Instante", _fmt_t(d["t"])), ("Resolución de análisis", f"{w} × {h} px"),
               ("Cámara", "fija" if pipe.camara_fija else "de TV (se mueve)")]))

    # 2. césped y zona de juego ---------------------------------------------------
    cesped, campo = d["cesped"] > 0, d["campo"] > 0
    img = _oscurecer(frame, campo, 0.3)
    verde = img.copy()
    verde[cesped] = (0.55 * img[cesped] + 0.45 * np.array([60, 200, 90])).astype(np.uint8)
    img = verde
    cont, _ = cv2.findContours(d["campo"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, cont, -1, (0, 230, 255), 2 * g, cv2.LINE_AA)
    graf = d.get("graficos")
    n_graf = 0
    if graf is not None and graf.any():
        n_graf = 1
        rojo = graf > 0
        img[rojo] = (0.4 * img[rojo] + 0.6 * np.array([40, 40, 230])).astype(np.uint8)
        ys, xs = np.where(rojo)
        _texto(img, "grafico de TV: ignorado", (xs.min() + 4, max(18, ys.min() - 6)), 0.5)
    frac = float(cesped.mean())
    como = ("Los píxeles verdes (tono, saturación y brillo) forman el césped, en verde. La región que "
            "encierran, con borde amarillo, es la zona de juego. Si más del 25% del cuadro es césped "
            "es un plano general útil; si no, es un primer plano o una repetición y se ignora.")
    if not pipe.camara_fija:
        como += (" En TV, lo que no cambia en pantalla mientras la cámara se mueve es un gráfico "
                 "(marcador, logo), en rojo, y no se usa para detectar.")
    etapas.append(Etapa(
        "cesped", "Césped", "Césped y zona de juego",
        "Separa la cancha de todo lo demás: tribuna, público, bancos, cielo y gráficos. Nada que "
        "esté fuera de esta zona puede ser un jugador ni el balón.", como, imagen=img,
        datos=[("Césped en el cuadro", f"{100 * frac:.0f}%"),
               ("Tipo de plano", "general: se analiza" if frac > 0.25 else "primer plano: se salta")]
        + ([("Gráficos de TV", "detectados y excluidos")] if n_graf else [])))

    # 3. líneas -------------------------------------------------------------------
    cajas = d["jug"].cajas
    excl = mascara_cajas(frame.shape, cajas, holgura=0.25)
    lineas = mascara_lineas(frame, d["campo"], excl)
    img = _oscurecer(frame, None, 0.35)
    m = lineas > 0
    img[cv2.dilate(lineas, np.ones((2, 2), np.uint8)) > 0] = (255, 255, 120)
    for x1, y1, x2, y2 in cajas.astype(int):
        cv2.rectangle(img, (x1, y1), (x2, y2), (90, 90, 90), 1)
    etapas.append(Etapa(
        "lineas", "Líneas", "Líneas de la cancha",
        "Encuentra las líneas blancas pintadas: bandas, áreas, círculo central. Son la referencia "
        "para saber qué parte de la cancha se está viendo y para verificar la calibración.",
        "Un filtro morfológico (top-hat) resalta lo claro y delgado sobre el césped. Las zonas "
        "tapadas por jugadores (rectángulos grises) no cuentan, porque camisetas y medias blancas "
        "se confundirían con líneas.", imagen=img,
        datos=[("Píxeles de línea", f"{int(m.sum()):,}".replace(",", ".")),
               ("Jugadores excluidos", str(len(cajas)))]))

    # 4. calibración ----------------------------------------------------------------
    img = frame.copy()
    datos = [("Cancha", f"{cancha.largo:g} × {cancha.ancho:g} m")]
    if H is not None:
        proy = mascara_lineas_proyectadas(np.linalg.inv(H), frame.shape, cancha, grosor=2 * g)
        img[proy > 0] = (40, 40, 255)
        estados = {"clave": "calibrada a mano en este cuadro", "propagada": "siguiendo la cámara",
                   "refinada": "ajustada con las líneas", "reanclada": "re-anclada tras un corte",
                   "fija": "cámara fija"}
        datos.append(("Estado", estados.get(estado.fuente, estado.fuente)))
        H_c2i = np.linalg.inv(H)

        def ppm(v):
            uv = np.array([[w / 2, v]])
            xy = aplicar_h(H, uv)[0]
            a, b = aplicar_h(H_c2i, [[xy[0] - 0.5, xy[1]], [xy[0] + 0.5, xy[1]]])
            return float(np.linalg.norm(b - a))
        datos.append(("1 metro mide", f"{ppm(0.92 * h):.0f} px cerca · {ppm(0.35 * h):.0f} px lejos"))
    flujo = d.get("flujo")
    if flujo is not None and not pipe.camara_fija:
        p0, p1, inl = flujo
        inl = np.ones(len(p0), bool) if inl is None else inl
        for (a, b), ok in zip(zip(p0, p1), inl):
            fin = a + (b - a) * 6        # flechas exageradas x6 para que se vean
            cv2.arrowedLine(img, tuple(np.round(a).astype(int)), tuple(np.round(fin).astype(int)),
                            (80, 230, 80) if ok else (60, 60, 230), 1, cv2.LINE_AA, tipLength=0.3)
        datos.append(("Puntos del césped seguidos", f"{int(inl.sum())} usados · {int((~inl).sum())} descartados"))
        datos.append(("Calce con las líneas", f"{estado.calidad:.2f} (1 = perfecto)"))
        como = ("Se calibra a mano un cuadro (4 o más puntos). Después, cada flecha verde es un punto del "
                "césped seguido al cuadro siguiente (flujo óptico, exagerado ×6): su movimiento común es "
                "el paneo y el zoom, y con eso se actualiza la calibración. Las flechas rojas no siguen ese "
                "movimiento (jugadores, ruido) y se descartan. Este seguimiento acumula un error pequeño "
                "con el tiempo (en las pruebas, alrededor de 1 m en tomas de hasta 30 s), por eso las líneas rojas pueden "
                "no calzar del todo; un modelo de puntos clave de la cancha lo corregiría en cada cuadro.")
    else:
        como = ("Una homografía calculada con los puntos marcados a mano y ajustada después contra las "
                "líneas blancas. Como la cámara no se mueve, la misma sirve para todo el video.")
    etapas.append(Etapa(
        "camara", "Calibración", "Calibración: de la imagen a la cancha",
        "Relaciona cada píxel del suelo con su punto en metros. En rojo, la cancha teórica dibujada "
        "con esa relación: si calza con las líneas reales, la calibración está bien.", como,
        imagen=img, datos=datos))

    # 5. personas -------------------------------------------------------------------
    crudas = d["crudas"]
    personas = crudas.jugadores() if hasattr(crudas, "jugadores") else d["jug"]
    kept = d["jug"].cajas
    img = frame.copy()
    n_desc = 0
    for (x1, y1, x2, y2), c in zip(personas.cajas.astype(int), personas.confianza):
        dentro = any(np.allclose([x1, y1, x2, y2], k, atol=1.5) for k in kept)
        col = (255, 255, 255) if dentro else (120, 120, 120)
        n_desc += not dentro
        cv2.rectangle(img, (x1, y1), (x2, y2), col, g, cv2.LINE_AA)
        if not dentro:
            cv2.line(img, (x1, y1), (x2, y2), col, g)
            cv2.line(img, (x1, y2), (x2, y1), col, g)
        if (y2 - y1) > 28:
            cv2.putText(img, f"{c:.2f}", (x1, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.35 + 0.1 * g, col, 1, cv2.LINE_AA)
    nombre_modelo = "YOLO" if hasattr(pipe.detector, "modelo") else "detector clásico (manchas sobre el césped)"
    etapas.append(Etapa(
        "personas", "Personas", "Detección de personas",
        "Una red neuronal marca cada persona visible con una caja y una confianza (0 a 1). Las que "
        "tienen los pies fuera de la zona de juego (público, cuerpo técnico) se descartan, con una X.",
        f"Modelo {nombre_modelo} a {getattr(pipe.detector, 'imgsz', w)} px, entrenado con el dataset "
        "COCO. Un modelo afinado para fútbol distinguiría además arqueros, árbitros y el balón." + nota_zoom,
        imagen=_con_zoom(img, roi),
        datos=[("Personas detectadas", str(len(personas))), ("Descartadas fuera de la cancha", str(n_desc)),
               ("Confianza mediana", f"{np.median(personas.confianza):.2f}" if len(personas) else "–")]))

    # 6. equipos --------------------------------------------------------------------
    img = frame.copy()
    torsos = {"A": [], "B": [], "otro": []}
    for (x1, y1, x2, y2), eq in zip(kept.astype(int), d["etiquetas"]):
        col = BGR.get(eq, BGR[None])
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2 * g, cv2.LINE_AA)
        al, an = y2 - y1, x2 - x1
        ty1, ty2 = y1 + int(0.15 * al), y1 + int(0.5 * al)
        tx1, tx2 = x1 + int(0.2 * an), x2 - int(0.2 * an)
        cv2.rectangle(img, (tx1, ty1), (tx2, ty2), (255, 255, 255), 1)
        rec = frame[max(0, y1):y2, max(0, x1):x2]
        if rec.size and eq in torsos:
            torsos[eq].append(cv2.resize(rec, (40, 64), interpolation=cv2.INTER_AREA))
    alto_tira = 92
    tira = np.full((alto_tira * 3, w, 3), 24, np.uint8)
    etiquetas_fila = {"A": nombres.get("A", "A"), "B": nombres.get("B", "B"), "otro": "otro (arbitro, arquero)"}
    for fila, eq in enumerate(("A", "B", "otro")):
        y0 = fila * alto_tira
        cv2.rectangle(tira, (0, y0), (8, y0 + alto_tira - 4), BGR[eq], -1)
        cv2.putText(tira, str(etiquetas_fila[eq])[:26], (16, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1, cv2.LINE_AA)
        x = 16
        for t_img in torsos[eq][:40]:
            if x + 44 > w:
                break
            tira[y0 + 26:y0 + 90, x:x + 40] = t_img
            x += 44
    img = np.vstack([_con_zoom(img, roi), tira])
    datos = []
    hexes = pipe.equipos.colores_hex() if pipe.equipos.listo else {}
    for eq in ("A", "B"):
        datos.append((f"{nombres.get(eq, eq)}", f"{len(torsos[eq])} en cuadro · camiseta {hexes.get(eq, '¿?')}"))
    datos.append(("Otros", f"{len(torsos['otro'])} en cuadro"))
    etapas.append(Etapa(
        "equipos", "Equipos", "Equipo por color de camiseta",
        "Asigna cada jugador a un equipo según el color de su camiseta, sin que nadie le diga cuál es cuál. "
        "Abajo, los jugadores de este cuadro agrupados por equipo.",
        "Se toma el torso (rectángulo blanco), se quitan los píxeles de césped y se resume el color en "
        "espacio Lab. Un k-means con las primeras ~400 muestras encuentra los dos colores dominantes; "
        "lo que queda lejos de ambos se marca como otro. Cada pista vota en el tiempo, así un cuadro "
        "dudoso no le cambia el equipo a un jugador.", imagen=img, datos=datos))

    # 7. balón ----------------------------------------------------------------------
    img = _oscurecer(frame, None, 0.55)
    todos = d["bal_todos"]
    elegidos = d["bal"]
    for (x1, y1, x2, y2) in todos.cajas.astype(int):
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        en_juego = any(np.allclose([x1, y1, x2, y2], k, atol=1.0) for k in elegidos.cajas)
        if en_juego:
            cv2.circle(img, (cx, cy), 7 * g, (0, 220, 255), g, cv2.LINE_AA)
        else:   # fuera de la cancha (tribuna, carteles): se descartan
            cv2.circle(img, (cx, cy), max(2, g), (150, 150, 150), -1, cv2.LINE_AA)
    datos = [("Manchas candidatas", f"{len(todos)} en el cuadro · {len(elegidos)} dentro de la cancha "
                                    f"(el resto, puntos grises, cae fuera: tribuna, carteles)")]
    if d["k_balon"] is not None and len(elegidos):
        x1, y1, x2, y2 = elegidos.cajas[d["k_balon"]].astype(int)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        cv2.circle(img, (cx, cy), 16 * g, (60, 255, 60), 2 * g, cv2.LINE_AA)
        _texto(img, "balon", (cx + 18 * g, cy - 12 * g), 0.55)
        img = _con_zoom(img, roi)
        r = 22
        rec = frame[max(0, cy - r):cy + r, max(0, cx - r):cx + r]
        if rec.size:
            z = cv2.resize(rec, (rec.shape[1] * 5, rec.shape[0] * 5), interpolation=cv2.INTER_NEAREST)
            zh, zw = z.shape[:2]
            y0, x0 = 12, w - zw - 12
            img[y0:y0 + zh, x0:x0 + zw] = z
            cv2.rectangle(img, (x0, y0), (x0 + zw, y0 + zh), (60, 255, 60), 2)
            _texto(img, "lupa x5", (x0 + 6, y0 + zh - 8), 0.45)
        b = d["balon"]
        datos.append(("Balón elegido", f"x {b['x']:.1f} m · y {b['y']:.1f} m · {max(x2 - x1, y2 - y1)} px"))
    else:
        img = _con_zoom(img, roi)
        datos.append(("Balón elegido", "no en este cuadro (tapado, en el aire o sin candidato coherente)"))
    etapas.append(Etapa(
        "balon", "Balón", "Balón",
        "Busca el balón. Es lo más difícil: en un plano general mide pocos píxeles, va por el aire "
        "y se esconde entre los jugadores. Círculos amarillos: candidatos; verde: el elegido.",
        "Candidatos: manchas claras, pequeñas y redondas, rodeadas de césped, que no están sobre una "
        "línea pintada ni dentro de un jugador, más lo que detecte la red. Un seguidor elige el que "
        "es coherente con la trayectoria anterior; si lo pierde, solo adopta uno nuevo cuando se "
        "repite varios cuadros seguidos." + nota_zoom, imagen=img, datos=datos))

    # 8. de píxeles a metros ----------------------------------------------------------
    img = frame.copy()
    svg = SvgCancha(cancha, ancho_px=640)
    partes = []
    huella = estado.huella(frame.shape) if H is not None else None
    if huella is not None:
        pts = " ".join(f"{u:.1f},{v:.1f}" for u, v in (svg.p(x, y) for x, y in huella))
        partes.append(f'<polygon points="{pts}" class="huella"/>')
    ejemplo = None
    for j in sorted(d["jugadores"], key=lambda j: -j["v"]):
        eq = j.get("equipo") if j.get("equipo") in ("A", "B") else "otro"
        col = BGR[eq]
        u, v = int(j["u"]), int(j["v"])
        cv2.circle(img, (u, v), 4 * g, col, -1, cv2.LINE_AA)
        cv2.circle(img, (u, v), 4 * g, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(img, str(j["id"]), (u + 5 * g, v - 4 * g), cv2.FONT_HERSHEY_SIMPLEX, 0.35 + 0.1 * g, (255, 255, 255), 1, cv2.LINE_AA)
        px, py = svg.p(j["x"], j["y"])
        cls = {"A": "d-A", "B": "d-B"}.get(eq, "d-O")
        partes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" class="{cls}"><title>pista {j["id"]}: '
                      f'({j["u"]:.0f}, {j["v"]:.0f}) px → ({j["x"]:.1f}, {j["y"]:.1f}) m</title></circle>'
                      f'<text x="{px:.1f}" y="{py + 3:.1f}" class="lbl">{j["id"]}</text>')
        if ejemplo is None:
            ejemplo = j
    if d["balon"] is not None:
        bx, by = svg.p(d["balon"]["x"], d["balon"]["y"])
        partes.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" class="bal"/>')
    datos = [("Jugadores ubicados", str(len(d["jugadores"])))]
    if ejemplo is not None:
        datos.append((f"Ejemplo, pista {ejemplo['id']}",
                       f"píxel ({ejemplo['u']:.0f}, {ejemplo['v']:.0f}) → ({ejemplo['x']:.1f} m, {ejemplo['y']:.1f} m)"))
    etapas.append(Etapa(
        "metros", "A metros", "De píxeles a metros",
        "El punto de apoyo de cada jugador se convierte en una posición en metros sobre la cancha. "
        "Desde aquí ya no importa la cámara: distancias, velocidades y formaciones son reales.",
        "Se proyecta el pie (centro del borde inferior de la caja), no el centro del cuerpo, porque "
        "el pie está en el suelo, que es el plano que describe la calibración. El polígono claro del "
        "mapa es la parte de la cancha que la cámara ve en este cuadro." + nota_zoom,
        imagen=_con_zoom(img, roi), svg=svg.envolver("".join(partes), "Jugadores proyectados a la cancha"), datos=datos))

    # 9. seguimiento -------------------------------------------------------------------
    t_now = d["t"]
    activos = {j["id"]: j for j in d["jugadores"]}
    fuera = {f["id"]: f for f in d["fuera"]}
    hist = {}
    for r in pipe.registros:
        if r["id"] in activos and t_now - 3.0 <= r["t"] <= t_now:
            hist.setdefault(r["id"], []).append((r["t"], r["x"], r["y"]))
    partes = []
    if huella is not None:
        pts = " ".join(f"{u:.1f},{v:.1f}" for u, v in (svg.p(x, y) for x, y in huella))
        partes.append(f'<polygon points="{pts}" class="huella"/>')
    for pid, serie in hist.items():
        eq = activos[pid].get("equipo") if activos[pid].get("equipo") in ("A", "B") else "O"
        serie.sort()
        if len(serie) >= 2:
            xs = np.array([s[1] for s in serie])
            ys = np.array([s[2] for s in serie])
            k = max(1, len(xs) // 25)
            pts = " ".join(f"{u:.1f},{v:.1f}" for u, v in (svg.p(x, y) for x, y in zip(xs[::k], ys[::k])))
            partes.append(f'<polyline points="{pts}" class="tr-{eq}"/>')
    for pid, j in activos.items():
        eq = j.get("equipo") if j.get("equipo") in ("A", "B") else "O"
        px, py = svg.p(j["x"], j["y"])
        partes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" class="d-{eq}"/><text x="{px:.1f}" y="{py + 3:.1f}" class="lbl">{pid}</text>')
    for pid, f in fuera.items():
        eq = f.get("equipo") if f.get("equipo") in ("A", "B") else "O"
        px, py = svg.p(f["x"], f["y"])
        partes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="7" class="hueco-{eq}"><title>pista {pid}: fuera de cuadro, posición estimada</title></circle>')
    vels = []
    for serie in hist.values():
        if len(serie) >= 6:
            (t0, x0, y0), (t1, x1, y1) = serie[-6], serie[-1]
            if t1 > t0:
                vels.append(np.hypot(x1 - x0, y1 - y0) / (t1 - t0))
    etapas.append(Etapa(
        "seguimiento", "Seguimiento", "Seguimiento en el tiempo",
        "Une las detecciones de cuadro en cuadro para que cada jugador conserve su número (pista). "
        "Las líneas son el recorrido de los últimos 3 segundos.",
        "Se sigue en metros, no en píxeles, porque la cámara se mueve. Un filtro de Kalman predice "
        "dónde estará cada jugador; se asignan primero las detecciones seguras y luego las dudosas, "
        "sin permitir cambios de equipo. Quien sale de cuadro se mantiene unos segundos con su "
        "posición estimada (círculos huecos) para recuperar su número al volver.",
        svg=svg.envolver("".join(partes), "Recorridos de los últimos 3 segundos"),
        datos=[("Pistas en cuadro", str(len(activos))), ("Fuera de cuadro (estimadas)", str(len(fuera))),
               ("Más rápido ahora", f"{3.6 * max(vels):.0f} km/h" if vels else "–")]))

    # 10. lectura táctica ------------------------------------------------------------
    etapas.append(_etapa_tactica(pipe, d, svg, huella, nombres))
    return etapas


def _etapa_tactica(pipe, d, svg: SvgCancha, huella, nombres) -> Etapa:
    cancha = pipe.cancha
    try:
        df = pipe.tabla()
        direccion = direccion_ataque(df, ParametrosMetricas(largo=cancha.largo, ancho=cancha.ancho)) if len(df) else {"A": 1, "B": -1}
    except Exception:  # noqa: BLE001
        direccion = {"A": 1, "B": -1}
    jug = [j for j in d["jugadores"] if j.get("equipo") in ("A", "B")]
    partes, datos = [], []
    if huella is not None:
        pts = " ".join(f"{u:.1f},{v:.1f}" for u, v in (svg.p(x, y) for x, y in huella))
        partes.append(f'<polygon points="{pts}" class="huella"/>')
    for eq in ("A", "B"):
        P = np.array([[j["x"], j["y"]] for j in jug if j["equipo"] == eq])
        if len(P) >= 3:
            try:
                hull = ConvexHull(P)
                pts = " ".join(f"{u:.1f},{v:.1f}" for u, v in (svg.p(*P[i]) for i in hull.vertices))
                partes.append(f'<polygon points="{pts}" class="bloque-{eq}"/>')
            except Exception:  # noqa: BLE001
                pass
        if len(P) >= 2:
            dd = direccion.get(eq, 1)
            propio = -dd * cancha.L2
            orden = np.argsort(np.abs(P[:, 0] - propio))
            x_linea = P[orden[min(1, len(orden) - 1)], 0]
            u0, v0 = svg.p(x_linea, cancha.W2)
            u1, v1 = svg.p(x_linea, -cancha.W2)
            partes.append(f'<line x1="{u0:.1f}" y1="{v0:.1f}" x2="{u1:.1f}" y2="{v1:.1f}" class="linea-{eq}"/>')
            altura = abs(x_linea - propio)
            ancho = P[:, 1].max() - P[:, 1].min()
            datos.append((nombres.get(eq, eq), f"{len(P)} visibles · ancho {ancho:.0f} m · última línea a {altura:.0f} m de su arco"))
    b = d["balon"]
    bx = by = None
    if b is not None:
        bx, by = svg.p(b["x"], b["y"])
        if jug:
            dist = [np.hypot(j["x"] - b["x"], j["y"] - b["y"]) for j in jug]
            k = int(np.argmin(dist))
            if dist[k] <= 2.0:
                p = jug[k]
                px, py = svg.p(p["x"], p["y"])
                r5 = 5.0 * svg.esc
                rivales = sum(1 for j in jug if j["equipo"] != p["equipo"] and np.hypot(j["x"] - p["x"], j["y"] - p["y"]) <= 5)
                partes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="{r5:.1f}" class="presion"/>')
                partes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="11" class="poseedor"/>')
                datos.insert(0, ("Tiene el balón", f"{nombres.get(p['equipo'], p['equipo'])}, pista {p['id']} · "
                                                    f"{rivales} rival{'es' if rivales != 1 else ''} a ≤5 m"))
            else:
                datos.insert(0, ("Tiene el balón", f"nadie a ≤2 m (balón en disputa o en vuelo, a {dist[k]:.1f} m del más cercano)"))
    else:
        datos.insert(0, ("Tiene el balón", "balón no ubicado en este cuadro"))
    for j in d["jugadores"]:
        eq = j.get("equipo") if j.get("equipo") in ("A", "B") else "O"
        px, py = svg.p(j["x"], j["y"])
        partes.append(f'<circle cx="{px:.1f}" cy="{py:.1f}" r="6" class="d-{eq}"/>')
    if bx is not None:
        partes.append(f'<circle cx="{bx:.1f}" cy="{by:.1f}" r="4.5" class="bal"/>')
    ataca = [nombres.get(eq, eq) for eq, dd in direccion.items() if dd == 1]
    if ataca:
        datos.append(("Ataca hacia la derecha", ataca[0]))
    return Etapa(
        "tactica", "Táctica", "Lectura táctica del momento",
        "Con todos ubicados en metros se calculan las métricas del instante: quién tiene el balón, "
        "cuánta presión recibe, el bloque de cada equipo y la altura de su última línea.",
        "Poseedor: jugador a ≤2 m del balón (anillo). Presión: rivales dentro del círculo de 5 m. "
        "Bloque: la figura que envuelve a los jugadores visibles de cada equipo. Línea defensiva "
        "(punteada): el penúltimo jugador más cercano a su propio arco. Sumando cuadros salen la "
        "posesión, los pases, los mapas de calor y el resto del reporte.",
        svg=svg.envolver("".join(partes), "Lectura táctica del momento"), datos=datos)


# ====================================================================== correr
def capturar(pipe, fuente, tiempos: list[float], nombres: dict | None = None, progreso=None,
             detener=None) -> dict[float, list[Etapa]]:
    """Procesa la fuente con el pipeline y arma las etapas en los instantes pedidos."""
    pendientes = sorted(tiempos)
    salida: dict[float, list[Etapa]] = {}
    dt = 1.0 / (fuente.fps or 25)
    n = 0
    for indice, t, frame in fuente:
        if detener is not None and detener():
            break
        objetivo = pendientes[0] if pendientes else None
        if objetivo is None:
            break
        pipe.guardar_diag = t >= objetivo - dt / 2
        res = pipe.procesar(indice, t, frame)
        n += 1
        if progreso:
            progreso(n, t, objetivo)
        if pipe.guardar_diag:
            if not pipe.equipos.listo:
                pipe.equipos.forzar_ajuste()
            salida[objetivo] = construir_etapas(pipe, res, nombres)
            pendientes.pop(0)
            pipe.guardar_diag = False
    return salida


def guardar_imagenes(etapas: list[Etapa], carpeta: Path, prefijo: str, ancho_max: int = 1600) -> list[dict]:
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    out = []
    for n, e in enumerate(etapas, 1):
        img_rel = None
        if e.imagen is not None:
            im = e.imagen
            if im.shape[1] > ancho_max:
                s = ancho_max / im.shape[1]
                im = cv2.resize(im, (ancho_max, int(round(im.shape[0] * s))), interpolation=cv2.INTER_AREA)
            nombre = f"{prefijo}_{n:02d}_{e.clave}.jpg"
            cv2.imwrite(str(carpeta / nombre), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
            img_rel = f"{carpeta.name}/{nombre}"
        out.append({"clave": e.clave, "corto": e.corto, "titulo": e.titulo, "que": e.que, "como": e.como,
                    "img": img_rel, "svg": e.svg, "datos": [[str(a), str(b)] for a, b in e.datos]})
    return out


def explicar_video(video: str, calibracion, cfg, tiempos: list[float], carpeta: Path, prefijo: str,
                   nombres: dict | None = None, progreso=None, detener=None) -> list[dict]:
    """Corre el pipeline sobre `video` y devuelve los momentos con sus etapas (imágenes en carpeta)."""
    from .fuentes import FuenteArchivo
    from .pipeline import Pipeline
    pipe = Pipeline(calibracion, cfg)
    # Con cámara fija basta calentar 3 s antes (colores de equipo, seguimiento). Con cámara de TV
    # hay que empezar donde se calibró a mano y seguir el movimiento desde ahí.
    claves = [max(0, int(k.get("cuadro") or 0)) for k in calibracion.claves] or [0]   # en vivo: -1, -2…
    f = FuenteArchivo(video)
    fps = f.fps
    f.cerrar()
    if pipe.camara_fija:
        inicio = max(0.0, min(tiempos) - 3.0)
    else:
        previas = [c for c in claves if c / fps <= min(tiempos)]
        inicio = (max(previas) if previas else 0) / fps
    fuente = FuenteArchivo(video, inicio, max(tiempos) + 0.5)
    por_t = capturar(pipe, fuente, tiempos, nombres, progreso, detener)
    fuente.cerrar()
    momentos = []
    for n, t in enumerate(sorted(por_t)):
        momentos.append({"t": t, "etapas": guardar_imagenes(por_t[t], carpeta, f"{prefijo}_{n}")})
    return momentos


# ====================================================================== página
def generar_pagina(ruta: Path, clips: list[dict], titulo: str = "Detección paso a paso",
                   intro: str = "", artefacto: bool = False, inicio: str | None = None) -> Path:
    """clips: [{"nombre", "descripcion", "nombres": {"A","B"}, "video": ruta relativa o None,
               "momentos": [{"etiqueta", "t", "etapas": [dict de guardar_imagenes]}]}]"""
    datos = json.dumps(clips, ensure_ascii=False).replace("</", "<\\/")
    a1, a2 = SERIE["A"]
    b1, b2 = SERIE["B"]
    cabeza = "" if artefacto else '<!doctype html><html lang="es"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
    meta = '<meta name="claude:viewport-fit-sides" content="cover">' if artefacto else ""
    cuerpo_ini = "" if artefacto else "</head><body>"
    cuerpo_fin = "" if artefacto else "</body></html>"
    doc = f"""{cabeza}<title>{_e(titulo)}</title>{meta}
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@600;700&family=Barlow:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
/* Una barra de etapas numeradas (el orden real del proceso) sobre un visor grande con su explicación al lado */
:root{{--bg:#f2f4f0;--surface:#fbfcfa;--ink:#111612;--ink2:#465047;--muted:#7d877e;--line:#dbe0d8;
--accent:#14774a;--accent-ink:#ffffff;--a:{a1};--b:{b1};--o:#b8960c;--pitch:#e6ede4;--pitchline:#aebbab;
--f-display:"Barlow Condensed","Arial Narrow",system-ui,sans-serif;--f-body:"Barlow",system-ui,-apple-system,"Segoe UI",sans-serif;
--f-mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#0e1310;--surface:#151b17;--ink:#edf2eb;--ink2:#b5bfb4;
--muted:#7c877c;--line:#263027;--accent:#45c286;--accent-ink:#06150d;--a:{a2};--b:{b2};--o:#e0bf3a;--pitch:#18211b;--pitchline:#3b4b3e;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#0e1310;--surface:#151b17;--ink:#edf2eb;--ink2:#b5bfb4;--muted:#7c877c;--line:#263027;
--accent:#45c286;--accent-ink:#06150d;--a:{a2};--b:{b2};--o:#e0bf3a;--pitch:#18211b;--pitchline:#3b4b3e;color-scheme:dark}}
*{{box-sizing:border-box}}[hidden]{{display:none!important}}
body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 var(--f-body)}}
main{{max-width:1320px;margin:0 auto;padding-block:28px 56px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}}
h1{{font:700 clamp(30px,5vw,46px)/1.02 var(--f-display);letter-spacing:.005em;margin:0;text-wrap:balance;text-transform:uppercase}}
.intro{{color:var(--ink2);max-width:68ch;margin:10px 0 0}}
.volver{{display:inline-block;margin-bottom:14px;color:var(--accent);font:600 15px/1 var(--f-body);text-decoration:none}}
.controles{{display:flex;flex-wrap:wrap;gap:10px 22px;align-items:center;margin:22px 0 14px}}
.grupo{{display:flex;flex-wrap:wrap;gap:6px;align-items:center}}
.grupo .et{{font:600 12px/1 var(--f-body);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin-right:4px}}
.seg{{font:500 15px/1 var(--f-body);color:var(--ink2);background:var(--surface);border:1px solid var(--line);border-radius:999px;padding:8px 14px;cursor:pointer}}
.seg[aria-pressed="true"]{{background:var(--ink);color:var(--bg);border-color:var(--ink)}}
.seg:focus-visible,.paso:focus-visible,.nav:focus-visible,.visor img:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
.desc{{color:var(--ink2);font-size:15px;margin:0 0 14px;max-width:90ch}}
.riel{{display:flex;gap:0;overflow-x:auto;border-top:1px solid var(--line);border-bottom:1px solid var(--line);scrollbar-width:thin}}
.paso{{flex:1 0 auto;min-width:96px;background:none;border:0;border-bottom:3px solid transparent;color:var(--muted);padding:10px 10px 9px;cursor:pointer;text-align:left;font:500 14px/1.2 var(--f-body)}}
.paso b{{display:block;font:700 22px/1 var(--f-display);color:inherit}}
.paso[aria-current="step"]{{color:var(--ink);border-bottom-color:var(--accent)}}
.paso.hecho{{color:var(--ink2)}}
.escena{{display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:22px;margin-top:20px;align-items:start}}
@media (max-width:980px){{.escena{{grid-template-columns:1fr}}}}
.visor{{display:grid;gap:12px;min-width:0}}
.visor.doble{{grid-template-columns:minmax(0,1.25fr) minmax(0,1fr);align-items:start}}
@media (max-width:760px){{.visor.doble{{grid-template-columns:1fr}}}}
.visor img{{width:100%;height:auto;display:block;border-radius:6px;background:#000;cursor:zoom-in;max-width:100%}}
.mapa{{background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:10px}}
.mapa svg{{width:100%;height:auto;display:block}}
.texto{{min-width:0}}
.contador{{font:500 13px/1 var(--f-mono);color:var(--muted);letter-spacing:.02em}}
.texto h2{{font:700 30px/1.05 var(--f-display);text-transform:uppercase;margin:6px 0 12px;text-wrap:balance}}
.texto h3{{font:600 12px/1 var(--f-body);letter-spacing:.08em;text-transform:uppercase;color:var(--muted);margin:16px 0 6px}}
.texto p{{margin:0;color:var(--ink2)}}
.texto p.que{{color:var(--ink);font-size:17px}}
dl{{margin:16px 0 0;display:grid;gap:0;border-top:1px solid var(--line)}}
dl div{{display:grid;grid-template-columns:minmax(0,1fr);padding:8px 0;border-bottom:1px solid var(--line)}}
dt{{font-size:13px;color:var(--muted)}}dd{{margin:2px 0 0;font:500 14px/1.4 var(--f-mono);color:var(--ink);overflow-wrap:anywhere}}
.navs{{display:flex;gap:8px;margin-top:18px}}
.nav{{font:600 15px/1 var(--f-body);border:1px solid var(--line);background:var(--surface);color:var(--ink);padding:10px 14px;border-radius:8px;cursor:pointer}}
.nav.sig{{background:var(--accent);border-color:var(--accent);color:var(--accent-ink)}}
.nav:disabled{{opacity:.4;cursor:default}}
.leyenda{{display:flex;flex-wrap:wrap;gap:6px 16px;font-size:13px;color:var(--ink2);margin-top:4px}}
.leyenda span{{display:inline-flex;align-items:center;gap:6px}}.sw{{width:11px;height:11px;border-radius:50%;display:inline-block}}
.final{{margin-top:44px;border-top:1px solid var(--line);padding-top:22px;display:grid;grid-template-columns:minmax(0,1fr) 360px;gap:22px}}
@media (max-width:980px){{.final{{grid-template-columns:1fr}}}}
.final h2{{font:700 26px/1.05 var(--f-display);text-transform:uppercase;margin:0 0 8px}}
.final p{{color:var(--ink2);margin:0}}
video{{width:100%;max-width:100%;aspect-ratio:16/9;background:#000;border-radius:6px;display:block}}
.pitch-bg{{fill:var(--pitch)}}.pitch-line{{fill:none;stroke:var(--pitchline);stroke-width:1.2}}
.huella{{fill:var(--ink);fill-opacity:.06;stroke:var(--ink2);stroke-dasharray:4 3;stroke-width:1}}
.d-A{{fill:var(--a);stroke:var(--surface);stroke-width:1.5}}.d-B{{fill:var(--b);stroke:var(--surface);stroke-width:1.5}}.d-O{{fill:var(--o);stroke:var(--surface);stroke-width:1.5}}
.hueco-A{{fill:none;stroke:var(--a);stroke-width:1.6}}.hueco-B{{fill:none;stroke:var(--b);stroke-width:1.6}}.hueco-O{{fill:none;stroke:var(--o);stroke-width:1.6}}
.tr-A{{fill:none;stroke:var(--a);stroke-width:2;stroke-opacity:.55;stroke-linecap:round;stroke-linejoin:round}}
.tr-B{{fill:none;stroke:var(--b);stroke-width:2;stroke-opacity:.55;stroke-linecap:round;stroke-linejoin:round}}
.tr-O{{fill:none;stroke:var(--o);stroke-width:2;stroke-opacity:.55}}
.lbl{{font:600 8px var(--f-body);fill:#fff;text-anchor:middle;pointer-events:none}}
.bal{{fill:var(--ink);stroke:var(--surface);stroke-width:1.5}}
.bloque-A{{fill:var(--a);fill-opacity:.13;stroke:var(--a);stroke-width:1.2}}.bloque-B{{fill:var(--b);fill-opacity:.13;stroke:var(--b);stroke-width:1.2}}
.linea-A{{stroke:var(--a);stroke-width:1.6;stroke-dasharray:5 4}}.linea-B{{stroke:var(--b);stroke-width:1.6;stroke-dasharray:5 4}}
.presion{{fill:var(--ink);fill-opacity:.06;stroke:var(--ink);stroke-width:1;stroke-dasharray:2 3}}
.poseedor{{fill:none;stroke:var(--ink);stroke-width:2}}
#lupa{{position:fixed;inset:0;background:rgba(0,0,0,.86);display:flex;align-items:flex-start;justify-content:center;overflow:auto;z-index:20;
padding:max(16px,env(safe-area-inset-top,0px)) max(16px,env(safe-area-inset-right,0px)) 16px max(16px,env(safe-area-inset-left,0px));cursor:zoom-out}}
#lupa img{{max-width:none;width:auto;height:auto;border-radius:4px}}
@media (prefers-reduced-motion:no-preference){{.visor img,.mapa{{animation:entra .25s ease-out}}@keyframes entra{{from{{opacity:.6}}to{{opacity:1}}}}}}
</style>{cuerpo_ini}
<main>
{f'<a class="volver" href="{_e(inicio)}">← Analista de fútbol</a>' if inicio else ""}
<h1>{_e(titulo)}</h1>
<p class="intro">{_e(intro)}</p>
<div class="controles">
  <div class="grupo" role="group" aria-label="Video" id="g-clip"><span class="et">Video</span></div>
  <div class="grupo" role="group" aria-label="Momento" id="g-mom"><span class="et">Momento</span></div>
</div>
<p class="desc" id="desc"></p>
<nav class="riel" id="riel" aria-label="Etapas de la detección"></nav>
<section class="escena" aria-live="polite">
  <div class="visor" id="visor"></div>
  <div class="texto">
    <div class="contador" id="contador"></div>
    <h2 id="tit"></h2>
    <p class="que" id="que"></p>
    <h3>Cómo lo hace</h3>
    <p id="como"></p>
    <dl id="datos"></dl>
    <div class="leyenda" id="leyenda"></div>
    <div class="navs"><button class="nav" id="ant">← Anterior</button><button class="nav sig" id="sig">Siguiente →</button></div>
  </div>
</section>
<section class="final" id="final" hidden>
  <video id="video" controls playsinline preload="metadata"></video>
  <div><h2>El resultado</h2><p>Todas las etapas, cuadro tras cuadro: así queda el video que entrega la app, con cada
  jugador identificado por equipo y número, el balón y el minimapa.</p></div>
</section>
</main>
<div id="lupa" hidden><img id="lupa-img" alt="Imagen ampliada"></div>
<script type="application/json" id="datos-json">{datos}</script>
<script>
(()=>{{
const C = JSON.parse(document.getElementById('datos-json').textContent);
const $ = s => document.querySelector(s);
const st = {{c:0, m:0, e:0}};
// Enlace directo a una etapa: pagina.html#e7 (o #c2e7 para el segundo video)
(()=>{{const m=/^#(?:c(\\d+))?e(\\d+)$/.exec(location.hash||'');
  if(m){{st.c=Math.min(C.length-1,Math.max(0,(+m[1]||1)-1));st.e=Math.max(0,+m[2]-1);}}}})();
function boton(txt, on, sel){{const b=document.createElement('button');b.className='seg';b.textContent=txt;b.setAttribute('aria-pressed',sel);b.onclick=on;return b;}}
function controles(){{
  const gc=$('#g-clip'), gm=$('#g-mom');
  gc.querySelectorAll('.seg').forEach(n=>n.remove()); gm.querySelectorAll('.seg').forEach(n=>n.remove());
  gc.hidden = C.length < 2;
  C.forEach((c,i)=>gc.appendChild(boton(c.nombre,()=>{{st.c=i;st.m=0;dibujar();}},i===st.c)));
  C[st.c].momentos.forEach((m,i)=>gm.appendChild(boton(m.etiqueta,()=>{{st.m=i;dibujar();}},i===st.m)));
  $('#desc').textContent = C[st.c].descripcion||'';
}}
function dibujar(){{
  controles();
  const clip=C[st.c], mom=clip.momentos[st.m], et=mom.etapas;
  st.e=Math.min(st.e, et.length-1);
  const e=et[st.e];
  const riel=$('#riel'); riel.innerHTML='';
  et.forEach((x,i)=>{{const b=document.createElement('button');b.className='paso'+(i<st.e?' hecho':'');
    b.innerHTML=`<b>${{i+1}}</b>${{x.corto}}`; if(i===st.e) b.setAttribute('aria-current','step');
    b.onclick=()=>{{st.e=i;dibujar();}}; riel.appendChild(b);}});
  const cur = riel.querySelector('[aria-current]'); if(cur) cur.scrollIntoView({{block:'nearest',inline:'nearest'}});
  const v=$('#visor'); v.innerHTML=''; v.className='visor'+(e.img&&e.svg?' doble':'');
  if(e.img){{const im=new Image(); im.src=e.img; im.alt=e.titulo; im.tabIndex=0; im.title='Clic para ver en tamaño real';
    im.onclick=()=>abrir(e.img); im.onkeydown=k=>{{if(k.key==='Enter')abrir(e.img);}}; v.appendChild(im);}}
  if(e.svg){{const d=document.createElement('div'); d.className='mapa'; d.innerHTML=e.svg; v.appendChild(d);}}
  $('#contador').textContent=`Etapa ${{st.e+1}} de ${{et.length}} · ${{mom.etiqueta}}`;
  $('#tit').textContent=e.titulo; $('#que').textContent=e.que; $('#como').textContent=e.como;
  $('#datos').innerHTML=e.datos.map(([a,b])=>`<div><dt>${{esc(a)}}</dt><dd>${{esc(b)}}</dd></div>`).join('');
  const n=clip.nombres||{{A:'Equipo A',B:'Equipo B'}};
  $('#leyenda').innerHTML = (e.svg||['personas','equipos','metros'].includes(e.clave)) ?
    `<span><i class="sw" style="background:var(--a)"></i>${{esc(n.A)}}</span><span><i class="sw" style="background:var(--b)"></i>${{esc(n.B)}}</span><span><i class="sw" style="background:var(--o)"></i>otro</span>${{e.svg?'<span><i class="sw" style="background:var(--ink)"></i>balón</span>':''}}` : '';
  $('#ant').disabled=st.e===0; $('#sig').disabled=st.e===et.length-1;
  const f=$('#final'); if(clip.video){{f.hidden=false; const vd=$('#video');
    if(vd.dataset.src!==clip.video){{vd.dataset.src=clip.video; vd.innerHTML='';
      [['webm','video/webm'],['mp4','video/mp4']].forEach(([ext,tipo])=>{{const s=document.createElement('source');
        s.src=clip.video.replace(/\\.(mp4|webm)$/,'.'+ext); s.type=tipo; vd.appendChild(s);}}); vd.load();}}}} else f.hidden=true;
}}
function esc(s){{return String(s).replace(/[&<>"]/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}})[c]);}}
function abrir(src){{$('#lupa-img').src=src; $('#lupa').hidden=false;}}
$('#lupa').onclick=()=>{{$('#lupa').hidden=true;}};
$('#ant').onclick=()=>{{if(st.e>0){{st.e--;dibujar();}}}};
$('#sig').onclick=()=>{{const n=C[st.c].momentos[st.m].etapas.length; if(st.e<n-1){{st.e++;dibujar();}}}};
document.addEventListener('keydown',k=>{{if(k.target.closest('input,textarea,select'))return;
  if(k.key==='ArrowRight')$('#sig').click(); if(k.key==='ArrowLeft')$('#ant').click(); if(k.key==='Escape')$('#lupa').hidden=true;}});
dibujar();
}})();
</script>{cuerpo_fin}"""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(doc, encoding="utf-8")
    return ruta
