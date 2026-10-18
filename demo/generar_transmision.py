"""Transmisión sintética a partir de tracking real (SkillCorner), con la verdad de cada cuadro.

Sirve para medir la app con exactitud: se sabe dónde está cada jugador, el balón y la cámara
en todo momento. Los jugadores son recortes de jugadores reales (ver `extraer_sprites.py`);
las posiciones y la cámara de TV salen del partido real Auckland FC – Newcastle Jets
(A-League 2024/25, SkillCorner Open Data, licencia MIT).

    # plano completo con cámara fija (1920x1080), el gol de N. Moreno
    python demo/generar_transmision.py --camara completa --tramos 50377-50657 -o demo/clips/completo
    # cámara de TV que sigue la jugada, dos goles separados por una repetición
    python demo/generar_transmision.py --camara tv --tramos 50377-50657,53770-53858 --marcadores 0-0,1-0 -o demo/clips/tv

Salida: video.mp4, calibracion.json (clics simulados con ruido), calibracion_por_tramo.json,
gt_jugadores.csv, gt_balon.csv, gt_camara.csv e info.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from analista.cancha import Cancha, aplicar_h, h_desde_puntos
from analista.datos import skillcorner as sc
from analista.vision.pipeline import ruta_ffmpeg

PPM = 40                       # resolución del césped (píxeles por metro): textura fina como en la TV
PPM_PUB = 8                    # resolución del entorno (pista, carteles, público)
X0, X1, Y0, Y1 = -130.0, 130.0, -70.0, 140.0     # extensión del entorno (m)


# ====================================================================== cámara
class Camara:
    """Cámara estenopeica sobre el suelo: posición fija, paneo, inclinación y focal."""

    def __init__(self, C, ancho, alto):
        self.C = np.asarray(C, float)
        self.w, self.h = ancho, alto

    def matrices(self, pan, tilt, f):
        fwd = np.array([np.sin(pan) * np.cos(tilt), np.cos(pan) * np.cos(tilt), -np.sin(tilt)])
        r = np.array([np.cos(pan), -np.sin(pan), 0.0])
        u = np.cross(r, fwd)
        R = np.vstack([r, -u, fwd])
        K = np.array([[f, 0, self.w / 2], [0, f, self.h / 2], [0, 0, 1.0]])
        return K, R

    def proyectar(self, params, P):
        K, R = self.matrices(*params)
        P = np.atleast_2d(np.asarray(P, float))
        if P.shape[1] == 2:
            P = np.hstack([P, np.zeros((len(P), 1))])
        Xc = (P - self.C) @ R.T
        uv = Xc @ K.T
        return uv[:, :2] / uv[:, 2:3]

    def homografia(self, params):
        """Suelo (x, y, 0) en metros -> píxel."""
        K, R = self.matrices(*params)
        M = np.column_stack([R[:, 0], R[:, 1], -R @ self.C])
        H = K @ M
        return H / H[2, 2]


def ajustar_tv(cam: Camara, esquinas: np.ndarray, x0=None):
    """Paneo, inclinación y focal que llevan las esquinas inferiores de la imagen a los puntos de
    SkillCorner (sus esquinas superiores vienen recortadas a 5 m de la banda y no sirven)."""
    bl, br = esquinas
    obj = np.array([[0, cam.h], [cam.w, cam.h]], float)

    def res(p):
        return (cam.proyectar(p, [bl, br]) - obj).ravel() / 10

    x0 = x0 if x0 is not None else [np.arctan2((bl[0] + br[0]) / 2 - cam.C[0], 40 - cam.C[1]), 0.3, 1800]
    r = least_squares(res, x0, bounds=([-1.5, 0.05, 300], [1.5, 1.2, 8000]))
    return r.x


class CamaraPlana:
    """Plano completo fijo: homografía del suelo con las esquinas de la cancha en posiciones dadas.
    La altura (jugadores, balón en el aire) se dibuja hacia arriba con la escala local de la cancha,
    como en un plano general alto: una cámara física que vea toda la cancha en 16:9 deja a los
    jugadores de 10 px, demasiado chicos para una prueba útil."""

    def __init__(self, cancha: Cancha, ancho, alto):
        L, W = cancha.L2, cancha.W2
        src = np.float32([[-L, W], [L, W], [L, -W], [-L, -W]])
        dst = np.float32([[0.21 * ancho, 0.17 * alto], [0.79 * ancho, 0.17 * alto], [0.99 * ancho, 0.93 * alto], [0.01 * ancho, 0.93 * alto]])
        self.H = cv2.getPerspectiveTransform(src, dst)
        self.w, self.h = ancho, alto

    def homografia(self, params=None):
        return self.H / self.H[2, 2]

    def proyectar(self, params, P):
        P = np.atleast_2d(np.asarray(P, float))
        z = P[:, 2] if P.shape[1] > 2 else np.zeros(len(P))
        pie = aplicar_h(self.H, P[:, :2])
        ppm = np.linalg.norm(aplicar_h(self.H, P[:, :2] + [0.5, 0]) - aplicar_h(self.H, P[:, :2] - [0.5, 0]), axis=1)
        return pie - np.column_stack([np.zeros(len(P)), z * ppm * 0.95])


# ====================================================================== escena
def _ruido(rng, h, w, sigma_px, amp):
    """Ruido gaussiano suavizado (float32) de una escala dada."""
    n = rng.normal(0, 1, (h, w)).astype(np.float32)
    if sigma_px > 0:
        n = cv2.GaussianBlur(n, (0, 0), sigma_px)
        n /= max(float(n.std()), 1e-6)
    return n * amp


def lienzo_entorno(cancha: Cancha, semilla=5) -> np.ndarray:
    """Pista, carteles y público, a baja resolución (se ve lejos y desenfocado)."""
    rng = np.random.default_rng(semilla)
    w, h = int((X1 - X0) * PPM_PUB), int((Y1 - Y0) * PPM_PUB)
    colores = np.array([[40, 40, 200], [220, 220, 220], [30, 30, 30], [200, 120, 40], [60, 160, 230],
                        [140, 50, 150], [30, 120, 30], [240, 200, 60]], np.uint8)
    pub = colores[rng.integers(0, len(colores), (h // 2 + 1, w // 2 + 1))]
    pub = cv2.resize(pub, (w, h), interpolation=cv2.INTER_NEAREST)[:h, :w]
    pub = (pub.astype(np.float32) * rng.uniform(0.45, 0.9, (h, w, 1)).astype(np.float32)).astype(np.uint8)
    pub = cv2.GaussianBlur(pub, (0, 0), 0.8)
    xs = X0 + (np.arange(w, dtype=np.float32) + 0.5) / PPM_PUB
    ys = Y1 - (np.arange(h, dtype=np.float32) + 0.5) / PPM_PUB
    XX, YY = np.meshgrid(xs, ys)
    pista = (np.abs(XX) <= cancha.L2 + 10) & (np.abs(YY) <= cancha.W2 + 9)
    pub[pista] = (rng.normal(0, 6, (int(pista.sum()), 3)) + [70, 80, 95]).clip(0, 255).astype(np.uint8)
    for k, x in enumerate(np.arange(-cancha.L2, cancha.L2, 12)):
        c = [(40, 40, 170), (200, 200, 200), (160, 90, 20)][k % 3]
        p0 = (int((x - X0) * PPM_PUB), int((Y1 - (cancha.W2 + 7.5)) * PPM_PUB))
        p1 = (int((x + 11.6 - X0) * PPM_PUB), int((Y1 - (cancha.W2 + 6.5)) * PPM_PUB))
        cv2.rectangle(pub, p0, p1, c, -1)
    return pub


def lienzo_cesped(cancha: Cancha, semilla=3) -> np.ndarray:
    """Césped con franjas de corte, manchas y textura fina, más las líneas (BGRA, alfa = césped)."""
    rng = np.random.default_rng(semilla)
    mx, my = cancha.L2 + 5, cancha.W2 + 4.5
    w, h = int(2 * mx * PPM), int(2 * my * PPM)
    xs = -mx + (np.arange(w, dtype=np.float32) + 0.5) / PPM
    franja = (((np.floor((xs + cancha.L2) / (cancha.largo / 20)) % 2) * 2 - 1) * 9).astype(np.float32)
    g = np.full((h, w), 118, np.float32) + franja[None, :]
    g += cv2.resize(_ruido(rng, h // 50 + 2, w // 50 + 2, 1.2, 9), (w, h), interpolation=cv2.INTER_CUBIC)   # manchas
    g += _ruido(rng, h, w, 3.0, 5)      # matas (~8 cm)
    g += _ruido(rng, h, w, 0.7, 7)      # textura fina
    img = np.dstack([0.38 * g, g, 0.48 * g, np.full_like(g, 255)]).clip(0, 255).astype(np.uint8)
    for pl in cancha.lineas(paso=0.2):
        pts = np.column_stack([(pl[:, 0] + mx) * PPM, (my - pl[:, 1]) * PPM]).round().astype(np.int32)
        cv2.polylines(img, [pts], False, (236, 240, 236, 255), max(2, int(0.12 * PPM)), cv2.LINE_AA)
    return img


def H_entorno():
    """Píxel del lienzo del entorno -> metros."""
    return np.array([[1 / PPM_PUB, 0, X0], [0, -1 / PPM_PUB, Y1], [0, 0, 1.0]])


def H_cesped(cancha: Cancha):
    mx, my = cancha.L2 + 5, cancha.W2 + 4.5
    return np.array([[1 / PPM, 0, -mx], [0, -1 / PPM, my], [0, 0, 1.0]])


def render_suelo(entorno, cesped, He, Hc, H_c2i, ancho, alto):
    img = cv2.warpPerspective(entorno, H_c2i @ He, (ancho, alto), flags=cv2.INTER_LINEAR, borderValue=(60, 60, 60))
    c = cv2.warpPerspective(cesped, H_c2i @ Hc, (ancho, alto), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = c[..., 3:4].astype(np.float32) / 255
    return (c[..., :3] * a + img * (1 - a)).astype(np.uint8)


class Sprites:
    """Jugadores recortados de video real, agrupados por color de camiseta."""

    def __init__(self, carpeta: Path):
        idx = json.loads((Path(carpeta) / "indice.json").read_text(encoding="utf-8"))
        self.por_kit: dict[str, list[np.ndarray]] = {}
        for d in idx:
            im = cv2.imread(str(Path(carpeta) / d["archivo"]), cv2.IMREAD_UNCHANGED)
            if im is None or im.shape[2] != 4:
                continue
            a = im[..., 3] > 0
            filas = a.sum(1)
            if filas.max() == 0 or im.shape[0] < 60:
                continue
            # camiseta nítida: descarta recortes oscuros (azul marino) que confundirían el color del equipo
            h = im.shape[0]
            torso = im[int(0.2 * h):int(0.45 * h)]
            m = torso[..., 3] > 0
            if d["kit"] in ("azul", "blanco") and m.sum() and np.median(cv2.cvtColor(torso[..., :3], cv2.COLOR_BGR2HSV)[..., 2][m]) < 120:
                continue
            self.por_kit.setdefault(d["kit"], []).append(im)

    def teñido(self, base: str, color_bgr) -> list[np.ndarray]:
        out = []
        for im in self.por_kit.get(base, []):
            t = im.copy()
            gris = cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
            camiseta = (cv2.cvtColor(im[..., :3], cv2.COLOR_BGR2HSV)[..., 1] < 90) & (gris > 0.55)
            col = (gris[..., None] * np.array(color_bgr, np.float32)).clip(0, 255).astype(np.uint8)
            t[..., :3][camiseta] = col[camiseta]
            out.append(t)
        return out


def pegar(img, spr, pie_u, pie_v, alto_px, espejo=False):
    if alto_px < 6:
        return None
    s = alto_px / spr.shape[0]
    w = max(2, int(round(spr.shape[1] * s)))
    h = max(3, int(round(alto_px)))
    sp = cv2.resize(spr, (w, h), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR)
    if espejo:
        sp = sp[:, ::-1]
    x0, y0 = int(round(pie_u - w / 2)), int(round(pie_v - h))
    X0_, Y0_ = max(0, x0), max(0, y0)
    X1_, Y1_ = min(img.shape[1], x0 + w), min(img.shape[0], y0 + h)
    if X1_ <= X0_ or Y1_ <= Y0_:
        return None
    rec = sp[Y0_ - y0:Y1_ - y0, X0_ - x0:X1_ - x0]
    a = rec[..., 3:4].astype(np.float32) / 255
    zona = img[Y0_:Y1_, X0_:X1_]
    zona[:] = (rec[..., :3] * a + zona * (1 - a)).astype(np.uint8)
    return (x0, y0, x0 + w, y0 + h)


def sombra(img, u, v, ancho_px):
    if ancho_px < 3:
        return
    over = img.copy()
    cv2.ellipse(over, (int(u), int(v)), (max(2, int(ancho_px * 0.55)), max(1, int(ancho_px * 0.16))), 0, 0, 360, (20, 40, 20), -1, cv2.LINE_AA)
    cv2.addWeighted(over, 0.45, img, 0.55, 0, img)


def marcador(img, nombres, goles, reloj, escala=1.0):
    s = escala
    x, y = int(28 * s), int(24 * s)
    ancho, alto = int(250 * s), int(34 * s)
    cv2.rectangle(img, (x, y), (x + ancho, y + alto), (25, 18, 12), -1)
    cv2.rectangle(img, (x, y), (x + int(8 * s), y + alto), (240, 0, 40), -1)
    txt = f"{nombres[0]} {goles[0]} - {goles[1]} {nombres[1]}"
    cv2.putText(img, txt, (x + int(16 * s), y + int(24 * s)), cv2.FONT_HERSHEY_DUPLEX, 0.62 * s, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.rectangle(img, (x + ancho, y), (x + ancho + int(74 * s), y + alto), (240, 240, 240), -1)
    cv2.putText(img, reloj, (x + ancho + int(8 * s), y + int(24 * s)), cv2.FONT_HERSHEY_DUPLEX, 0.62 * s, (25, 18, 12), 1, cv2.LINE_AA)


def tarjeta_repeticion(ancho, alto, k, n):
    img = np.zeros((alto, ancho, 3), np.uint8)
    g = np.linspace(30, 90, ancho).astype(np.uint8)
    img[:] = np.dstack([g * 0.4, g * 0.25, g])[0][None].astype(np.uint8)
    txt = "REPETICION"
    esc = ancho / 640
    (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_DUPLEX, 1.6 * esc, 2)
    x = int((ancho - tw) / 2 + (1 - k / max(n - 1, 1)) * 40 * esc)
    cv2.putText(img, txt, (x, (alto + th) // 2), cv2.FONT_HERSHEY_DUPLEX, 1.6 * esc, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def clic_simulado(H_c2i, cancha: Cancha, ancho, alto, ruido, rng, max_puntos=8):
    """Puntos de referencia visibles, marcados con un error de `ruido` píxeles (como una persona)."""
    pts = cancha.puntos_referencia()
    nombres = list(pts)
    xy = np.array([pts[n] for n in nombres])
    uv = aplicar_h(H_c2i, xy)
    m = (uv[:, 0] > 0.03 * ancho) & (uv[:, 0] < 0.97 * ancho) & (uv[:, 1] > 0.05 * alto) & (uv[:, 1] < 0.97 * alto)
    idx = np.where(m)[0]
    if len(idx) < 4:
        return None
    # repartidos: el más lejano al conjunto elegido en cada paso
    elegidos = [idx[np.argmax(uv[idx, 1])]]
    while len(elegidos) < min(max_puntos, len(idx)):
        d = np.min(np.linalg.norm(uv[idx][:, None] - uv[elegidos][None], axis=2), axis=1)
        elegidos.append(idx[int(np.argmax(d))])
    pares = [{"imagen": (uv[k] + rng.normal(0, ruido, 2)).round(1).tolist(), "cancha": list(map(float, xy[k])),
              "nombre": nombres[k]} for k in elegidos]
    return pares


# ====================================================================== principal
def esquinas_raw(ruta_jsonl: Path, cuadros: set[int]) -> dict[int, np.ndarray]:
    out = {}
    hasta = max(cuadros)
    with open(ruta_jsonl, encoding="utf-8") as f:
        for linea in f:
            if '"frame": ' not in linea[:20]:
                continue
            fr = int(linea[10:linea.index(",")])
            if fr > hasta:
                break
            if fr in cuadros:
                c = json.loads(linea)["image_corners_projection"]
                if c.get("x_bottom_left") is not None:
                    out[fr] = np.array([[c["x_bottom_left"], c["y_bottom_left"]], [c["x_bottom_right"], c["y_bottom_right"]]])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--partido", type=int, default=1886347)
    ap.add_argument("--tramos", default="50377-50657", help="cuadros de SkillCorner (10 Hz): a-b,c-d")
    ap.add_argument("--marcadores", default=None, help="marcador al inicio de cada tramo: 0-0,1-0")
    ap.add_argument("--camara", default="completa", choices=["completa", "tv"])
    ap.add_argument("--resolucion", default=None, help="1920x1080 (completa) o 1280x720 (tv)")
    ap.add_argument("--fps", type=float, default=25.0)
    ap.add_argument("--sprites", default="demo/sprites")
    ap.add_argument("--ruido-clic", type=float, default=2.0, help="error de los clics de calibración (px)")
    ap.add_argument("--semilla", type=int, default=7)
    ap.add_argument("-o", "--salida", required=True)
    a = ap.parse_args()
    rng = np.random.default_rng(a.semilla)
    salida = Path(a.salida)
    salida.mkdir(parents=True, exist_ok=True)
    ancho, alto = (int(v) for v in (a.resolucion or ("1920x1080" if a.camara == "completa" else "1280x720")).split("x"))

    d = sc.cargar(a.partido, progreso=False)
    info_p = d["info"]
    cancha = Cancha(info_p["cancha"]["largo"], info_p["cancha"]["ancho"])
    tr, ba = d["tracking"], d["balon"]
    tramos = [tuple(int(v) for v in t.split("-")) for t in a.tramos.split(",")]
    marc = [tuple(int(v) for v in m.split("-")) for m in a.marcadores.split(",")] if a.marcadores else [(0, 0)] * len(tramos)
    goles_sc = sc.goles(a.partido, info_p)
    abrev = [info_p["equipos"][k]["nombre"].split()[0][:3].upper() for k in "AB"]

    # Volteo de cada periodo (el esquema común hace que A ataque a +x; la cámara de SkillCorner está en crudo)
    voltea: dict[int, bool | None] = {}
    ruta_raw = sc._cache(str(a.partido), "tracking.jsonl")
    todos = set()
    for t0, t1 in tramos:
        todos |= set(range(t0, t1 + 1))
    crudo = {}
    with open(ruta_raw, encoding="utf-8") as f:
        for linea in f:
            fr = int(linea[10:linea.index(",")])
            if fr > max(todos):
                break
            if fr in todos and fr % 10 == 0:
                dd = json.loads(linea)
                if dd["player_data"]:
                    crudo[fr] = (dd["period"], dd["player_data"][0])
    for fr, (per, p0) in crudo.items():
        if voltea.get(per) is None:
            fila = tr[(tr["cuadro"] == fr) & (tr["id"] == p0["player_id"])]
            if len(fila):
                voltea[per] = bool(np.sign(fila["x"].iloc[0]) != np.sign(p0["x"]) and abs(p0["x"]) > 0.5)
    signo_tramo = []
    for t0, _ in tramos:
        per = int(tr.loc[tr["cuadro"] >= t0, "periodo"].iloc[0])
        signo_tramo.append(-1.0 if voltea.get(per) else 1.0)

    entorno, cesped = lienzo_entorno(cancha), lienzo_cesped(cancha)
    He, Hc = H_entorno(), H_cesped(cancha)
    sprites = Sprites(Path(a.sprites))
    kits = {"A": sprites.por_kit.get("azul", []), "B": sprites.por_kit.get("blanco", [])}
    arqueros = {"A": sprites.teñido("blanco", (40, 220, 245)), "B": sprites.teñido("blanco", (40, 90, 240))}
    jug_info = tr.drop_duplicates("id").set_index("id")[["equipo", "rol", "nombre", "numero"]].to_dict("index")
    asign = {}
    for pid, ji in jug_info.items():
        lista = arqueros[ji["equipo"]] if ji["rol"] == "arquero" else kits[ji["equipo"]]
        k = rng.choice(len(lista), size=min(3, len(lista)), replace=False)
        asign[pid] = [lista[i] for i in k]

    if a.camara == "completa":
        cam_fija, params_fijos = CamaraPlana(cancha, ancho, alto), None
    else:
        cam_tv = Camara([0.0, -cancha.W2 - 32.0, 21.0], ancho, alto)

    ff = ruta_ffmpeg()
    proc = subprocess.Popen([ff, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{ancho}x{alto}",
                             "-r", str(a.fps), "-i", "-", "-c:v", "libx264", "-preset", "medium", "-crf", "20",
                             "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(salida / "video.mp4")],
                            stdin=subprocess.PIPE)
    gt_j, gt_b, gt_c, claves, goles_out = [], [], [], [], []
    k_out = 0
    tramos_out = []
    for nt, ((t0, t1), signo) in enumerate(zip(tramos, signo_tramo)):
        if nt > 0:                                   # corte con repetición entre tramos
            n_card = int(1.2 * a.fps)
            for k in range(n_card):
                proc.stdin.write(tarjeta_repeticion(ancho, alto, k, n_card).tobytes())
                gt_c.append({"cuadro": k_out, "t": round(k_out / a.fps, 3), "tramo": -1})
                k_out += 1
        seg = tr[(tr["cuadro"] >= t0) & (tr["cuadro"] <= t1)]
        piv_x = seg.pivot_table(index="cuadro", columns="id", values="x")
        piv_y = seg.pivot_table(index="cuadro", columns="id", values="y")
        bseg = ba[(ba["cuadro"] >= t0) & (ba["cuadro"] <= t1)].set_index("cuadro")[["x", "y", "z"]]
        bseg = bseg.reindex(range(t0, t1 + 1)).interpolate(limit_direction="both")
        idx = np.arange(t0, t1 + 1)
        piv_x, piv_y = piv_x.reindex(idx).interpolate(limit_direction="both"), piv_y.reindex(idx).interpolate(limit_direction="both")
        # Cámara de TV por cuadro de SkillCorner, suavizada
        if a.camara == "tv":
            esq = esquinas_raw(ruta_raw, set(idx))
            params, prev = {}, None
            for fr in idx:
                if fr in esq:
                    prev = ajustar_tv(cam_tv, esq[fr] * signo, prev)
                    params[fr] = prev
            P = pd.DataFrame(params).T.reindex(idx).interpolate(limit_direction="both")
            P = P.rolling(9, center=True, min_periods=1).mean()
        n_out = int((t1 - t0) / 10.0 * a.fps) + 1
        ini_out = k_out
        dist = {pid: 0.0 for pid in piv_x.columns}
        prev_xy = {}
        pares_clave = None
        for k in range(n_out):
            s = t0 + k * 10.0 / a.fps
            i0 = int(np.floor(s)) - t0
            i1 = min(i0 + 1, len(idx) - 1)
            w = s - np.floor(s)
            if a.camara == "tv":
                pr = P.iloc[i0].values * (1 - w) + P.iloc[i1].values * w
                cam, prm = cam_tv, pr
            else:
                cam, prm = cam_fija, params_fijos
            H_c2i = cam.homografia(prm)
            img = render_suelo(entorno, cesped, He, Hc, H_c2i, ancho, alto)
            xs = piv_x.iloc[i0].values * (1 - w) + piv_x.iloc[i1].values * w
            ys = piv_y.iloc[i0].values * (1 - w) + piv_y.iloc[i1].values * w
            bx, by, bz = (bseg.iloc[i0].values * (1 - w) + bseg.iloc[i1].values * w)
            t_out = k_out / a.fps
            # jugadores de atrás hacia adelante
            orden = []
            for pid, x, y in zip(piv_x.columns, xs, ys):
                if np.isnan(x):
                    continue
                pie = cam.proyectar(prm, [[x, y, 0.0]])[0]
                cabeza = cam.proyectar(prm, [[x, y, 1.82]])[0]
                orden.append((pie[1], pid, x, y, pie, pie[1] - cabeza[1]))
            orden.sort()
            for v, pid, x, y, pie, alto_px in orden:
                if pid in prev_xy:
                    dist[pid] += float(np.hypot(x - prev_xy[pid][0], y - prev_xy[pid][1]))
                vx = x - prev_xy.get(pid, (x, y))[0]
                prev_xy[pid] = (x, y)
                lista = asign[pid]
                spr = lista[int(dist[pid] / 1.3) % len(lista)]
                sombra(img, pie[0], pie[1], alto_px * 0.32)
                caja = pegar(img, spr, pie[0], pie[1], alto_px, espejo=vx * signo < 0)
                ji = jug_info[pid]
                visible = caja is not None and 0 <= pie[0] < ancho and 0 <= pie[1] < alto
                gt_j.append({"cuadro": k_out, "t": round(t_out, 3), "tramo": nt, "id": pid, "nombre": ji["nombre"],
                             "equipo": ji["equipo"], "rol": ji["rol"], "x": round(x, 3), "y": round(y, 3),
                             "u": round(float(pie[0]), 1), "v": round(float(pie[1]), 1),
                             "alto_px": round(float(alto_px), 1), "visible": bool(visible)})
            # balón (con su sombra en el suelo)
            if not np.isnan(bx):
                suelo_b = cam.proyectar(prm, [[bx, by, 0.0]])[0]
                aire = cam.proyectar(prm, [[bx, by, bz + 0.11]])[0]
                r_px = max(1.6, float(np.linalg.norm(cam.proyectar(prm, [[bx + 0.11, by, 0.11]])[0] - cam.proyectar(prm, [[bx - 0.11, by, 0.11]])[0]) / 2))
                sombra(img, suelo_b[0], suelo_b[1], r_px * 2.2)
                cv2.circle(img, (int(round(aire[0])), int(round(aire[1]))), int(round(r_px)), (245, 245, 245), -1, cv2.LINE_AA)
                gt_b.append({"cuadro": k_out, "t": round(t_out, 3), "tramo": nt, "x": round(bx, 3), "y": round(by, 3),
                             "z": round(bz, 3), "u": round(float(aire[0]), 1), "v": round(float(aire[1]), 1),
                             "visible": bool(0 <= aire[0] < ancho and 0 <= aire[1] < alto)})
            # gráfico de TV (marcador y reloj)
            if a.camara == "tv":
                goles_ahora = list(marc[nt])
                seg_partido = 45 * 60 + (s - tr.loc[tr["periodo"] == 2, "cuadro"].min()) / 10.0
                reloj = f"{int(seg_partido // 60):02d}:{int(seg_partido % 60):02d}"
                marcador(img, abrev, goles_ahora, reloj, escala=alto / 720)
            if a.camara == "tv":
                img = cv2.GaussianBlur(img, (3, 3), 0.6)
            proc.stdin.write(img.tobytes())
            H_i2c = np.linalg.inv(H_c2i)
            gt_c.append({"cuadro": k_out, "t": round(t_out, 3), "tramo": nt,
                         **{f"h{i}": float(v) for i, v in enumerate((H_i2c / H_i2c[2, 2]).ravel())}})
            if k == 0:
                pares_clave = clic_simulado(H_c2i, cancha, ancho, alto, a.ruido_clic, rng)
                if pares_clave:
                    Hk = h_desde_puntos([p["imagen"] for p in pares_clave], [p["cancha"] for p in pares_clave])
                    claves.append({"cuadro": k_out, "puntos": pares_clave, "H_img2cancha": Hk.tolist(), "tramo": nt})
            k_out += 1
        tramos_out.append({"tramo": nt, "cuadros_sc": [t0, t1], "cuadro_ini": ini_out, "cuadro_fin": k_out - 1,
                           "t_ini": round(ini_out / a.fps, 2), "t_fin": round((k_out - 1) / a.fps, 2)})
        for g in goles_sc.itertuples():
            if t0 <= g.cuadro <= t1:
                goles_out.append({"t": round(ini_out / a.fps + (g.cuadro - t0) / 10.0, 2), "equipo": g.equipo,
                                  "jugador": g.jugador, "cuadro_sc": int(g.cuadro)})
    proc.stdin.close()
    proc.wait()

    pd.DataFrame(gt_j).to_csv(salida / "gt_jugadores.csv", index=False)
    pd.DataFrame(gt_b).to_csv(salida / "gt_balon.csv", index=False)
    pd.DataFrame(gt_c).to_csv(salida / "gt_camara.csv", index=False)
    base = {"cancha": {"largo": cancha.largo, "ancho": cancha.ancho}, "resolucion": [ancho, alto],
            "camara_fija": a.camara == "completa", "lente": None}
    (salida / "calibracion.json").write_text(json.dumps({**base, "claves": claves[:1]}, indent=2, ensure_ascii=False), encoding="utf-8")
    (salida / "calibracion_por_tramo.json").write_text(json.dumps({**base, "claves": claves}, indent=2, ensure_ascii=False), encoding="utf-8")
    info = {"fuente": f"SkillCorner Open Data, partido {a.partido}", "camara": a.camara, "fps": a.fps,
            "resolucion": [ancho, alto], "cancha": base["cancha"], "tramos": tramos_out, "goles": goles_out,
            "equipos": {k: {"nombre": info_p["equipos"][k]["nombre"], "kit": "azul" if k == "A" else "blanco"} for k in "AB"},
            "ruido_clic_px": a.ruido_clic, "cuadros": k_out}
    (salida / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"{k_out} cuadros ({k_out / a.fps:.1f} s) en {salida / 'video.mp4'} · goles: "
          + ", ".join(f"{g['t']:.1f} s {g['jugador']}" for g in goles_out))


if __name__ == "__main__":
    main()
