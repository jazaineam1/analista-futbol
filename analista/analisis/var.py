"""Revisión tipo VAR a partir de posiciones: fuera de lugar, gol, balón fuera y faltas en el área.

Sirve para cualquier tracking en el esquema común (el que sale del video, SkillCorner) y para las
capturas 360 de StatsBomb (posiciones de los jugadores visibles en el instante de cada evento).

Reglas aplicadas (Ley 11 de la IFAB, fuera de lugar)
  - Posición adelantada: en el campo rival y más cerca de la línea de meta rival que el balón y
    que el penúltimo adversario (el arquero cuenta). Estar en línea es estar habilitado.
  - Se juzga en el instante en que un compañero juega el balón (el "momento del pase").
  - No hay fuera de lugar en saques de banda, de meta ni de esquina.
  - Solo se sanciona al que interviene: aquí, el receptor del pase. Los demás adelantados se
    listan como "en posición adelantada, sin intervenir".

Margen e incertidumbre
  Cada revisión da el margen en metros (positivo = adelantado) y una incertidumbre σ que suma:
    - el error de posición de la fuente (atacante y defensor), mayor si un jugador está fuera de
      cámara y su posición es estimada;
    - la parte del cuerpo: el VAR usa el punto más adelantado (pie, rodilla, hombro); con un punto
      por jugador se usa el centro sobre el suelo y la diferencia entra como error;
    - el instante del pase: con 10 o 25 cuadros por segundo el balón sale entre dos cuadros. Lo que
      se acercan o alejan atacante y defensor en ese tiempo entra en la incertidumbre.
  Si el margen está dentro de ±2σ la jugada es "dudosa": la medición no alcanza para decidir.

Cobertura
  Con video de TV la cámara no ve toda la cancha. Si el arquero no está a la vista se supone en su
  arco (lo habitual) y se avisa. Si un defensor podría estar fuera de imagen más atrás que el
  atacante, un "fuera de lugar" pasa a "dudoso": ese defensor lo habilitaría.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

ALTO_ARCO = 2.44
POSTE = 3.66            # medio ancho del arco (7,32 m)
AREA_LARGO = 16.5
AREA_MEDIO_ANCHO = 20.16


@dataclass
class ParametrosVar:
    largo: float = 105.0
    ancho: float = 68.0
    error_posicion_m: float = 0.3      # σ de la posición de un jugador visible
    error_estimado_m: float = 1.5      # σ de un jugador fuera de cámara (posición extrapolada)
    error_cuerpo_m: float = 0.15       # punto más adelantado del cuerpo vs centro del jugador
    error_tiempo_s: float = 0.05       # σ del instante del pase (medio cuadro a 10 cuadros/s)
    error_balon_m: float = 0.3         # σ de la posición del balón
    radio_balon: float = 0.11
    k_dudoso: float = 2.0              # |margen| < k σ -> dudoso
    radio_control: float = 1.6         # m: jugador que toca el balón
    fuente: str = "tracking"

    @property
    def L2(self) -> float:
        return self.largo / 2

    @property
    def W2(self) -> float:
        return self.ancho / 2


def parametros_para(fuente: str, fps: float, largo: float = 105.0, ancho: float = 68.0,
                    camara_fija: bool = False) -> ParametrosVar:
    """Errores típicos de cada fuente (los del video salen de la validación con verdad conocida)."""
    if fuente == "video":
        pos = 0.15 if camara_fija else 0.3     # error del punto del pie; la cámara se suma por cuadro
        return ParametrosVar(largo, ancho, error_posicion_m=pos, error_estimado_m=2.0, error_balon_m=0.5,
                             error_tiempo_s=0.5 / max(fps, 1.0), fuente="video")
    if fuente == "statsbomb360":
        return ParametrosVar(largo, ancho, error_posicion_m=0.6, error_estimado_m=2.0, error_balon_m=0.6,
                             error_tiempo_s=0.0, fuente="statsbomb360")
    return ParametrosVar(largo, ancho, error_posicion_m=0.3, error_estimado_m=1.5, error_tiempo_s=0.5 / max(fps, 1.0),
                         fuente=fuente)


# ====================================================================== posiciones en un instante
@dataclass
class Instante:
    """Foto de la cancha en un instante: jugadores (id, equipo, rol, x, y, vx, vy, visible) y balón."""
    t: float
    jugadores: pd.DataFrame
    balon: tuple[float, float] | None = None
    balon_v: tuple[float, float] | None = None
    zona_visible: np.ndarray | None = None      # polígono (x, y) de la cancha que se ve; None = toda
    cuadro: int | None = None
    calidad_camara: float | None = None         # 0-1: qué tan bien calzan las líneas (solo video)


class Posiciones:
    """Acceso rápido a las posiciones de un tracking por tiempo, con interpolación entre cuadros."""

    def __init__(self, tracking: pd.DataFrame, balon: pd.DataFrame | None = None,
                 camara: pd.DataFrame | None = None, forma_imagen: tuple | None = None):
        cx, cy = ("x_suave", "y_suave") if "x_suave" in tracking.columns else ("x", "y")
        tr = tracking[tracking["equipo"].isin(["A", "B"])].dropna(subset=[cx, cy])
        cols = {"id": tr["id"].to_numpy(), "equipo": tr["equipo"].to_numpy(),
                "rol": (tr["rol"] if "rol" in tr else pd.Series("campo", index=tr.index)).fillna("campo").to_numpy(),
                "x": tr[cx].to_numpy(float), "y": tr[cy].to_numpy(float),
                "visible": (tr["detectado"].astype(bool) if "detectado" in tr else pd.Series(True, index=tr.index)).to_numpy()}
        if {"u", "v", "alto_px", "ancho_px"} <= set(tr.columns):        # caja en la imagen (video)
            for c in ("u", "v", "alto_px", "ancho_px"):
                cols[c] = tr[c].to_numpy(float)
        self.cajas = "u" in cols
        self.df = pd.DataFrame(cols, index=tr["cuadro"].to_numpy())
        tiempos = tr.groupby("cuadro")["t"].first()
        self.cuadros = tiempos.index.to_numpy()
        self.tiempos = tiempos.to_numpy(float)
        self.grupos = {c: g for c, g in self.df.groupby(level=0)}
        self.balon = None
        if balon is not None and len(balon):
            b = balon.dropna(subset=["x", "y"]).sort_values("t")
            self.balon = b
            self.bt = b["t"].to_numpy(float)
            self.bxy = b[["x", "y"]].to_numpy(float)
            self.bz = b["z"].to_numpy(float) if "z" in b else None
        self.camara = None
        if camara is not None and len(camara) and "h0" in camara and forma_imagen is not None:
            self.camara = camara.set_index("cuadro")
            self.forma = forma_imagen

    def _bracket(self, t: float) -> tuple[int, int, float]:
        i = int(np.searchsorted(self.tiempos, t))
        if i <= 0:
            return 0, 0, 0.0
        if i >= len(self.tiempos):
            return len(self.tiempos) - 1, len(self.tiempos) - 1, 0.0
        t0, t1 = self.tiempos[i - 1], self.tiempos[i]
        if t1 - t0 > 0.5:                               # hueco: no interpolar a través de un corte
            return (i - 1, i - 1, 0.0) if t - t0 < t1 - t else (i, i, 0.0)
        return i - 1, i, (t - t0) / max(t1 - t0, 1e-9)

    def balon_en(self, t: float) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
        if self.balon is None or not len(self.bt):
            return None, None
        i = int(np.searchsorted(self.bt, t))
        i0, i1 = max(0, i - 1), min(len(self.bt) - 1, i)
        if min(abs(self.bt[i0] - t), abs(self.bt[i1] - t)) > 0.4:
            return None, None
        if i1 == i0 or self.bt[i1] - self.bt[i0] > 0.5:
            k = i0 if abs(self.bt[i0] - t) <= abs(self.bt[i1] - t) else i1
            return tuple(self.bxy[k]), None
        f = (t - self.bt[i0]) / (self.bt[i1] - self.bt[i0])
        p = self.bxy[i0] * (1 - f) + self.bxy[i1] * f
        v = (self.bxy[i1] - self.bxy[i0]) / (self.bt[i1] - self.bt[i0])
        return (float(p[0]), float(p[1])), (float(v[0]), float(v[1]))

    @staticmethod
    def _ocluidos(g: pd.DataFrame) -> dict:
        """Jugadores cuya caja en la imagen se superpone con otra: su pie se ubica peor."""
        if len(g) < 2:
            return {}
        u, v, h, w = (g[c].to_numpy(float) for c in ("u", "v", "alto_px", "ancho_px"))
        x1, x2, y1, y2 = u - w / 2, u + w / 2, v - h, v
        ix = np.clip(np.minimum(x2[:, None], x2[None]) - np.maximum(x1[:, None], x1[None]), 0, None)
        iy = np.clip(np.minimum(y2[:, None], y2[None]) - np.maximum(y1[:, None], y1[None]), 0, None)
        inter = ix * iy
        area = (w * h)
        frac = inter / np.maximum(area[:, None], 1e-6)              # parte de mi caja tapada por otra
        np.fill_diagonal(frac, 0)
        tapado = np.nan_to_num(frac).max(axis=1) > 0.25
        return dict(zip(g["id"].to_numpy(), tapado))

    def instante(self, t: float) -> Instante:
        i0, i1, f = self._bracket(t)
        g0, g1 = self.grupos[self.cuadros[i0]], self.grupos[self.cuadros[i1]]
        dt = max(self.tiempos[i1] - self.tiempos[i0], 1e-9)
        a = g0.set_index("id")
        b = g1.set_index("id")
        filas = []
        for pid in a.index.union(b.index):
            ra = a.loc[pid] if pid in a.index else None
            rb = b.loc[pid] if pid in b.index else None
            if isinstance(ra, pd.DataFrame):
                ra = ra.iloc[0]
            if isinstance(rb, pd.DataFrame):
                rb = rb.iloc[0]
            if ra is not None and rb is not None:
                x, y = ra["x"] * (1 - f) + rb["x"] * f, ra["y"] * (1 - f) + rb["y"] * f
                vx, vy = ((rb["x"] - ra["x"]) / dt, (rb["y"] - ra["y"]) / dt) if i1 != i0 else (0.0, 0.0)
                r, vis = ra, bool(ra["visible"] and rb["visible"])
            else:
                r = ra if ra is not None else rb
                x, y, vx, vy, vis = r["x"], r["y"], 0.0, 0.0, bool(r["visible"])
            filas.append({"id": pid, "equipo": r["equipo"], "rol": r["rol"], "x": float(x), "y": float(y),
                          "vx": float(vx), "vy": float(vy), "visible": vis,
                          "ocluido": bool(self._ocluidos(g0 if f < 0.5 else g1).get(pid, False)) if self.cajas else False})
        bal, bv = self.balon_en(t)
        zona, calidad = None, None
        cuadro = int(self.cuadros[i0] if f < 0.5 else self.cuadros[i1])
        if self.camara is not None and cuadro in self.camara.index:
            from ..vision.camara import huella_camara
            fila = self.camara.loc[cuadro]
            if isinstance(fila, pd.DataFrame):
                fila = fila.iloc[0]
            if bool(fila.get("calibrado", True)):
                H = fila[[f"h{k}" for k in range(9)]].to_numpy(float).reshape(3, 3)
                zona = huella_camara(H, self.forma)
                calidad = float(fila["calidad"]) if "calidad" in fila and pd.notna(fila["calidad"]) else None
        return Instante(float(t), pd.DataFrame(filas), bal, bv, zona, cuadro, calidad)


def _dentro_poligono(pts: np.ndarray, poli: np.ndarray) -> np.ndarray:
    """Puntos (n, 2) dentro de un polígono convexo o no (regla par-impar)."""
    x, y = pts[:, 0], pts[:, 1]
    dentro = np.zeros(len(pts), bool)
    px, py = poli[:, 0], poli[:, 1]
    j = len(poli) - 1
    for i in range(len(poli)):
        cruza = ((py[i] > y) != (py[j] > y)) & (x < (px[j] - px[i]) * (y - py[i]) / (py[j] - py[i] + 1e-12) + px[i])
        dentro ^= cruza
        j = i
    return dentro


# ====================================================================== fuera de lugar
def error_camara_m(calidad: float | None) -> float:
    """Error típico de la calibración de la cámara según la calidad de calce (validación con verdad)."""
    if calidad is None:
        return 0.0
    return float(np.interp(calidad, [0.0, 0.3, 0.45, 0.6, 0.75, 0.85, 1.0], [1.7, 1.6, 1.25, 1.05, 0.9, 0.15, 0.12]))


def revisar_fuera_de_lugar(ins: Instante, equipo: str, direccion: int, p: ParametrosVar,
                           pasador=None, receptor=None, reanudacion: str | None = None,
                           receptor_xy: tuple[float, float] | None = None) -> dict:
    """Decide si el receptor (o cualquier atacante) estaba en posición adelantada en `ins`.

    direccion: +1 si `equipo` ataca hacia x positivo. receptor: id del jugador que recibe; si no se
    conoce, `receptor_xy` (p. ej. el destino del pase) elige al compañero más cercano a ese punto.
    """
    j = ins.jugadores
    d = float(direccion)
    out = {"tipo": "fuera de lugar", "t": round(ins.t, 3), "cuadro": ins.cuadro, "equipo": equipo,
           "pasador": pasador, "receptor": receptor, "direccion": int(direccion), "avisos": []}
    if reanudacion in ("saque de banda", "saque de meta", "córner"):
        out.update(veredicto="no aplica", explicacion=f"No hay fuera de lugar en un {reanudacion}.")
        return out
    ata = j[(j["equipo"] == equipo) & (j["id"] != pasador)].copy()
    dfn = j[(j["equipo"] != equipo)].copy()
    if not len(ata):
        out.update(veredicto="no revisable", explicacion="No se ve a ningún compañero del pasador.")
        return out
    ata["a"] = ata["x"] * d
    dfn["a"] = dfn["x"] * d
    zona = ins.zona_visible
    # Arquero no visible: se supone en su arco
    if not (dfn["rol"] == "arquero").any():
        out["avisos"].append("El arquero defensor no está a la vista: se supone en su arco.")
        dfn = pd.concat([dfn, pd.DataFrame([{"id": "arquero_supuesto", "equipo": "rival", "rol": "arquero",
                                             "x": d * (p.L2 - 1.0), "y": 0.0, "vx": 0.0, "vy": 0.0,
                                             "visible": False, "a": p.L2 - 1.0}])], ignore_index=True)
    if len(dfn) < 2:
        out.update(veredicto="no revisable", explicacion="Se ve a menos de dos defensores: no se puede trazar la línea.")
        return out
    dfn = dfn.sort_values("a", ascending=False).reset_index(drop=True)
    penultimo = dfn.iloc[1]
    bal = ins.balon
    a_bal = bal[0] * d if bal is not None else -np.inf
    linea_a = max(float(penultimo["a"]), a_bal)
    linea_por = "balón" if a_bal > penultimo["a"] else "penúltimo defensor"

    # Con video, el error de la cámara en ese cuadro: la calidad de calce de las líneas predice el
    # error real de la calibración (correlación −0,92 en la validación). Atacante y defensor se
    # corren casi juntos con la cámara, así que al margen le llega cerca de la mitad.
    s_cam = error_camara_m(ins.calidad_camara) * 0.5 if ins.calidad_camara is not None else 0.0

    def sigma_jugador(r) -> float:
        base = p.error_posicion_m if bool(r["visible"]) else p.error_estimado_m
        oc = r.get("ocluido", False)
        tapado = 1.5 if isinstance(oc, (bool, np.bool_)) and bool(oc) else 0.0      # caja superpuesta: pie mal ubicado
        return float(np.sqrt(base ** 2 + s_cam ** 2 + tapado ** 2))

    def evaluar(r) -> dict:
        margen = float(r["a"]) - linea_a
        s_def = p.error_balon_m if linea_por == "balón" else sigma_jugador(penultimo)
        v_rel = (r["vx"] - (ins.balon_v[0] if (linea_por == "balón" and ins.balon_v) else penultimo["vx"])) * d
        sigma = float(np.sqrt(sigma_jugador(r) ** 2 + s_def ** 2 + 2 * p.error_cuerpo_m ** 2 +
                              (v_rel * p.error_tiempo_s) ** 2))
        en_campo_rival = float(r["a"]) > 0
        return {"id": r["id"], "x": round(float(r["x"]), 2), "y": round(float(r["y"]), 2),
                "margen_m": round(margen, 2), "sigma_m": round(sigma, 2), "visible": bool(r["visible"]),
                "campo_rival": en_campo_rival, "adelantado": en_campo_rival and margen > 0}

    evals = [evaluar(r) for _, r in ata.iterrows()]
    # receptor
    rec = None
    if receptor is not None:
        rec = next((e for e in evals if e["id"] == receptor), None)
        if rec is None:
            out["avisos"].append("El receptor no está a la vista en el momento del pase.")
    elif receptor_xy is not None:
        dist = [np.hypot(e["x"] - receptor_xy[0], e["y"] - receptor_xy[1]) for e in evals]
        k = int(np.argmin(dist))
        if dist[k] < 12:
            rec = evals[k]
            out["receptor"] = rec["id"]
            out["avisos"].append("Receptor estimado: el compañero más cercano al destino del pase.")
    out.update({
        "linea_x": round(linea_a * d, 2), "linea_por": linea_por,
        "defensor": penultimo["id"], "defensor_xy": [round(float(penultimo["x"]), 2), round(float(penultimo["y"]), 2)],
        "balon": [round(bal[0], 2), round(bal[1], 2)] if bal is not None else None,
        "adelantados": [e for e in evals if e["adelantado"] and (rec is None or e["id"] != rec["id"])],
        "jugadores": [{"id": r["id"], "equipo": r["equipo"], "rol": r["rol"], "x": round(float(r["x"]), 2),
                       "y": round(float(r["y"]), 2), "visible": bool(r["visible"])}
                      for _, r in j.iterrows()],
    })
    if rec is None:
        peor = max(evals, key=lambda e: e["margen_m"] if e["campo_rival"] else -99)
        out.update(veredicto="no revisable" if not out["adelantados"] else "sin receptor",
                   margen_m=peor["margen_m"], sigma_m=peor["sigma_m"],
                   explicacion="No se sabe quién recibe el pase." + (
                       f" {len(out['adelantados'])} atacante(s) en posición adelantada." if out["adelantados"] else ""))
        return out
    out["receptor_xy"] = [rec["x"], rec["y"]]
    out["margen_m"], out["sigma_m"] = rec["margen_m"], rec["sigma_m"]
    k = p.k_dudoso * rec["sigma_m"]
    if not rec["campo_rival"] and rec["x"] * d < -k:
        out.update(veredicto="habilitado", explicacion="El receptor estaba en su propio campo al momento del pase.")
        return out
    if rec["margen_m"] > k:
        veredicto = "fuera de lugar"
    elif rec["margen_m"] < -k:
        veredicto = "habilitado"
    else:
        veredicto = "dudoso"
    # ¿Podría haber un defensor fuera de imagen más atrás que el receptor?
    if veredicto == "fuera de lugar" and zona is not None and len(zona) >= 3:
        # La franja de 12 m detrás del receptor, a lo ancho: ahí estaría un defensor que lo habilite
        # (más atrás suele estar solo el arquero, que ya cuenta)
        a_rec = rec["x"] * d
        xs = np.linspace(a_rec, min(a_rec + 12.0, p.L2), 8) * d
        ys = np.linspace(-p.W2, p.W2, 11)
        malla = np.array([(x, y) for x in xs for y in ys])
        cubierto = _dentro_poligono(malla, zona).mean()
        visibles_detras = int((dfn["a"] >= a_rec).sum())
        if cubierto < 0.75 and visibles_detras < 2:
            veredicto = "dudoso"
            out["avisos"].append(f"La cámara solo muestra el {cubierto:.0%} de la franja detrás del receptor: "
                                 "un defensor fuera de imagen podría habilitarlo.")
    if not rec["visible"]:
        out["avisos"].append("El receptor estaba fuera de cámara: su posición es estimada.")
    tapados = [n for n, r_ in (("el receptor", ata[ata["id"] == rec["id"]]), ("el defensor de la línea", dfn.iloc[[1]]))
               if "ocluido" in r_ and len(r_) and r_["ocluido"].iloc[0] is not None
               and isinstance(r_["ocluido"].iloc[0], (bool, np.bool_)) and bool(r_["ocluido"].iloc[0])]
    if tapados:
        out["avisos"].append(f"En la imagen, {' y '.join(tapados)} está tapado por otro jugador: su pie se ubica con menos precisión.")
    out["veredicto"] = veredicto
    quien = "el balón" if linea_por == "balón" else "el penúltimo defensor"
    del_ = "del balón" if linea_por == "balón" else "del penúltimo defensor"
    m, sg = _m(rec["margen_m"]), _m(rec["sigma_m"])
    if veredicto == "fuera de lugar":
        out["explicacion"] = f"Adelantado {m} m respecto {del_} (±{sg} m)."
    elif veredicto == "habilitado":
        out["explicacion"] = f"Habilitado: {_m(-rec['margen_m'])} m detrás {del_} (±{sg} m)."
    else:
        lado = f"{m} m adelantado respecto {del_}" if rec["margen_m"] > 0 else \
            (f"{_m(-rec['margen_m'])} m detrás {del_}" if rec["margen_m"] < 0 else f"en línea con {quien}")
        out["explicacion"] = f"Dudoso: {lado}, dentro del error de medición (±{sg} m)."
    return out


def id_texto(v) -> str:
    """Id de jugador como texto estable: 6, 6.0 y "6" dan "6"."""
    try:
        f = float(v)
        if np.isfinite(f) and f.is_integer():
            return str(int(f))
    except (TypeError, ValueError):
        pass
    return str(v)


def _m(v: float) -> str:
    """Metros con coma decimal: 0,42."""
    return f"{v:.2f}".replace(".", ",")


# ====================================================================== pases con su instante exacto
def patada(pos: Posiciones, t_desde: float, pasador=None, hasta_s: float = 1.6, v_min: float = 8.0,
           preferido=None) -> float | None:
    """Primer instante después de `t_desde` en que el balón sale disparado (≥ v_min m/s en línea
    recta durante 6 muestras, tolerando detecciones falsas sueltas). Devuelve el instante en que esa
    recta pasa más cerca del pasador (el contacto con el pie), o None."""
    if pos.balon is None:
        return None
    m = (pos.bt > t_desde - 0.1) & (pos.bt < t_desde + hasta_s)
    tt, xy = pos.bt[m], pos.bxy[m]
    n = 6
    for s0 in range(0, max(0, len(tt) - n + 1)):
        w = slice(s0, s0 + n)
        tw, pw = tt[w], xy[w]
        if tw[-1] - tw[0] > 0.6 or np.diff(tw).max() > 0.13:
            continue
        pasos = np.hypot(*np.diff(pw, axis=0).T) / np.maximum(np.diff(tw), 1e-6)
        if pasos.min() < 0.5 * np.median(pasos) or pasos.max() > 1.7 * np.median(pasos):
            continue                                    # un salto: detección falsa en la ventana
        A = np.column_stack([tw - tw[0], np.ones(n)])
        cx, rx, _, _ = np.linalg.lstsq(A, pw[:, 0], rcond=None)
        cy, ry, _, _ = np.linalg.lstsq(A, pw[:, 1], rcond=None)
        res = np.sqrt(((A @ cx - pw[:, 0]) ** 2 + (A @ cy - pw[:, 1]) ** 2).mean())
        vx, vy = cx[0], cy[0]
        v2 = vx * vx + vy * vy
        if v2 < v_min ** 2 or res > 0.3:
            continue
        t_k = tw[0]
        ins = pos.instante(tw[0])
        j = ins.jugadores if pasador is None else ins.jugadores[ins.jugadores["id"] == pasador]
        if len(j):
            # la recta del balón hacia atrás: el contacto es donde pasa más cerca de un jugador
            px, py = j["x"].to_numpy(float), j["y"].to_numpy(float)
            dt = np.clip(((px - cx[1]) * vx + (py - cy[1]) * vy) / v2, -0.6, 0.0)
            d = np.hypot(cx[1] + vx * dt - px, cy[1] + vy * dt - py)
            # el último que tocó el balón es el candidato natural si la recta pasa cerca de él (con el
            # balón en el aire la proyección al suelo tuerce la recta y puede rozar a otro)
            pref = (j["id"] == preferido).to_numpy() if preferido is not None else np.zeros(len(j), bool)
            k = int(np.argmax(pref)) if pref.any() and d[pref].min() < 3.0 else int(np.argmin(d))
            t_k = tw[0] + float(dt[k])
            patada.ultimo = (j["id"].iloc[k], j["equipo"].iloc[k], float(d[k]))
        else:
            patada.ultimo = None
        return float(t_k)
    return None


patada.ultimo = None      # (id, equipo, distancia) del jugador que pateó en la última llamada


def pases_sin_recepcion(pos: Posiciones, toques: pd.DataFrame, ya: list[float], v_min: float = 8.0) -> pd.DataFrame:
    """Pases cuyo receptor no llegó a tocar el balón (p. ej. porque el árbitro pitó fuera de lugar) o
    cuyo toque no se detectó: el balón sale disparado tras el toque del pasador. El receptor buscado
    es el compañero más cercano a la trayectoria del balón."""
    filas = []
    if toques is None or not len(toques) or pos.balon is None:
        return pd.DataFrame(filas)
    lst = list(toques.itertuples())
    for i, a in enumerate(lst):
        sig = lst[i + 1] if i + 1 < len(lst) else None
        if sig is not None and sig.equipo == a.equipo and sig.id != a.id and sig.t_ini - a.t_fin <= 4.0:
            continue                                        # pase normal: ya está revisado
        tk = patada(pos, a.t_fin - 0.1, None, v_min=v_min, preferido=a.id)
        if tk is None or any(abs(tk - t) < 0.5 for t in ya) or patada.ultimo is None or patada.ultimo[2] > 2.5:
            continue
        pid, eq, _ = patada.ultimo                 # quien pateó (puede no ser el último que "tocó")
        if any(abs(tk - f["t_pase"]) < 0.5 for f in filas):
            continue
        m = (pos.bt > tk) & (pos.bt <= tk + 2.0)
        if m.sum() < 4:
            continue
        tray = pos.bxy[m]
        ins = pos.instante(tk)
        comp = ins.jugadores[(ins.jugadores["equipo"] == eq) & (ins.jugadores["id"] != pid)]
        pj = ins.jugadores[ins.jugadores["id"] == pid]
        if len(pj):     # pegado al pasador no puede ser el destino de un pase fuerte (suele ser la misma persona)
            comp = comp[np.hypot(comp["x"] - pj["x"].iloc[0], comp["y"] - pj["y"].iloc[0]) > 3.0]
        if not len(comp):
            continue
        # El receptor buscado está sobre el tramo final del recorrido (no junto al pasador)
        largo_tray = np.concatenate([[0], np.cumsum(np.hypot(*np.diff(tray, axis=0).T))])
        if largo_tray[-1] < 6.0:
            continue
        final = tray[largo_tray >= 0.4 * largo_tray[-1]]
        dist = np.array([np.min(np.hypot(final[:, 0] - x, final[:, 1] - y)) for x, y in zip(comp["x"], comp["y"])])
        k = int(np.argmin(dist))
        receptor = comp["id"].iloc[k] if dist[k] <= 6.0 else None
        filas.append({"t_toque": tk, "t_pase": tk, "t_ini_toque": a.t_ini, "equipo": eq, "pasador": pid,
                      "receptor": receptor, "x_ini": a.x_fin, "y_ini": a.y_fin, "estimado": True,
                      "x_fin": float(tray[-1, 0]), "y_fin": float(tray[-1, 1])})
    return pd.DataFrame(filas)


def momento_del_pase(pos: Posiciones, t_ultimo_toque: float, pasador, ventana_s: float = 0.5) -> float:
    """Instante en que el balón sale del pie: se ajusta una recta a la trayectoria del balón justo
    después del último cuadro en que el pasador lo tenía y se busca cuándo estaba junto a él."""
    if pos.balon is None:
        return t_ultimo_toque
    tk = patada(pos, t_ultimo_toque - 0.15, pasador, hasta_s=ventana_s + 0.3)
    if tk is not None:
        return tk
    m = (pos.bt >= t_ultimo_toque - 0.05) & (pos.bt <= t_ultimo_toque + ventana_s)
    if m.sum() < 3:
        return t_ultimo_toque
    tt, xy = pos.bt[m], pos.bxy[m]
    A = np.column_stack([tt - t_ultimo_toque, np.ones(len(tt))])
    (vx, x0), _, _, _ = np.linalg.lstsq(A, xy[:, 0], rcond=None)
    (vy, y0), _, _, _ = np.linalg.lstsq(A, xy[:, 1], rcond=None)
    v2 = vx * vx + vy * vy
    if v2 < 9.0:                                # menos de 3 m/s: no es un pase claro
        return t_ultimo_toque
    ins = pos.instante(t_ultimo_toque)
    pj = ins.jugadores[ins.jugadores["id"] == pasador]
    if not len(pj):
        return t_ultimo_toque
    px, py = float(pj["x"].iloc[0]), float(pj["y"].iloc[0])
    dt = ((px - x0) * vx + (py - y0) * vy) / v2      # cuándo la recta pasa más cerca del pasador
    return float(t_ultimo_toque + np.clip(dt, -0.3, 0.05))


def reanudacion_de(pos: Posiciones, t_ini_toque: float, x: float, y: float, p: ParametrosVar) -> str | None:
    """Reconoce saques (sin fuera de lugar) por dónde estaba el balón al jugarse."""
    if abs(y) > p.W2 - 0.3:
        return "saque de banda"
    if abs(x) > p.L2 - 1.5 and abs(y) > p.W2 - 1.5:
        return "córner"
    if pos.balon is not None:
        antes = (pos.bt >= t_ini_toque - 3.0) & (pos.bt < t_ini_toque)
        if antes.any():
            fuera = (np.abs(pos.bxy[antes, 1]) > p.W2 + 0.5)
            if fuera.mean() > 0.3:
                return "saque de banda"
            quieto = np.ptp(pos.bxy[antes], axis=0).max() < 0.6 if antes.sum() > 3 else False
            en_area_chica = abs(x) > p.L2 - 6.0 and abs(y) < 10.0
            if quieto and en_area_chica:
                return "saque de meta"
    return None


def pases_para_revisar(toques: pd.DataFrame, max_vuelo_s: float = 4.0) -> pd.DataFrame:
    """Pases entre compañeros (incluye el toque siguiente aunque sea corto)."""
    filas = []
    if toques is None or len(toques) < 2:
        return pd.DataFrame(filas)
    for a, b in zip(toques.itertuples(), toques.iloc[1:].itertuples()):
        if a.id == b.id or b.t_ini - a.t_fin > max_vuelo_s or a.equipo != b.equipo:
            continue
        filas.append({"t_toque": a.t_fin, "t_ini_toque": a.t_ini, "equipo": a.equipo, "pasador": a.id,
                      "receptor": b.id, "t_recepcion": b.t_ini, "x_ini": a.x_fin, "y_ini": a.y_fin,
                      "x_fin": b.x_ini, "y_fin": b.y_ini})
    return pd.DataFrame(filas)


def revisar_pases(pos: Posiciones, pases: pd.DataFrame, direccion: dict, p: ParametrosVar,
                  solo_campo_rival: bool = True) -> list[dict]:
    """Revisión de fuera de lugar de cada pase (tabla de `pases_para_revisar` o de eventos con
    columnas t_toque | t_pase, equipo, pasador, receptor y opcional reanudacion / x_fin, y_fin)."""
    out = []
    for r in pases.itertuples():
        d = direccion.get(r.equipo, 1)
        t = getattr(r, "t_pase", None)
        if t is None or (isinstance(t, float) and np.isnan(t)):
            t = momento_del_pase(pos, r.t_toque, r.pasador)
        rean = getattr(r, "reanudacion", None)
        if rean is None and hasattr(r, "t_ini_toque"):
            rean = reanudacion_de(pos, r.t_ini_toque, r.x_ini, r.y_ini, p)
        ins = pos.instante(t)
        if r.receptor is None or (isinstance(r.receptor, float) and np.isnan(r.receptor)):
            # pase hacia afuera de la imagen: no se ve a quién iba
            rev = revisar_fuera_de_lugar(ins, r.equipo, d, p, r.pasador, None, rean)
            if not rev.get("adelantados") and rev.get("veredicto") in ("no revisable", "sin receptor"):
                if not _va_hacia_ataque(r, d):
                    continue
            rev.update(veredicto="no revisable", t_toque=round(float(r.t_toque), 3), fuera_de_camara=True,
                       explicacion="El pase va hacia una zona que la cámara no muestra en el instante del pase: "
                                   "no se ve al receptor (ni quizá al último defensor). Hace falta otra cámara.")
            out.append(rev)
            continue
        if solo_campo_rival:
            rj = ins.jugadores[ins.jugadores["id"] == r.receptor]
            if len(rj) and rj["x"].iloc[0] * d < -3:      # receptor claramente en su campo: no hace falta
                continue
        rev = revisar_fuera_de_lugar(ins, r.equipo, d, p, r.pasador, r.receptor, rean)
        rev["t_toque"] = round(float(r.t_toque), 3)
        if bool(getattr(r, "estimado", False)) is True:
            rev["avisos"].append("El receptor no tocó el balón (o no se detectó): es el compañero hacia el que iba el pase.")
        if hasattr(r, "evento"):
            rev["evento"] = r.evento
        if hasattr(r, "resultado"):
            rev["resultado_evento"] = r.resultado
            marcar_intervencion(rev, r.resultado)
        out.append(rev)
    return out


def _va_hacia_ataque(r, direccion: int) -> bool:
    """El balón avanzó al menos 8 m hacia el arco rival (un pase largo hacia adelante)."""
    xf, xi = getattr(r, "x_fin", None), getattr(r, "x_ini", None)
    return xf is not None and xi is not None and np.isfinite(xf) and (xf - xi) * direccion > 8.0


NO_LLEGO = {"unsuccessful", "Incomplete", "Out", "Unknown", "Pass Unknown"}


def marcar_intervencion(rev: dict, resultado: str | None):
    """Si el pase no le llegó al receptor, estar adelantado no es infracción (no intervino),
    salvo que el árbitro lo haya sancionado igual."""
    rev["sancionado"] = resultado in ("offside", "Pass Offside")
    if resultado in NO_LLEGO and rev.get("veredicto") in ("fuera de lugar", "dudoso"):
        rev["posicion"] = rev["veredicto"]
        rev["veredicto"] = "sin intervención"
        rev["explicacion"] = rev.get("explicacion", "") + " El pase no le llegó: no intervino en el juego."


# ====================================================================== gol, balón fuera
def revisar_gol(pos: Posiciones, t_gol: float, lado: int, p: ParametrosVar, ventana_s: float = 4.0) -> dict:
    """¿El balón cruzó entera la línea de meta entre los postes y bajo el travesaño?"""
    out = {"tipo": "línea de gol", "t": round(t_gol, 2), "lado": lado}
    if pos.balon is None:
        out.update(veredicto="no revisable", explicacion="No hay posiciones del balón.")
        return out
    m = (pos.bt >= t_gol - 1.0) & (pos.bt <= t_gol + ventana_s)
    if not m.any():
        out.update(veredicto="no revisable", explicacion="No se ve el balón cerca del instante del gol.")
        return out
    xy = pos.bxy[m]
    z = pos.bz[m] if pos.bz is not None else np.zeros(m.sum())
    entre_postes = np.abs(xy[:, 1]) <= POSTE - p.radio_balon + p.error_balon_m
    bajo = z <= ALTO_ARCO - p.radio_balon + 0.3
    pen = xy[:, 0] * lado - (p.L2 + p.radio_balon)          # > 0: el balón entero pasó la línea
    validos = entre_postes & bajo
    if not validos.any():
        out.update(veredicto="no revisable", margen_m=None,
                   explicacion="El balón no se ve entre los postes cerca de la línea.")
        return out
    k = int(np.argmax(np.where(validos, pen, -np.inf)))
    margen = float(pen[k])
    sigma = p.error_balon_m
    out.update({"margen_m": round(margen, 2), "sigma_m": round(sigma, 2),
                "balon": [round(float(xy[k, 0]), 2), round(float(xy[k, 1]), 2)],
                "t_max": round(float(pos.bt[m][k]), 2)})
    if margen > p.k_dudoso * sigma:
        out.update(veredicto="gol", explicacion=f"El balón cruzó entero la línea: {_m(margen)} m dentro (±{_m(sigma)} m).")
    elif margen < -p.k_dudoso * sigma:
        out.update(veredicto="no entró", explicacion=f"Al balón le faltaron {_m(-margen)} m para cruzar entero (±{_m(sigma)} m).")
    else:
        out.update(veredicto="dudoso", explicacion=f"Sobre la línea: {_m(margen)} m (±{_m(sigma)} m), dentro del error.")
    return out


def salidas_del_balon(pos: Posiciones, p: ParametrosVar, max_fuera_s: float = 1.2) -> list[dict]:
    """Momentos en que el balón cruzó entero una línea y el juego siguió (volvió enseguida).

    Una salida larga (saque de banda, de meta, córner) no se reporta: el árbitro la vio. Lo que
    interesa al VAR es la pelota que sale un instante y vuelve sin que se detenga el juego."""
    if pos.balon is None or len(pos.bt) < 5:
        return []
    x, y, t = pos.bxy[:, 0], pos.bxy[:, 1], pos.bt
    exceso_x = np.abs(x) - (p.L2 + p.radio_balon)
    exceso_y = np.abs(y) - (p.W2 + p.radio_balon)
    exceso = np.maximum(exceso_x, exceso_y)
    # el balón en el arco o en la red (hasta ~2,5 m detrás de la línea) no es "salida"
    en_arco = (exceso_x > 0) & (exceso_x < 3.0) & (np.abs(y) <= POSTE + 1.5)
    fuera = (exceso > 0) & ~en_arco
    out = []
    i, n = 0, len(t)
    while i < n:
        if not fuera[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and fuera[j + 1] and t[j + 1] - t[j] < 0.5:
            j += 1
        dur = t[j] - t[i]
        siguio = j + 1 < n and t[j + 1] - t[j] < 0.5 and not fuera[j + 1]
        # en juego al menos 1 s antes y después, y el balón rodando (no en la mano de un saque)
        antes = (t >= t[i] - 1.0) & (t < t[i])
        despues = (t > t[j]) & (t <= t[j] + 1.0)
        en_juego = antes.sum() >= 3 and despues.sum() >= 3 and not fuera[antes].any() and not fuera[despues].any()
        vel = np.hypot(*(pos.bxy[min(j + 1, n - 1)] - pos.bxy[max(i - 1, 0)])) / max(t[min(j + 1, n - 1)] - t[max(i - 1, 0)], 1e-6)
        if dur <= max_fuera_s and siguio and en_juego and vel > 3.0:
            k = i + int(np.argmax(exceso[i:j + 1]))
            linea = "de meta" if exceso_x[k] >= exceso_y[k] else "de banda"
            m = float(exceso[k])
            s = p.error_balon_m
            ver = "salió" if m > p.k_dudoso * s else "dudoso"
            out.append({"tipo": "balón fuera", "t": round(float(t[i]), 2), "t_fin": round(float(t[j]), 2),
                        "linea": linea, "margen_m": round(m, 2), "sigma_m": round(s, 2),
                        "balon": [round(float(x[k]), 2), round(float(y[k]), 2)], "veredicto": ver,
                        "explicacion": (f"El balón cruzó entero la línea {linea} por {_m(m)} m (±{_m(s)} m) "
                                        f"y se siguió jugando." if ver == "salió" else
                                        f"El balón pisó la línea {linea}: {_m(m)} m fuera (±{_m(s)} m), dentro del error.")})
        i = j + 1
    return out


def fase_de_ataque(toques: pd.DataFrame, t_gol: float, equipo: str, max_s: float = 30.0) -> pd.DataFrame:
    """Toques del equipo que marcó desde que recuperó el balón hasta el gol (la "APP" del VAR)."""
    if toques is None or not len(toques):
        return pd.DataFrame()
    previos = toques[(toques["t_fin"] <= t_gol + 0.5) & (toques["t_ini"] >= t_gol - max_s)].sort_values("t_ini")
    inicio = t_gol - max_s
    for r in previos.itertuples():
        if r.equipo != equipo and r.duracion >= 0.4:      # el rival la controló: empieza después
            inicio = r.t_fin
    return previos[(previos["t_ini"] >= inicio) & (previos["equipo"] == equipo)]


# ====================================================================== faltas en el área
def en_area(x: float, y: float, lado: int, p: ParametrosVar) -> tuple[bool, float]:
    """¿El punto está dentro del área penal del arco del lado `lado`? Devuelve también la
    distancia a la línea del área (positiva = dentro). La línea es parte del área."""
    a = x * lado
    dentro_x = a - (p.L2 - AREA_LARGO)             # > 0 dentro en largo
    dentro_y = AREA_MEDIO_ANCHO - abs(y)            # > 0 dentro en ancho
    fondo = p.L2 - a                                # > 0 antes de la línea de meta
    if dentro_x >= 0 and dentro_y >= 0 and fondo >= 0:
        return True, float(min(dentro_x, dentro_y))
    afuera = np.hypot(min(dentro_x, 0), min(dentro_y, 0))
    return False, -float(afuera)


def revisar_falta(x: float, y: float, lado_arco: int, p: ParametrosVar, sigma_m: float,
                  sancion: str | None = None) -> dict:
    dentro, dist = en_area(x, y, lado_arco, p)
    k = p.k_dudoso * sigma_m
    if dist > k:
        ver = "dentro del área"
    elif dist < -k:
        ver = "fuera del área"
    else:
        ver = "dudoso"
    txt = {"dentro del área": f"La falta fue {_m(dist)} m dentro del área: es penal.",
           "fuera del área": f"La falta fue {_m(-dist)} m fuera del área: es tiro libre.",
           "dudoso": f"La falta fue sobre la línea del área ({_m(dist)} m, ±{_m(sigma_m)} m)."}[ver]
    if sancion:
        coincide = (sancion == "penal") == (ver == "dentro del área") or ver == "dudoso"
        txt += f" Se sancionó {sancion}" + ("." if coincide else ": la posición sugiere revisarlo.")
    return {"tipo": "falta en el área", "x": round(x, 2), "y": round(y, 2), "margen_m": round(dist, 2),
            "sigma_m": round(sigma_m, 2), "veredicto": ver, "sancion": sancion, "explicacion": txt}



# ====================================================================== tiros: rebote e interferencia
def _arquero(ins: Instante, equipo_def: str, direccion: int, p: ParametrosVar) -> tuple[float, float, bool]:
    """Posición del arquero que defiende el arco hacia el que ataca `direccion` (o la supuesta)."""
    j = ins.jugadores
    a = j[(j["equipo"] == equipo_def) & (j["rol"] == "arquero")]
    if len(a):
        return float(a["x"].iloc[0]), float(a["y"].iloc[0]), True
    return direccion * (p.L2 - 1.0), 0.0, False


def _en_triangulo(pt, a, b, c) -> bool:
    def lado(p1, p2, p3):
        return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])
    d1, d2, d3 = lado(pt, a, b), lado(pt, b, c), lado(pt, c, a)
    return not ((d1 < 0 or d2 < 0 or d3 < 0) and (d1 > 0 or d2 > 0 or d3 > 0))


def interferencias(ins: Instante, adelantados: list[dict], direccion: int, p: ParametrosVar,
                   equipo_def: str) -> list[dict]:
    """Adelantados que, en el instante del tiro, tapan la visión del arquero, están en la trayectoria
    del balón hacia el arco o encima del arquero (Ley 11: interferir con un adversario)."""
    if ins.balon is None:
        return []
    bx, by = ins.balon
    gx, gy, visto = _arquero(ins, equipo_def, direccion, p)
    poste1, poste2 = (direccion * p.L2, -POSTE), (direccion * p.L2, POSTE)
    out = []
    for a in adelantados:
        x, y = a["x"], a["y"]
        motivos = []
        # sobre la recta balón-arquero, entre ambos: le tapa la visión
        vx, vy = gx - bx, gy - by
        largo2 = vx * vx + vy * vy
        if largo2 > 1.0:
            u = ((x - bx) * vx + (y - by) * vy) / largo2
            dist = abs((x - bx) * vy - (y - by) * vx) / np.sqrt(largo2)
            if 0.15 <= u <= 0.97 and dist <= 1.0:
                motivos.append("tapa la visión del arquero")
        if _en_triangulo((x, y), (bx, by), poste1, poste2) and "tapa la visión del arquero" not in motivos:
            motivos.append("está en la trayectoria del tiro")
        if np.hypot(x - gx, y - gy) <= 1.5:
            motivos.append("está encima del arquero")
        if motivos:
            out.append({**a, "motivos": motivos})
    return out


def revisar_tiro(pos: Posiciones | None, ins: Instante, equipo: str, direccion: int, p: ParametrosVar,
                 tirador=None, siguientes: list[dict] | None = None) -> dict | None:
    """Fuera de lugar en un tiro: el que estaba adelantado cuando su compañero remató y después
    juega el rebote (tras atajada, palo o desvío), o el que interfiere con el arquero.

    siguientes: los toques posteriores al tiro, en orden: [{"id", "equipo", "rol", "duracion"}].
    Devuelve None si nadie estaba adelantado."""
    base = revisar_fuera_de_lugar(ins, equipo, direccion, p, tirador)
    if base.get("linea_x") is None:
        return None
    ade = [a for a in base.get("adelantados", []) if a["margen_m"] > 0]
    if not ade:
        return None
    eq_def = "B" if equipo == "A" else "A"
    out = {k: v for k, v in base.items() if k not in ("adelantados",)}
    out.update(tipo="fuera de lugar", pasador=tirador, t=round(ins.t, 3), adelantados=ade)
    gx, gy, _ = _arquero(ins, eq_def, direccion, p)
    if ins.balon is not None:
        out["vision"] = [[round(ins.balon[0], 2), round(ins.balon[1], 2)], [round(gx, 2), round(gy, 2)]]
    k = p.k_dudoso

    # Rebote: el primer jugador del equipo que remató en jugar el balón después del tiro
    rebote = None
    for s_ in siguientes or []:
        if s_["equipo"] != equipo:
            if s_.get("rol") == "arquero" or s_.get("duracion", 0) < 0.4:
                continue                    # atajada o desvío: no "habilita"
            break                           # un defensor la jugó deliberadamente: se reinicia
        if s_["id"] == tirador:
            break                           # el mismo que remató: no hay fuera de lugar
        rebote = next((a for a in ade if a["id"] == s_["id"]), None)
        break
    if rebote is not None:
        m, sg = rebote["margen_m"], rebote["sigma_m"]
        ver = "fuera de lugar" if m > k * sg else "dudoso"
        out.update(motivo="rebote", receptor=rebote["id"], receptor_xy=[rebote["x"], rebote["y"]], margen_m=m, sigma_m=sg,
                   veredicto=ver,
                   explicacion=(f"Rebote: estaba {_m(m)} m adelantado (±{_m(sg)} m) cuando su compañero remató "
                                f"y luego jugó el balón." if ver == "fuera de lugar" else
                                f"Rebote: {_m(m)} m adelantado en el remate, dentro del error (±{_m(sg)} m)."))
        return out
    inter = interferencias(ins, ade, direccion, p, eq_def)
    if inter:
        peor = max(inter, key=lambda a: a["margen_m"] - k * a["sigma_m"])
        m, sg = peor["margen_m"], peor["sigma_m"]
        ver = "revisar" if m > k * sg else "dudoso"
        out.update(motivo="interferencia", receptor=peor["id"], receptor_xy=[peor["x"], peor["y"]], margen_m=m,
                   sigma_m=sg, veredicto=ver, interfieren=inter,
                   explicacion=(f"Adelantado {_m(m)} m (±{_m(sg)} m) en el remate y {' y '.join(peor['motivos'])}: "
                                f"si impidió al arquero jugar el balón es fuera de lugar (decide el árbitro)."
                                if ver == "revisar" else
                                f"En el remate {' y '.join(peor['motivos'])}, {_m(m)} m adelantado: dentro del error "
                                f"(±{_m(sg)} m)."))
        return out
    out.update(motivo="sin intervención", receptor=None, veredicto="sin intervención",
               margen_m=max(a["margen_m"] for a in ade), sigma_m=min(a["sigma_m"] for a in ade),
               explicacion=f"{len(ade)} atacante(s) adelantado(s) en el remate, sin jugar el rebote ni interferir con el arquero.")
    return out


# ====================================================================== ocasión manifiesta de gol (DOGSO)
def revisar_ocasion(ins: Instante, equipo_atacante: str, direccion: int, x: float, y: float, p: ParametrosVar,
                    victima=None, infractor=None) -> dict:
    """¿La falta cortó una ocasión manifiesta de gol? (Ley 12: distancia al arco, dirección de la
    jugada, control del balón y cantidad y ubicación de los defensores). Aproximación con posiciones:
    los defensores que quedan entre el atacante y el arco son los que podrían haber evitado el gol."""
    gx = direccion * p.L2
    dist = float(np.hypot(gx - x, y))
    eq_def = "B" if equipo_atacante == "A" else "A"
    j = ins.jugadores
    defs = j[(j["equipo"] == eq_def) & (j["id"] != infractor)] if len(j) else j
    entre, arquero_entre = [], False
    for r in defs.itertuples():
        a_def, a_at = r.x * direccion, x * direccion
        if a_def <= a_at + 0.5:
            # a su lado o apenas detrás: también puede disputarle el balón
            if a_def > a_at - 2.0 and abs(r.y - y) <= 2.5 and r.rol != "arquero":
                entre.append(r.id)
            continue                                  # más atrás: no llega
        # dentro del corredor del atacante hacia los postes (ensanchado 3 m)
        # el corredor del atacante hacia el arco, ancho: 4 m a su lado y los postes + 1 m en la línea
        f = min(1.0, (a_def - a_at) / max(p.L2 - a_at, 1e-6))
        if abs(r.y - y * (1 - f)) <= 4.0 + f * (POSTE + 1.0):
            if r.rol == "arquero":
                arquero_entre = True
            else:
                entre.append(r.id)
    control = True
    if ins.balon is not None:
        control = bool(np.hypot(ins.balon[0] - x, ins.balon[1] - y) <= 3.0)
    frente = abs(y) <= AREA_MEDIO_ANCHO + 4.0
    # ¿iba hacia el arco? (con tracking hay velocidad; con una foto no se sabe y se asume que sí)
    vic = j[j["id"] == victima] if victima is not None and len(j) else j.iloc[0:0]
    hacia_arco = True
    if len(vic) and ("vx" in vic) and np.hypot(vic["vx"].iloc[0], vic["vy"].iloc[0]) > 1.0:
        vx, vy = float(vic["vx"].iloc[0]), float(vic["vy"].iloc[0])
        hacia_arco = (vx * (gx - x) + vy * (0 - y)) / (np.hypot(vx, vy) * max(dist, 1e-6)) > 0.3
    # ¿la cámara mostraba el corredor? (con una foto parcial un defensor podría estar fuera)
    cubierto = 1.0
    if ins.zona_visible is not None and len(ins.zona_visible) >= 3:
        fs = np.linspace(0, 1, 7)
        malla = np.array([((x * direccion + f_ * (p.L2 - x * direccion)) * direccion, y * (1 - f_) + k * (4.0 + f_ * 4.66))
                          for f_ in fs for k in (-1, 0, 1)])
        cubierto = float(_dentro_poligono(malla, ins.zona_visible).mean())
    en_area_, _ = en_area(x, y, direccion, p)
    criterios = {"distancia_m": round(dist, 1), "defensores_entre": len(entre), "arquero_entre": arquero_entre,
                 "controla_balon": control, "frente_al_arco": frente, "hacia_el_arco": bool(hacia_arco),
                 "corredor_visible": round(cubierto, 2)}
    if dist <= 22.0 and abs(y) <= 16.0 and control and hacia_arco and len(entre) == 0 and cubierto < 0.8:
        ver = "dudoso"
        txt = (f"A {_m(dist)} m del arco y sin defensores a la vista entre él y el arco, pero la imagen solo muestra "
               f"el {cubierto:.0%} de ese camino: un defensor fuera de cámara podría cambiarlo.")
    elif dist <= 22.0 and abs(y) <= 16.0 and control and hacia_arco and len(entre) == 0:
        ver = "ocasión manifiesta"
        san = ("penal; roja si no intentó jugar el balón (amarilla si lo intentó)" if en_area_ else "tiro libre y roja")
        txt = (f"A {_m(dist)} m del arco, con el balón y sin defensores de campo entre él y el arco: ocasión "
               f"manifiesta de gol. Sanción: {san}.")
    elif dist <= 25.0 and frente and len(entre) <= 1 and hacia_arco:
        ver = "dudoso"
        txt = (f"A {_m(dist)} m del arco con {len(entre)} defensor(es) de campo por delante"
               f"{'' if control else ' y sin el balón controlado'}: hay que mirar la jugada.")
    else:
        ver = "no es ocasión manifiesta"
        motivo = ("lejos del arco" if dist > 25 else "en la banda" if not frente else
                  "no iba hacia el arco" if not hacia_arco else f"{len(entre)} defensores podían intervenir")
        txt = f"No es ocasión manifiesta de gol ({motivo})."
    return {"tipo": "ocasión manifiesta", "veredicto": ver, "criterios": criterios, "explicacion": txt,
            "victima": victima, "infractor": infractor, "x": round(x, 2), "y": round(y, 2), "t": round(ins.t, 3),
            "equipo": equipo_atacante, "direccion": int(direccion),
            "jugadores": [{"id": r.id, "equipo": r.equipo, "rol": r.rol, "x": round(float(r.x), 2),
                           "y": round(float(r.y), 2), "visible": bool(r.visible)} for r in j.itertuples()] if len(j) else []}


# ====================================================================== resumen
@dataclass
class RevisionVar:
    fueras_de_lugar: list = field(default_factory=list)
    goles: list = field(default_factory=list)
    salidas: list = field(default_factory=list)
    faltas: list = field(default_factory=list)
    tiros: list = field(default_factory=list)          # fuera de lugar en remates: rebote e interferencia
    tiros_revisados: int = 0

    def todas(self) -> list[dict]:
        return sorted(self.fueras_de_lugar + self.goles + self.salidas + self.faltas + self.tiros,
                      key=lambda r: r.get("t") or 0)

    def resumen(self) -> dict:
        fl = self.fueras_de_lugar
        cont = pd.Series([r["veredicto"] for r in fl]).value_counts().to_dict() if fl else {}
        ti = pd.Series([r["veredicto"] for r in self.tiros]).value_counts().to_dict() if self.tiros else {}
        oc = [f.get("ocasion", {}).get("veredicto") for f in self.faltas]
        return {"pases_revisados": len(fl), "fuera_de_lugar": int(cont.get("fuera de lugar", 0)),
                "dudosos": int(cont.get("dudoso", 0)), "habilitados": int(cont.get("habilitado", 0)),
                "goles_revisados": len(self.goles), "salidas_al_limite": len(self.salidas),
                "faltas_revisadas": len(self.faltas),
                "tiros_revisados": int(self.tiros_revisados), "rebotes_fuera_de_lugar": int(ti.get("fuera de lugar", 0)),
                "interferencias": int(ti.get("revisar", 0)), "tiros_dudosos": int(ti.get("dudoso", 0)),
                "ocasiones_manifiestas": int(sum(v == "ocasión manifiesta" for v in oc)),
                "manos": int(sum(1 for f in self.faltas if f.get("motivo") == "mano"))}

    def destacadas(self, margen_cerca_m: float = 0.5) -> list[dict]:
        """Lo que revisaría un VAR: fueras de lugar (también los que pitó el árbitro), dudosos,
        habilitados por muy poco, jugadas de gol, balones que salieron y faltas junto al área."""
        sel = []
        for r in self.fueras_de_lugar:
            if r["veredicto"] in ("fuera de lugar", "dudoso") or r.get("sancionado") or r.get("en_gol") \
                    or r.get("fuera_de_camara") or (
                    r["veredicto"] == "habilitado" and r.get("margen_m") is not None and r["margen_m"] > -margen_cerca_m):
                sel.append(r)
        sal = [s for s in self.salidas if s["veredicto"] == "salió" or s.get("en_gol")]
        tiros = [r for r in self.tiros if r["veredicto"] in ("fuera de lugar", "revisar", "dudoso") or r.get("en_gol")]
        return sorted(sel + self.goles + sal + self.faltas + tiros, key=lambda r: r.get("t") or 0)


def revisar_tiros_tracking(pos: Posiciones, tiros: list[dict], toques: pd.DataFrame, roles: dict,
                           direccion: dict, p: ParametrosVar) -> list[dict]:
    """tiros: [{"t", "equipo", "tirador"}]. El instante exacto es cuando el balón sale del pie."""
    out = []
    for ti in tiros:
        eq, tirador = ti["equipo"], ti.get("tirador")
        tk = patada(pos, ti["t"] - 0.3, tirador, hasta_s=0.8, v_min=10.0) if pos.balon is not None else None
        t = tk if tk is not None else ti["t"]
        sig = []
        if toques is not None and len(toques):
            post = toques[(toques["t_ini"] > t + 0.1) & (toques["t_ini"] < t + 6.0)].sort_values("t_ini")
            sig = [{"id": r.id, "equipo": r.equipo, "rol": roles.get(r.id, "campo"), "duracion": float(r.duracion)}
                   for r in post.itertuples()]
        r = revisar_tiro(pos, pos.instante(t), eq, direccion.get(eq, 1), p, tirador, sig)
        if r is not None:
            r["t_tiro"] = round(float(ti["t"]), 2)
            out.append(r)
    return out


def tiros_de_toques(eventos: pd.DataFrame, toques: pd.DataFrame) -> list[dict]:
    """Remates detectados en la trayectoria del balón -> [{"t", "equipo", "tirador"}] (el último que tocó)."""
    out = []
    if eventos is None or not len(eventos):
        return out
    for e in eventos[eventos["tipo"].isin(["remate", "gol", "posible gol"])].itertuples():
        eq = e.equipo_atacante
        if eq not in ("A", "B"):
            continue
        tirador, t = None, float(e.t)
        if toques is not None and len(toques):
            prev = toques[(toques["t_ini"] <= e.t + 0.1) & (toques["t_fin"] >= e.t - 2.0) & (toques["equipo"] == eq)]
            if len(prev):
                fila = prev.sort_values("t_fin").iloc[-1]
                tirador, t = fila["id"], float(fila["t_fin"])
        if any(abs(t - o["t"]) < 1.0 for o in out):
            continue
        out.append({"t": t, "equipo": eq, "tirador": tirador})
    return out


def revisar_tracking(tracking: pd.DataFrame, balon: pd.DataFrame, p: ParametrosVar, direccion: dict,
                     toques: pd.DataFrame | None = None, goles: list[dict] | None = None,
                     camara: pd.DataFrame | None = None, forma_imagen: tuple | None = None,
                     pases: pd.DataFrame | None = None, tiros: list[dict] | None = None,
                     afinar=None) -> tuple[RevisionVar, Posiciones]:
    """VAR completo de un tracking: fuera de lugar en cada pase, línea de gol y la jugada previa
    de cada gol, y balones que salieron sin que se detuviera el juego.

    goles: [{"t", "equipo"}] (equipo que marcó). pases: tabla de eventos de pase si existe (si no,
    se detectan con los toques)."""
    from .tracking import ParametrosMetricas, poseedor_por_cuadro, secuencia_toques
    pos = Posiciones(tracking, balon, camara, forma_imagen)
    rev = RevisionVar()
    if toques is None:
        pm = ParametrosMetricas(largo=p.largo, ancho=p.ancho, radio_control=p.radio_control)
        pose = poseedor_por_cuadro(tracking, balon, pm)
        toques = secuencia_toques(pose, min_dur_s=0.0) if len(pose) else pd.DataFrame()
    if pases is None:
        pases = pases_para_revisar(toques)
        largos = pases_sin_recepcion(pos, toques, [])
        if len(largos):
            pases = pd.concat([pases, largos], ignore_index=True).sort_values("t_toque").reset_index(drop=True)
    if len(pases):
        rev.fueras_de_lugar = revisar_pases(pos, pases, direccion, p)
        if afinar is not None:          # p. ej. el punto más adelantado del cuerpo, medido en el video
            afinar(rev.fueras_de_lugar)
    for g in goles or []:
        lado = direccion.get(g["equipo"], 1)
        rg = revisar_gol(pos, g["t"], lado, p)
        rg["equipo"] = g["equipo"]
        app = fase_de_ataque(toques, g["t"], g["equipo"])
        if len(app):
            t0 = float(app["t_ini"].min())
            rg["jugada_desde"] = round(t0, 2)
            en_app = [r for r in rev.fueras_de_lugar if t0 - 0.2 <= r["t"] <= g["t"]]
            for r in en_app:
                r["en_gol"] = round(float(g["t"]), 2)
            sal = [s for s in salidas_del_balon(pos, p) if t0 <= s["t"] <= g["t"]]
            for s_ in sal:
                s_["en_gol"] = round(float(g["t"]), 2)
            rev.salidas.extend(s_ for s_ in sal if s_ not in rev.salidas)
            rg["revision_jugada"] = {
                "pases": len(en_app),
                "fuera_de_lugar": [r["t"] for r in en_app if r["veredicto"] == "fuera de lugar"],
                "dudosos": [r["t"] for r in en_app if r["veredicto"] == "dudoso"],
                "salidas": [s["t"] for s in sal if s["veredicto"] == "salió"],
                "mas_justo_m": min((abs(r["margen_m"]) for r in en_app if r.get("margen_m") is not None), default=None),
            }
            rj = rg["revision_jugada"]
            problemas = []
            if rj["fuera_de_lugar"]:
                problemas.append(f"fuera de lugar en el pase de {_mmss(rj['fuera_de_lugar'][0])}")
            if rj["salidas"]:
                problemas.append(f"el balón salió en {_mmss(rj['salidas'][0])}")
            if rj["dudosos"]:
                problemas.append(f"pase dudoso en {_mmss(rj['dudosos'][0])}")
            rg["decision_var"] = ("revisar: " + "; ".join(problemas)) if problemas else "gol válido"
            rg["explicacion"] = (rg.get("explicacion", "") + f" Jugada previa: {rj['pases']} pase(s) revisado(s)"
                                 + (f", el más justo a {_m(rj['mas_justo_m'])} m." if rj["mas_justo_m"] is not None else ".")
                                 + (" " + "; ".join(problemas).capitalize() + "." if problemas else " Sin infracciones."))
        rev.goles.append(rg)
    roles = tracking.groupby("id")["rol"].agg(lambda s_: s_.mode().iloc[0] if len(s_.mode()) else "campo").to_dict() \
        if "rol" in tracking else {}
    if tiros is None:
        from .tracking import ParametrosMetricas, eventos_balon
        pm = ParametrosMetricas(largo=p.largo, ancho=p.ancho)
        tiros = tiros_de_toques(eventos_balon(balon, pm, direccion, camara), toques)
    rev.tiros = revisar_tiros_tracking(pos, tiros, toques, roles, direccion, p)
    if afinar is not None:
        afinar(rev.tiros)
    rev.tiros_revisados = len(tiros)
    for g in goles or []:                 # ¿un rebote o interferencia en la jugada del gol?
        for r in rev.tiros:
            if g["t"] - 15.0 <= r["t"] <= g["t"] + 0.5 and r["equipo"] == g["equipo"]:
                r["en_gol"] = round(float(g["t"]), 2)
                if r["veredicto"] in ("fuera de lugar", "revisar"):
                    rg = next((x for x in rev.goles if x.get("equipo") == g["equipo"] and abs(x["t"] - g["t"]) < 1), None)
                    if rg is not None and str(rg.get("decision_var", "")).startswith("gol válido"):
                        rg["decision_var"] = f"revisar: {r['motivo']} en el remate de {_mmss(r['t'])}"
    vistas = {s["t"] for s in rev.salidas}
    t_goles = [g["t"] for g in goles or []]
    rev.salidas += [s for s in salidas_del_balon(pos, p) if s["t"] not in vistas
                    and not any(-1.0 <= s["t"] - tg <= 10.0 for tg in t_goles)]     # tras el gol el balón queda en la red
    rev.salidas.sort(key=lambda s: s["t"])
    return rev, pos


def _mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


# ====================================================================== animación
def ventana_animacion(pos: Posiciones, t: float, antes_s: float = 3.0, despues_s: float = 2.0,
                      fps: float = 10.0, nombres: dict | None = None) -> dict:
    """Cuadros de una jugada para la repetición 2D: {t0, fps, cuadros: [[[id, eq, x, y], ...], [bx, by]]}."""
    cuadros = []
    for k in range(int(round((antes_s + despues_s) * fps)) + 1):
        tk = t - antes_s + k / fps
        if tk < pos.tiempos[0] - 0.2 or tk > pos.tiempos[-1] + 0.2:
            continue
        ins = pos.instante(tk)
        js = [[id_texto(r.id), r.equipo, round(r.x, 2), round(r.y, 2), 1 if r.rol == "arquero" else 0, 1 if r.visible else 0]
              for r in ins.jugadores.itertuples()]
        b = None
        if ins.balon:
            b = [round(ins.balon[0], 2), round(ins.balon[1], 2)]
            if pos.bz is not None and len(pos.bt):
                z = float(np.interp(tk, pos.bt, np.nan_to_num(pos.bz)))
                if z > 0.3:
                    b.append(round(z, 2))
        cuadros.append({"t": round(tk, 2), "j": js, "b": b})
    return {"t_evento": round(t, 3), "fps": fps, "cuadros": cuadros,
            "nombres": {id_texto(k): v for k, v in (nombres or {}).items()}}


CAMPOS_REPETICION = ("tipo", "veredicto", "linea_x", "linea_por", "direccion", "pasador", "receptor", "defensor",
                     "margen_m", "sigma_m", "receptor_xy", "defensor_xy", "balon", "explicacion", "avisos", "x", "y",
                     "t", "decision_var", "linea", "motivo", "vision", "ocasion")


def para_repeticion(rev: dict, pos: Posiciones | None, nombres: dict | None = None, largo: float = 105.0,
                    ancho: float = 68.0, antes_s: float = 3.0, despues_s: float = 2.0, fps: float = 10.0) -> dict:
    """Datos para `Repeticion.montar` (reportes/activos/repeticion.js): la jugada alrededor del
    instante revisado, con la revisión encima. Sin tracking (StatsBomb 360) es una sola foto."""
    r = {k: rev[k] for k in CAMPOS_REPETICION if k in rev and rev[k] is not None}
    for k in ("pasador", "receptor", "defensor"):
        if k in r:
            r[k] = id_texto(r[k])
    if rev.get("adelantados"):
        r["adelantados"] = [{"id": id_texto(a["id"]), "x": a["x"], "y": a["y"], "margen_m": a["margen_m"]}
                            for a in rev["adelantados"]]
    if rev.get("tipo") in ("falta en el área", "posible penal") and rev.get("receptor") is not None:
        r["pasador"] = id_texto(rev["receptor"])          # la víctima, para marcarla
        r.pop("receptor", None)
    t = float(rev.get("t") or 0.0)
    if rev.get("tipo") == "línea de gol" and rev.get("t_max") is not None:
        t = float(rev["t_max"])
    if pos is not None and len(pos.tiempos):
        datos = ventana_animacion(pos, t, antes_s, despues_s, fps, nombres)
        if pos.camara is not None:
            ins = pos.instante(t)
            if ins.zona_visible is not None:
                r["zona"] = np.round(ins.zona_visible, 1).tolist()
    else:
        js = [[id_texto(j["id"]), j["equipo"], j["x"], j["y"], 1 if j.get("rol") == "arquero" else 0, 1 if j.get("visible", True) else 0]
              for j in rev.get("jugadores", [])]
        datos = {"t_evento": t, "fps": 1, "cuadros": [{"t": t, "j": js, "b": rev.get("balon")}],
                 "nombres": {id_texto(k): v for k, v in (nombres or {}).items()}}
        if rev.get("zona") is not None:
            r["zona"] = rev["zona"]
    datos["t_evento"] = round(t, 3)
    datos["cancha"] = {"largo": largo, "ancho": ancho}
    datos["revision"] = r
    return datos


def revisar_instante(pos: Posiciones, t: float, equipo: str, direccion: int, p: ParametrosVar) -> dict:
    """Revisión manual: la línea en el instante elegido (el cuadro del pase) y el margen de cada
    atacante. Como hace el operador del VAR, sin depender de que la app detecte el pase."""
    ins = pos.instante(t)
    rev = revisar_fuera_de_lugar(ins, equipo, direccion, p)
    if rev.get("linea_x") is None:              # faltan defensores o atacantes a la vista
        return rev
    # el atacante más adelantado en campo rival hace de "receptor" para el dibujo
    j = ins.jugadores[(ins.jugadores["equipo"] == equipo)]
    if not len(j):
        return rev
    k = int(np.argmax(j["x"].to_numpy() * direccion))
    rev2 = revisar_fuera_de_lugar(ins, equipo, direccion, p, None, j["id"].iloc[k])
    rev2["manual"] = True
    rev2["adelantados"] = rev.get("adelantados", [])
    n = len(rev2["adelantados"]) + (1 if rev2.get("veredicto") == "fuera de lugar" else 0)
    rev2["explicacion"] = (f"Atacante más adelantado: {rev2.get('explicacion', '')}"
                           + (f" Hay {n} atacante(s) en posición adelantada; es infracción solo si interviene." if n else ""))
    return rev2
