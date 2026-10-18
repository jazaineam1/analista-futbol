"""Seguimiento de jugadores y balón en coordenadas de cancha (metros).

Se sigue en metros y no en píxeles porque la cámara de transmisión se mueve todo
el tiempo: en la imagen los jugadores "saltan" con cada paneo, en la cancha no.
Asociación en dos etapas al estilo ByteTrack (detecciones confiables primero,
luego las dudosas), con filtro de Kalman de velocidad constante.

Los jugadores que salen de cuadro se mantienen un tiempo como "fuera de cuadro"
(posición extrapolada y frenada), para recuperar su identidad cuando la cámara vuelve.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Pista:
    id: int
    estado: np.ndarray            # [x, y, vx, vy]
    P: np.ndarray
    ultimo_visto: float           # segundos
    golpes: int = 1
    votos_equipo: Counter = field(default_factory=Counter)
    confirmada: bool = False
    golpes_req: int = 3           # golpes para confirmarla (más si nació de una detección dudosa)

    @property
    def xy(self) -> np.ndarray:
        return self.estado[:2]

    def equipo(self, minimo: int = 3) -> str | None:
        if not self.votos_equipo:
            return None
        eq, n = self.votos_equipo.most_common(1)[0]
        return eq if n >= minimo else None


class SeguidorCancha:
    def __init__(self, conf_alta: float = 0.45, puerta_base: float = 2.2, vel_max: float = 8.0,
                 puerta_max: float = 9.0, perdida_visible_s: float = 1.2, perdida_fuera_s: float = 8.0,
                 golpes_confirmar: int = 3, ruido_medicion: float = 0.7, ruido_acel: float = 4.0,
                 conf_nueva: float = 0.0):
        self.conf_alta = conf_alta
        # En planos generales los jugadores miden ~20 px y la red rara vez pasa de 0,45 de confianza:
        # también se abren pistas con detecciones dudosas, pero deben repetirse más para confirmarse
        self.conf_nueva = conf_nueva
        self.puerta_base, self.vel_max, self.puerta_max = puerta_base, vel_max, puerta_max
        self.perdida_visible_s, self.perdida_fuera_s = perdida_visible_s, perdida_fuera_s
        self.golpes_confirmar = golpes_confirmar
        self.R = np.eye(2) * ruido_medicion ** 2
        self.q = ruido_acel
        self.pistas: list[Pista] = []
        self.siguiente_id = 1
        self.t_prev: float | None = None

    # ------------------------------------------------------------ Kalman
    def _predecir(self, dt: float):
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([[dt ** 2 / 2, 0], [0, dt ** 2 / 2], [dt, 0], [0, dt]])
        Q = G @ G.T * self.q ** 2
        for p in self.pistas:
            p.estado = F @ p.estado
            p.P = F @ p.P @ F.T + Q
            # Si lleva tiempo sin verse, frenar la extrapolación
            if self.t_prev is not None and (self.t_prev - p.ultimo_visto) > 0.5:
                p.estado[2:] *= 0.85
            v = np.linalg.norm(p.estado[2:])
            if v > self.vel_max:
                p.estado[2:] *= self.vel_max / v

    def _corregir(self, p: Pista, z: np.ndarray):
        Hm = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        S = Hm @ p.P @ Hm.T + self.R
        K = p.P @ Hm.T @ np.linalg.inv(S)
        p.estado = p.estado + K @ (z - Hm @ p.estado)
        p.P = (np.eye(4) - K @ Hm) @ p.P

    # ------------------------------------------------------------ asociación
    def _costos(self, pistas, pos, equipos, t):
        C = np.full((len(pistas), len(pos)), 1e6)
        for i, p in enumerate(pistas):
            sin_ver = max(0.0, t - p.ultimo_visto)
            puerta = min(self.puerta_max, self.puerta_base + 1.5 * sin_ver)
            d = np.linalg.norm(pos - p.xy, axis=1)
            eq_p = p.equipo()
            for j in range(len(pos)):
                if d[j] > puerta:
                    continue
                if eq_p and equipos[j] in ("A", "B") and eq_p in ("A", "B") and equipos[j] != eq_p:
                    continue
                C[i, j] = d[j] + 0.4 * sin_ver
        return C

    def _asignar(self, pistas, idx_det, pos, equipos, t):
        if not pistas or not idx_det:
            return [], list(range(len(pistas))), list(idx_det)
        C = self._costos(pistas, pos[idx_det], [equipos[k] for k in idx_det], t)
        fil, col = linear_sum_assignment(C)
        pares, usadas_p, usadas_d = [], set(), set()
        for i, j in zip(fil, col):
            if C[i, j] < 1e5:
                pares.append((i, idx_det[j]))
                usadas_p.add(i)
                usadas_d.add(idx_det[j])
        libres_p = [i for i in range(len(pistas)) if i not in usadas_p]
        libres_d = [k for k in idx_det if k not in usadas_d]
        return pares, libres_p, libres_d

    # ------------------------------------------------------------ paso
    def actualizar(self, t: float, pos: np.ndarray, conf: np.ndarray, equipos: list,
                   huella: np.ndarray | None = None) -> list[tuple[int, int]]:
        """Recibe detecciones en metros. Devuelve pares (id_pista, índice_detección)."""
        dt = 0.04 if self.t_prev is None else max(1e-3, t - self.t_prev)
        self._predecir(dt)
        pos = np.asarray(pos, float).reshape(-1, 2)
        conf = np.asarray(conf, float)
        altas = [k for k in range(len(pos)) if conf[k] >= self.conf_alta]
        bajas = [k for k in range(len(pos)) if conf[k] < self.conf_alta]

        # Etapa 1: pistas confirmadas (ordenadas por recencia) con detecciones confiables
        orden = sorted(range(len(self.pistas)), key=lambda i: (not self.pistas[i].confirmada,
                                                                 t - self.pistas[i].ultimo_visto))
        pistas = [self.pistas[i] for i in orden]
        pares1, libres_p, libres_d = self._asignar(pistas, altas, pos, equipos, t)
        # Etapa 2: pistas restantes con detecciones dudosas
        resto = [pistas[i] for i in libres_p]
        pares2, _, libres_bajas = self._asignar(resto, bajas, pos, equipos, t)

        salida = []
        for lista, pares in ((pistas, pares1), (resto, pares2)):
            for i, k in pares:
                p = lista[i]
                self._corregir(p, pos[k])
                p.ultimo_visto = t
                p.golpes += 1
                if equipos[k] is not None:
                    p.votos_equipo[equipos[k]] += 1
                if p.golpes >= p.golpes_req:
                    p.confirmada = True
                if p.confirmada:
                    salida.append((p.id, k))

        # Nuevas pistas con detecciones confiables no asignadas (evitando duplicados)
        asignadas = {k for _, k in pares1} | {k for _, k in pares2}
        nuevas = [(k, self.golpes_confirmar) for k in libres_d] + \
                 [(k, self.golpes_confirmar + 3) for k in libres_bajas if conf[k] >= self.conf_nueva]
        for k, req in nuevas:
            if k in asignadas:
                continue
            if any(np.linalg.norm(pos[k] - pos[j]) < 0.8 for j in asignadas):
                continue
            p = Pista(self.siguiente_id, np.array([pos[k][0], pos[k][1], 0.0, 0.0]),
                      np.diag([1.0, 1.0, 4.0, 4.0]), ultimo_visto=t, golpes_req=req)
            if equipos[k] is not None:
                p.votos_equipo[equipos[k]] += 1
            self.siguiente_id += 1
            self.pistas.append(p)
            asignadas.add(k)

        # Borrado: rápido si debería verse y no aparece, lento si está fuera de cuadro
        vivas = []
        for p in self.pistas:
            sin_ver = t - p.ultimo_visto
            visible = huella is None or cv2.pointPolygonTest(
                huella.astype(np.float32), (float(p.xy[0]), float(p.xy[1])), False) >= 0
            limite = self.perdida_visible_s if visible else self.perdida_fuera_s
            if not p.confirmada:
                limite = min(limite, 0.4)
            if sin_ver <= limite:
                vivas.append(p)
        self.pistas = vivas
        self.t_prev = t
        return salida

    def fuera_de_cuadro(self, t: float) -> list[Pista]:
        return [p for p in self.pistas if p.confirmada and t - p.ultimo_visto > 1e-6]


class SeguidorBalon:
    """Seguimiento del balón en metros, robusto a falsos positivos.

    - Con balón seguido: se predice con velocidad constante y solo se aceptan candidatos
      dentro de una puerta que crece con el tiempo sin verlo.
    - Perdido: no se salta al primer punto blanco; un candidato tiene que repetirse en
      varios cuadros con movimiento posible (y mejor si hay jugadores cerca) para adoptarse.
    """

    def __init__(self, puerta: float = 2.5, vel_max: float = 35.0, perdido_s: float = 0.6,
                 golpes_adoptar: int = 3, radio_jugador: float = 4.0):
        self.puerta, self.vel_max, self.perdido_s = puerta, vel_max, perdido_s
        self.golpes_adoptar, self.radio_jugador = golpes_adoptar, radio_jugador
        self.xy: np.ndarray | None = None
        self.v = np.zeros(2)
        self.t: float | None = None
        self.hipotesis: list[dict] = []     # en modo búsqueda

    @property
    def seguido(self) -> bool:
        return self.xy is not None

    def _cerca_de_jugadores(self, c, cerca_de):
        if cerca_de is None or len(cerca_de) == 0:
            return np.full(len(c), np.inf)
        return np.min(np.linalg.norm(c[:, None] - np.asarray(cerca_de)[None], axis=2), axis=1)

    def actualizar(self, t: float, candidatos: np.ndarray, conf: np.ndarray,
                   cerca_de: np.ndarray | None = None) -> int | None:
        c = np.asarray(candidatos, float).reshape(-1, 2)
        conf = np.asarray(conf, float).reshape(-1)
        dj = self._cerca_de_jugadores(c, cerca_de) if len(c) else np.zeros(0)

        # --- modo seguimiento
        if self.xy is not None:
            dt = max(1e-3, t - self.t)
            pred = self.xy + self.v * min(dt, 1.0)
            if len(c):
                puerta = self.puerta + self.vel_max * dt
                d = np.linalg.norm(c - pred, axis=1)
                score = np.where(d <= puerta, conf - 0.25 * d / puerta + 0.1 * (dj <= self.radio_jugador), -np.inf)
                k = int(np.argmax(score))
                if np.isfinite(score[k]):
                    v_nueva = (c[k] - self.xy) / dt
                    vel = np.linalg.norm(v_nueva)
                    if vel > self.vel_max:
                        v_nueva *= self.vel_max / vel
                    self.v = 0.5 * self.v + 0.5 * v_nueva
                    self.xy, self.t = c[k], t
                    return k
            if t - self.t > self.perdido_s:          # perdido: pasar a búsqueda
                self.xy, self.v, self.hipotesis = None, np.zeros(2), []
            return None

        # --- modo búsqueda: acumular hipótesis consistentes en el tiempo
        if len(c) == 0:
            self.hipotesis = [h for h in self.hipotesis if t - h["t"] <= 0.5]
            return None
        usados = set()
        for h in self.hipotesis:
            dt = max(1e-3, t - h["t"])
            d = np.linalg.norm(c - h["xy"], axis=1)
            d[list(usados)] = np.inf
            k = int(np.argmin(d))
            if d[k] <= self.puerta + self.vel_max * dt:
                h["v"] = (c[k] - h["xy"]) / dt
                h["xy"], h["t"], h["k"] = c[k], t, k
                h["golpes"] += 1
                h["score"] += conf[k] + (0.3 if dj[k] <= self.radio_jugador else -0.2)
                usados.add(k)
        for k in range(len(c)):
            if k not in usados:
                self.hipotesis.append({"xy": c[k], "t": t, "k": k, "v": np.zeros(2), "golpes": 1,
                                       "score": conf[k] + (0.3 if dj[k] <= self.radio_jugador else -0.2)})
        self.hipotesis = [h for h in self.hipotesis if t - h["t"] <= 0.5]
        listas = [h for h in self.hipotesis if h["t"] == t and h["golpes"] >= self.golpes_adoptar
                  and h["score"] / h["golpes"] > 0.25]
        if not listas:
            return None
        mejor = max(listas, key=lambda h: h["score"])
        self.xy, self.v, self.t = mejor["xy"], mejor["v"], t
        self.hipotesis = []
        return int(mejor["k"])
