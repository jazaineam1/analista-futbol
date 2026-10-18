"""Seguimiento de la cámara: homografía imagen -> cancha cuadro a cuadro.

Estrategia:
  1. Calibración manual en un cuadro clave, con ajuste fino automático contra las
     líneas blancas detectadas (corrige la imprecisión de los clics).
  2. Propagación: se mide el movimiento de cámara entre cuadros con flujo óptico
     sobre el césped (sin jugadores) y se encadena la homografía.
  3. Tras un corte (repetición, primer plano), cuando vuelve el plano general se
     re-ancla comparando la imagen con cuadros de referencia ya calibrados (puntos ORB
     sobre el césped y las líneas -> homografía entre imágenes): la cámara de TV gira
     sobre un punto fijo, así que dos vistas suyas se relacionan por una homografía.
     Sirve también en vivo: se calibra un cuadro congelado y, al empezar, la app
     encuentra dónde está ahora la cámara. Si no hay coincidencia, se alinean las
     líneas con la última calibración buena (ECC).
  4. Si el plano no es táctico (poco césped), el cuadro se marca sin calibración.

En producción, los pasos 1 y 3 se reemplazan por un modelo de puntos clave de la
cancha (p. ej. los de SoccerNet Camera Calibration) que calibra cada cuadro solo.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from ..cancha import Cancha, aplicar_h, mascara_lineas_proyectadas
from .deteccion import mascara_cajas, mascara_cesped, mascara_lineas


@dataclass
class EstadoCamara:
    H_img2cancha: np.ndarray | None   # None = cuadro sin calibración (no táctico)
    calidad: float                    # 0..1 coincidencia de líneas proyectadas vs detectadas
    corte: bool                       # cambio de plano detectado
    fraccion_cesped: float
    movimiento: np.ndarray | None     # homografía cuadro anterior -> actual
    fuente: str                       # 'clave', 'propagada', 'refinada', 'reanclada', 'fija', 'perdida'

    @property
    def calibrado(self) -> bool:
        return self.H_img2cancha is not None

    def huella(self, forma) -> np.ndarray | None:
        """Polígono de la cancha visible en el cuadro (para el minimapa)."""
        if self.H_img2cancha is None:
            return None
        return huella_camara(self.H_img2cancha, forma)


def huella_camara(H_img2cancha, forma, limite=(70.0, 50.0)) -> np.ndarray | None:
    """Región de cancha que se ve: grilla de la imagen proyectada (sin cielo ni horizonte)."""
    h, w = forma[:2]
    uu, vv = np.meshgrid(np.linspace(0, w, 24), np.linspace(0, h, 14))
    H = np.asarray(H_img2cancha, float)
    signo = np.sign((H @ np.array([w / 2, h * 0.9, 1.0]))[2]) or 1.0  # abajo al centro = cancha
    pts = np.stack([uu.ravel(), vv.ravel(), np.ones(uu.size)], 1) @ (H * signo).T
    ok = pts[:, 2] > 1e-9
    xy = pts[ok, :2] / pts[ok, 2:3]
    xy = xy[(np.abs(xy[:, 0]) < limite[0]) & (np.abs(xy[:, 1]) < limite[1])]
    if len(xy) < 3:
        return None
    return cv2.convexHull(xy.astype(np.float32)).reshape(-1, 2).astype(float)


class EstimadorCamara:
    def __init__(self, cancha: Cancha, claves: dict[int, np.ndarray] | None = None,
                 refinar_cada: int = 0, calidad_min: float = 0.0, umbral_corte: float = 0.55,
                 umbral_reanclaje: float = 0.3, movimiento_refinado: int = cv2.MOTION_EUCLIDEAN,
                 cambio_max_refinado: float = 2.5, escala_ecc: float = 0.5, fondo_completo: bool = False):
        self.cancha = cancha
        # La cámara de TV gira sobre un punto fijo (paneo, inclinación, zoom): entre dos cuadros toda la
        # imagen, tribuna incluida, se mueve con la misma homografía. fondo_completo=True sigue también la
        # tribuna (más textura); por defecto solo el césped, que es lo validado. Siempre se excluyen los
        # jugadores (se mueven solos) y los gráficos de TV (quietos).
        self.fondo_completo = fondo_completo
        self.claves = {int(k): np.asarray(v, float) for k, v in (claves or {}).items()}
        # refinar_cada = 0: sin corrección por líneas durante la propagación. En las pruebas con
        # verdad conocida el flujo óptico solo derivó ~1 m en 35 s, y la alineación de líneas
        # (ECC) empeoraba el resultado cuando se ven pocas líneas. Se usa para re-anclar tras cortes.
        self.refinar_cada = refinar_cada
        self.umbral_reanclaje = umbral_reanclaje
        self.movimiento_refinado = movimiento_refinado
        self.cambio_max_refinado = cambio_max_refinado
        self.calidad_min = calidad_min
        self.umbral_corte = umbral_corte
        self.escala_ecc = escala_ecc
        self.H: np.ndarray | None = None
        self.ultima_buena: np.ndarray | None = None
        self.gris_prev = None
        self.hist_prev = None
        self.n = 0
        self.malos_seguidos = 0
        self.flujo_fallido = 0
        self.ultimo_flujo = None
        # Cuadros de referencia calibrados: fijos (claves) y recientes (vistas buenas del partido)
        self.escala_ref = 0.5
        self.orb = cv2.ORB_create(nfeatures=1500, fastThreshold=10)
        self.refs_fijas: list[dict] = []
        self.refs_recientes: list[dict] = []
        # En vivo: si entre dos cuadros analizados pasó mucho tiempo (el PC no alcanza el ritmo) o el
        # calce se pierde, el flujo óptico ya no sirve y se busca la cámara con los cuadros de referencia
        self.reanclar_si_malo = False
        self.salto_max_s = 0.45
        self.ultima_ref = -10 ** 9
        self.claves_con_ref: set[int] = set()

    # ----------------------------------------------------------- API principal
    # ----------------------------------------------------------- referencias
    def _rasgos(self, frame_o_gris, cesped=None, graficos=None, excluir=None):
        gris = frame_o_gris if frame_o_gris.ndim == 2 else cv2.cvtColor(frame_o_gris, cv2.COLOR_BGR2GRAY)
        if cesped is None:
            cesped, _ = mascara_cesped(frame_o_gris) if frame_o_gris.ndim == 3 else (np.full(gris.shape, 255, np.uint8), None)
        # césped y líneas (las líneas quedan dentro del césped dilatado); fuera gráficos y jugadores
        m = cv2.dilate((cesped > 0).astype(np.uint8) * 255, np.ones((9, 9), np.uint8))
        if graficos is not None:
            m[graficos > 0] = 0
        if excluir is not None:
            m[excluir > 0] = 0
        s = self.escala_ref
        g = cv2.resize(gris, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        return self.orb.detectAndCompute(g, m)

    def agregar_referencia(self, frame: np.ndarray, H_img2cancha: np.ndarray, cesped=None, fija: bool = False) -> bool:
        """Guarda un cuadro con su calibración para volver a ubicar la cámara más adelante."""
        kp, des = self._rasgos(frame, cesped)
        if des is None or len(kp) < 40:
            return False
        ref = {"pts": np.float32([k.pt for k in kp]), "des": des, "H": np.asarray(H_img2cancha, float).copy()}
        if fija:
            self.refs_fijas.append(ref)
        else:
            self.refs_recientes = (self.refs_recientes + [ref])[-4:]
        return True

    def desde_referencias(self, gris, cesped, graficos, excluir, lineas, campo, forma):
        """Homografía del cuadro actual a partir de las referencias (None si ninguna coincide)."""
        refs = self.refs_recientes[::-1] + self.refs_fijas
        if not refs:
            return None
        kp, des = self._rasgos(gris, cesped, graficos, excluir)
        if des is None or len(kp) < 30:
            return None
        pts = np.float32([k.pt for k in kp])
        bf = cv2.BFMatcher(cv2.NORM_HAMMING)
        S = np.diag([self.escala_ref, self.escala_ref, 1.0])
        mejor, mejor_puntaje = None, 0.0
        for ref in refs:
            pares = bf.knnMatch(ref["des"], des, k=2)
            buenos = [m for m in (p for p in pares if len(p) == 2) if m[0].distance < 0.8 * m[1].distance]
            if len(buenos) < 25:
                continue
            a = ref["pts"][[m[0].queryIdx for m in buenos]]
            b = pts[[m[0].trainIdx for m in buenos]]
            M, inl = cv2.findHomography(a, b, cv2.RANSAC, 3.0)
            if M is None or inl is None:
                continue
            n_in = int(inl.sum())
            if n_in < 25 or n_in < 0.3 * len(buenos):
                continue
            M = np.linalg.inv(S) @ M @ S                      # ref -> actual, en píxeles de proceso
            try:
                H = ref["H"] @ np.linalg.inv(M)
            except np.linalg.LinAlgError:
                continue
            H = H / H[2, 2]
            cal = self.calidad(H, lineas, campo) if lineas is not None else 0.0
            puntaje = n_in * (0.5 + cal)
            if puntaje > mejor_puntaje:
                mejor, mejor_puntaje = (H, cal, n_in), puntaje
        if mejor is None:
            return None
        H, cal, n_in = mejor
        if lineas is not None:
            H = self.ajustar_con_lineas(H, lineas, campo, forma, cambio_max_m=2.0)
        return H

    def procesar(self, idx: int, frame: np.ndarray, cesped: np.ndarray, campo: np.ndarray,
                 cajas_jugadores: np.ndarray, graficos: np.ndarray | None = None,
                 salto_s: float | None = None) -> EstadoCamara:
        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = self._histograma(frame)
        frac = float((cesped > 0).mean())
        corte = self.hist_prev is not None and \
            cv2.compareHist(self.hist_prev, hist, cv2.HISTCMP_CORREL) < self.umbral_corte
        plano_general = frac > 0.25

        excluir = mascara_cajas(frame.shape, cajas_jugadores, holgura=0.25)
        lineas = mascara_lineas(frame, campo, excluir) if plano_general else None
        mov = None
        fuente = "propagada"

        if idx in self.claves:
            self.H = self.ajustar_con_lineas(self.claves[idx].copy(), lineas, campo, frame.shape)
            fuente = "clave"
            if idx not in self.claves_con_ref:
                self.claves_con_ref.add(idx)
                self.agregar_referencia(frame, self.H, cesped, fija=True)
        elif not plano_general:
            self.H, fuente = None, "perdida"
        elif corte or self.H is None:
            H_ref = self.desde_referencias(gris, cesped, graficos, excluir, lineas, campo, frame.shape)
            if H_ref is not None:
                self.H, fuente = H_ref, "reanclada"
                self.flujo_fallido = self.malos_seguidos = 0
            else:
                self.H, fuente = self._reanclar(lineas, frame.shape)
        else:
            H_ref = None
            hay_refs = bool(self.refs_fijas or self.refs_recientes)
            salto = salto_s is not None and salto_s > self.salto_max_s
            if self.reanclar_si_malo and hay_refs and (salto or self.malos_seguidos % 8 == 3):
                H_ref = self.desde_referencias(gris, cesped, graficos, excluir, lineas, campo, frame.shape)
                if H_ref is not None and not salto and lineas is not None and self.H is not None:
                    # sin salto de tiempo, solo si calza mejor que lo que viene siguiendo el flujo
                    if self.calidad(H_ref, lineas, campo) <= self.calidad(self.H, lineas, campo) + 0.05:
                        H_ref = None
            if H_ref is not None:
                self.H, fuente = H_ref, "reanclada"
                self.flujo_fallido = self.malos_seguidos = 0
            else:
                mov = self._movimiento(self.gris_prev, gris, cesped, excluir, graficos)
                if mov is not None:
                    self.H = self.H @ np.linalg.inv(mov)
                    self.H /= self.H[2, 2]
                    self.flujo_fallido = 0
                else:
                    self.flujo_fallido += 1

        calidad = 0.0
        if self.H is not None and lineas is not None:
            calidad = self.calidad(self.H, lineas, campo)
            # Corrección con las líneas: solo si mejora claramente el calce y el cambio es moderado.
            # (El flujo óptico es muy estable cuadro a cuadro; una corrección mal convergida es peor
            # que la pequeña deriva que intenta arreglar.)
            if fuente != "clave" and ((self.refinar_cada and self.n % self.refinar_cada == 0)
                                      or calidad < self.calidad_min):
                H_ref = self._refinar(self.H, lineas, frame.shape,
                                      iteraciones=60 if calidad >= self.calidad_min else 120,
                                      movimiento=self.movimiento_refinado)
                if H_ref is not None and self._cambio_m(self.H, H_ref, frame.shape) < self.cambio_max_refinado:
                    cal_ref = self.calidad(H_ref, lineas, campo)
                    if cal_ref > calidad + 0.03:
                        self.H, calidad, fuente = H_ref, cal_ref, "refinada"
            if calidad >= self.calidad_min:
                self.malos_seguidos = 0
                self.ultima_buena = self.H.copy()
            elif fuente != "clave":
                self.malos_seguidos += 1
            # Cada ~2 s, una vista bien calibrada queda como referencia para después de un corte
            # (en vivo, con pocos cuadros por segundo, más seguido: la cámara se mueve mucho entre cuadros)
            cada = 3 if self.reanclar_si_malo else 50
            if calidad >= (0.45 if self.reanclar_si_malo else 0.3) and fuente != "perdida" and self.n - self.ultima_ref >= cada:
                if self.agregar_referencia(frame, self.H, cesped):
                    self.ultima_ref = self.n
        # Se declara perdida solo con evidencia fuerte: el flujo falla varios cuadros seguidos
        # o las líneas no calzan durante ~2 s (en zonas sin líneas, el flujo sigue mandando)
        if self.H is not None and fuente != "clave" and (self.flujo_fallido >= 6 or self.malos_seguidos >= 50):
            self.H, fuente = None, "perdida"
            self.malos_seguidos = self.flujo_fallido = 0

        self.gris_prev, self.hist_prev = gris, hist
        self.n += 1
        return EstadoCamara(None if self.H is None else self.H.copy(), calidad, bool(corte), frac, mov,
                            fuente if self.H is not None else "perdida")

    # ----------------------------------------------------------- componentes
    @staticmethod
    def _histograma(frame):
        hsv = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [30, 16], [0, 180, 0, 256])
        return cv2.normalize(h, h).flatten()

    def _movimiento(self, prev, cur, cesped, excluir, graficos=None) -> np.ndarray | None:
        """Homografía prev -> cur medida con flujo óptico sobre el fondo (o solo el césped)."""
        if prev is None:
            return None
        fondo = np.ones_like(excluir, bool) if self.fondo_completo else cesped > 0
        mascara = (fondo & (excluir == 0)).astype(np.uint8) * 255
        if graficos is not None:
            mascara[graficos > 0] = 0
        mascara = cv2.erode(mascara, np.ones((5, 5), np.uint8))
        p0 = cv2.goodFeaturesToTrack(prev, maxCorners=800, qualityLevel=0.005, minDistance=12,
                                     mask=mascara, blockSize=7)
        if p0 is None or len(p0) < 20:
            return None
        p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
        ok = st.ravel() == 1
        if ok.sum() < 20:
            return None
        M, inl = cv2.findHomography(p0[ok], p1[ok], cv2.RANSAC, 2.0)
        # Para la vista "paso a paso": puntos seguidos y cuáles se aceptaron como fondo
        self.ultimo_flujo = (p0[ok].reshape(-1, 2), p1[ok].reshape(-1, 2),
                             None if inl is None else inl.ravel().astype(bool))
        if M is None or inl.sum() < 15:
            return None
        return M / M[2, 2]

    def _refinar(self, H_img2cancha, lineas, forma, iteraciones: int = 60,
                 movimiento: int = cv2.MOTION_HOMOGRAPHY) -> np.ndarray | None:
        """Corrige la homografía alineando líneas proyectadas con las detectadas (ECC)."""
        if H_img2cancha is None or lineas is None or lineas.sum() < 255 * 150:
            return None
        s = self.escala_ecc
        try:
            H_c2i = np.linalg.inv(H_img2cancha)
        except np.linalg.LinAlgError:
            return None
        S = np.diag([s, s, 1.0])
        h, w = forma[:2]
        hs, ws = int(h * s), int(w * s)
        proy = mascara_lineas_proyectadas(S @ H_c2i, (hs, ws), self.cancha, grosor=2)
        if proy.sum() < 255 * 80:
            return None
        det = cv2.resize(lineas, (ws, hs), interpolation=cv2.INTER_AREA)
        plantilla = cv2.GaussianBlur(det.astype(np.float32) / 255, (0, 0), 3)
        entrada = cv2.GaussianBlur(proy.astype(np.float32) / 255, (0, 0), 3)
        W = np.eye(3 if movimiento == cv2.MOTION_HOMOGRAPHY else 2, 3, dtype=np.float32)
        try:
            _, W = cv2.findTransformECC(plantilla, entrada, W, movimiento,
                                        (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iteraciones, 1e-5),
                                        None, 5)
        except cv2.error:
            return None
        if W.shape[0] == 2:
            W = np.vstack([W, [0, 0, 1]])
        W_full = np.linalg.inv(S) @ W.astype(np.float64) @ S
        # Rechazar correcciones exageradas (señal de mala convergencia)
        esquinas = np.array([[0, 0], [w, 0], [w, h], [0, h]], float)
        desplazamiento = np.linalg.norm(aplicar_h(W_full, esquinas) - esquinas, axis=1).max()
        if not np.isfinite(desplazamiento) or desplazamiento > 0.25 * w:
            return None
        H_new = H_img2cancha @ W_full
        return H_new / H_new[2, 2]

    def ajustar_con_lineas(self, H, lineas, campo, forma, cambio_max_m: float = 3.0):
        """Ajuste fino de una calibración manual: alinea la cancha teórica con las líneas
        blancas detectadas. Se acepta solo si el calce mejora y el cambio es moderado
        (si no, queda la calibración original)."""
        if lineas is None:
            return H
        H_ref = self._refinar(H, lineas, forma, iteraciones=150)
        if H_ref is None or self._cambio_m(H, H_ref, forma) > cambio_max_m:
            return H
        return H_ref if self.calidad(H_ref, lineas, campo) > self.calidad(H, lineas, campo) + 0.03 else H

    def _cambio_m(self, H_a, H_b, forma) -> float:
        """Cuánto cambia (m, mediana) la proyección de una grilla de la imagen entre dos homografías."""
        h, w = forma[:2]
        uu, vv = np.meshgrid(np.linspace(0.1 * w, 0.9 * w, 6), np.linspace(0.4 * h, 0.95 * h, 4))
        P = np.stack([uu.ravel(), vv.ravel()], 1)
        a, b = aplicar_h(H_a, P), aplicar_h(H_b, P)
        ok = np.isfinite(a).all(1) & np.isfinite(b).all(1) & (np.abs(a) < 200).all(1)
        return float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1))) if ok.any() else np.inf

    def _reanclar(self, lineas, forma):
        """Tras un corte: probar la última calibración buena y las claves conocidas."""
        candidatas = []
        if self.ultima_buena is not None:
            candidatas.append(self.ultima_buena)
        candidatas += list(self.claves.values())
        mejor, mejor_cal = None, 0.0
        for H0 in candidatas:
            H1 = self._refinar(H0, lineas, forma, iteraciones=150)
            for H in (H1, H0):
                if H is None:
                    continue
                cal = self.calidad(H, lineas, None)
                if cal > mejor_cal:
                    mejor, mejor_cal = H, cal
        if mejor is not None and mejor_cal >= self.umbral_reanclaje:
            return mejor, "reanclada"
        return None, "perdida"

    def calidad(self, H_img2cancha, lineas, campo) -> float:
        """F1 entre líneas detectadas y líneas teóricas proyectadas (tolerancia 4 px)."""
        if H_img2cancha is None or lineas is None or lineas.sum() == 0:
            return 0.0
        try:
            H_c2i = np.linalg.inv(H_img2cancha)
        except np.linalg.LinAlgError:
            return 0.0
        proy = mascara_lineas_proyectadas(H_c2i, lineas.shape, self.cancha, grosor=1)
        if campo is not None:
            proy[campo == 0] = 0
        if proy.sum() == 0:
            return 0.0
        k = np.ones((9, 9), np.uint8)
        det_d = cv2.dilate(lineas, k) > 0
        proy_d = cv2.dilate(proy, k) > 0
        recall = float((det_d & (proy > 0)).sum()) / max(1, int((proy > 0).sum()))
        precision = float((proy_d & (lineas > 0)).sum()) / max(1, int((lineas > 0).sum()))
        return 0.0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)

    def alto_esperado(self, H_img2cancha, estatura: float = 1.8):
        """Función que estima la altura en píxeles de un jugador parado en (u, v)."""
        if H_img2cancha is None:
            return None
        H_c2i = np.linalg.inv(H_img2cancha)

        def f(pie):
            p = aplicar_h(H_img2cancha, [pie])[0]
            a, b = aplicar_h(H_c2i, [[p[0] - 0.5, p[1]], [p[0] + 0.5, p[1]]])
            escala = max(np.linalg.norm(b - a), 1e-3)  # px por metro en dirección horizontal
            return float(np.clip(estatura * escala * 1.05, 4, 2000))
        return f
