"""Orquestación: video -> detecciones -> cámara -> equipos -> seguimiento -> registros."""
from __future__ import annotations

import json
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ..cancha import LienzoCancha, aplicar_h, mascara_lineas_proyectadas
from .calibracion import Calibracion
from .camara import EstadoCamara, EstimadorCamara
from .deteccion import Detecciones, DetectorCesped, DetectorYOLO, mascara_cesped
from .equipos import ClasificadorEquipos
from .minimapa import Minimapa, anotar_cuadro, esquina_libre, hex_a_bgr
from .seguimiento import SeguidorBalon, SeguidorCancha


@dataclass
class ConfigPipeline:
    detector: str = "yolo"              # 'yolo' o 'cesped'
    modelo_yolo: str = "yolo11n.pt"
    imgsz: int = 1280
    conf: float | None = None          # None = automática: 0,03 a ≥1600 px (jugadores pequeños), si no 0,05
    ancho_proceso: int = 1280           # se reescala el video a este ancho
    paso: int = 1                       # procesar 1 de cada `paso` cuadros
    equipos: dict[str, str] = field(default_factory=dict)   # {"Nombre A": "#hex", "Nombre B": "#hex"}
    camara_fija: bool | None = None     # None = lo que diga la calibración
    minimapa: bool = True


@dataclass
class ResultadoCuadro:
    indice: int
    t: float
    frame: np.ndarray
    camara: EstadoCamara
    jugadores: list[dict]
    balon: dict | None
    fuera: list[dict]


class DetectorGraficos:
    """Detecta gráficos sobreimpresos (marcador, reloj, logos, marcas de agua).

    En una transmisión la cámara se mueve casi siempre; lo que no cambia en pantalla
    durante ~2 s mientras todo lo demás se mueve es un gráfico, no parte de la cancha.
    """

    def __init__(self, n: int = 40, escala: float = 0.25, umbral: int = 7):
        self.n, self.escala, self.umbral = n, escala, umbral
        self.buf: list[np.ndarray] = []
        self.mascara: np.ndarray | None = None
        self.forma = None

    def actualizar(self, frame: np.ndarray) -> np.ndarray | None:
        if self.mascara is not None and self.forma == frame.shape[:2]:
            return self.mascara
        self.forma = frame.shape[:2]
        g = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), None, fx=self.escala, fy=self.escala,
                       interpolation=cv2.INTER_AREA)
        self.buf.append(g.astype(np.int16))
        if len(self.buf) < self.n:
            return None
        pila = np.stack(self.buf[-self.n:])
        rango = pila.max(0) - pila.min(0)
        if np.median(rango) < 12:          # la cámara casi no se movió: esperar más
            self.buf = self.buf[-self.n // 2:]
            return None
        quieto = (rango < self.umbral).astype(np.uint8)
        # Solo zonas con textura (texto/logos), no manchas lisas de césped o cielo
        media = pila.mean(0).astype(np.uint8)
        bordes = cv2.dilate((cv2.Canny(media, 60, 160) > 0).astype(np.uint8), np.ones((3, 3), np.uint8))
        cand = cv2.morphologyEx(quieto & bordes, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(cand, 8)
        m = np.zeros_like(cand)
        for i in range(1, n):
            if st[i, cv2.CC_STAT_AREA] >= 12:
                m[lab == i] = 1
        m = cv2.dilate(m, np.ones((5, 5), np.uint8))
        h, w = self.forma
        self.mascara = (cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST) * 255).astype(np.uint8)
        self.buf = []
        return self.mascara


class Pipeline:
    def __init__(self, calibracion: Calibracion, cfg: ConfigPipeline | None = None):
        self.cal = calibracion
        self.cfg = cfg or ConfigPipeline()
        self.cancha = calibracion.cancha
        if self.cfg.detector == "yolo":
            grande = self.cfg.ancho_proceso >= 1600     # plano general: jugadores de ~20 px y balón de ~5 px
            conf = self.cfg.conf if self.cfg.conf is not None else (0.03 if grande else 0.05)
            self.detector = DetectorYOLO(self.cfg.modelo_yolo, imgsz=self.cfg.imgsz, conf=conf,
                                         conf_balon=conf if grande else 0.15)
        else:
            self.detector = DetectorCesped()
        self._balon_clasico = DetectorCesped()
        self.equipos = ClasificadorEquipos(colores=dict(self.cfg.equipos) if self.cfg.equipos else {})
        self.seguidor = SeguidorCancha()
        self.balon = SeguidorBalon()
        self.camara: EstimadorCamara | None = None
        self.lente = None
        self.forma = None
        self.escala_proc = 1.0
        self.registros: list[dict] = []
        self.registros_balon: list[dict] = []
        self.registros_camara: list[dict] = []
        self.lienzo = LienzoCancha(self.cancha, ppm=6)
        self.minimapa: Minimapa | None = None
        self.camara_fija = calibracion.camara_fija if self.cfg.camara_fija is None else self.cfg.camara_fija
        self.graficos = DetectorGraficos()
        self.esquina_minimapa: str | None = None
        self.guardar_diag = False          # True: guarda las etapas internas del último cuadro
        self.diag: dict | None = None
        self._graf = None
        self._t_prev: float | None = None
        self.en_vivo = False                # lo pone correr(); en vivo la cámara se reubica tras saltos de tiempo

    def calentar(self):
        """Una pasada en vacío del detector (carga perezosa de la red, hilos, memoria)."""
        try:
            h = int(round(self.cfg.ancho_proceso * 9 / 16))
            vacio = np.zeros((h, self.cfg.ancho_proceso, 3), np.uint8)
            vacio[:] = (40, 140, 40)
            self.detector(vacio, np.full(vacio.shape[:2], 255, np.uint8))
        except Exception:  # noqa: BLE001  (si falla, solo se pierde la ventaja)
            pass

    def _colores_bgr(self):
        if self.equipos.listo:
            hx = self.equipos.colores_hex()
            if self.cfg.equipos:
                vals = list(self.cfg.equipos.values())
                hx = {"A": vals[0], "B": vals[1] if len(vals) > 1 else hx["B"]}
            return {k: hex_a_bgr(v) for k, v in hx.items()}
        return {}

    def _preparar(self, frame):
        h, w = frame.shape[:2]
        self.escala_proc = self.cfg.ancho_proceso / w
        if w != self.cfg.ancho_proceso:
            s = self.cfg.ancho_proceso / w
            frame = cv2.resize(frame, (self.cfg.ancho_proceso, int(round(h * s))), interpolation=cv2.INTER_AREA)
        if self.forma is None:
            self.forma = frame.shape
            hh, ww = frame.shape[:2]
            self.lente = self.cal.corrector_lente(ww, hh)
            claves = self.cal.homografias(ww, hh)
            self.camara = EstimadorCamara(self.cancha, claves)
            self.camara.reanclar_si_malo = self.en_vivo
            # En vivo la calibración se hizo sobre un cuadro congelado: queda como referencia para
            # encontrar la cámara al empezar (y después de cada corte)
            for img, H in self.cal.referencias(ww, hh):
                if self.lente is not None:
                    img = self.lente(img)
                self.camara.agregar_referencia(img, H, fija=True)
        if self.lente is not None:
            frame = self.lente(frame)
        return frame

    # --------------------------------------------------------------- un cuadro
    def procesar(self, indice: int, t: float, frame: np.ndarray) -> ResultadoCuadro:
        frame = self._preparar(frame)
        cesped, campo = mascara_cesped(frame)
        if self.esquina_minimapa is None:
            # Con cámara fija el minimapa va en la esquina con menos cancha; en TV, abajo a la derecha
            self.esquina_minimapa = esquina_libre(campo) if self.camara_fija else "id"
        cesped_cam, campo_cam = cesped, campo          # la cámara usa las máscaras completas
        if not self.camara_fija:
            # Gráficos de TV (marcador, logos): quietos en pantalla mientras la cámara se mueve.
            # Solo se quitan para detectar jugadores y balón; la cámara necesita todas las líneas
            # (una línea horizontal tampoco cambia en un paneo horizontal y caería en la máscara).
            graf = self.graficos.actualizar(frame)
            self._graf = graf
            if graf is not None:
                campo = campo.copy()
                cesped = cesped.copy()
                campo[graf > 0] = 0
                cesped[graf > 0] = 0
        H_prev = self.camara.H
        # Marcas de la cancha proyectadas (para que el detector de balón no confunda el punto penal)
        excluir = _marcas_proyectadas(H_prev, frame.shape, self.cancha) if H_prev is not None else None
        self._balon_clasico.excluir_balon = excluir
        if isinstance(self.detector, DetectorCesped):
            self.detector.excluir_balon = excluir
            det = self.detector(frame, campo, cesped, self.camara.alto_esperado(H_prev))
        else:
            det = self.detector(frame, campo)
        jug, bal = det.jugadores(), det.balones()
        crudas = getattr(self.detector, "ultimas_crudas", det)
        if not isinstance(self.detector, DetectorCesped) and H_prev is not None:
            # Complemento: el balón es pequeño y la red COCO lo pierde; también se busca como
            # mancha clara redonda y se juntan ambos candidatos (los de la red pesan más)
            extra = self._balon_clasico._balon(frame, campo, jug, self.camara.alto_esperado(H_prev), excluir, cesped)
            # La red también "ve" balones en camisetas y medias blancas: fuera los que caen sobre el
            # cuerpo de un jugador (se deja el borde de abajo, donde está el balón cuando lo conduce)
            if len(bal) and len(jug):
                c = bal.centros
                sobre = np.zeros(len(bal), bool)
                for x1, y1, x2, y2 in jug.cajas:
                    sobre |= (c[:, 0] >= x1 + 0.1 * (x2 - x1)) & (c[:, 0] <= x2 - 0.1 * (x2 - x1)) & \
                             (c[:, 1] >= y1) & (c[:, 1] <= y2 - 0.08 * (y2 - y1))
                bal = Detecciones(bal.cajas[~sobre], bal.confianza[~sobre], bal.es_balon[~sobre])
            bal = Detecciones(np.vstack([bal.cajas, extra.cajas]),
                              np.concatenate([bal.confianza, extra.confianza * 0.6]),
                              np.ones(len(bal) + len(extra), bool))

        if self.camara_fija and self.camara.H is not None:
            estado = EstadoCamara(self.camara.H.copy(), 1.0, False, float((cesped > 0).mean()), None, "fija")
            self.camara.ultima_buena = self.camara.H
        else:
            idx_cam = indice
            if self.camara_fija and self.camara.H is None and self.camara.claves:
                # Cámara fija que empieza en otro cuadro: se usa la calibración clave (con su ajuste fino)
                idx_cam = next(iter(self.camara.claves))
            salto = None if self._t_prev is None else t - self._t_prev
            estado = self.camara.procesar(idx_cam, frame, cesped_cam, campo_cam, jug.cajas,
                                          None if self.camara_fija else self._graf, salto)
        self._t_prev = t

        feats = [ClasificadorEquipos.caracteristica(frame, c, cesped) for c in jug.cajas]
        self.equipos.agregar(feats)
        etiquetas = self.equipos.predecir(feats)

        jugadores, fuera, balon = [], [], None
        pos = np.zeros((0, 2))
        dentro = np.zeros(0, bool)
        idx = np.zeros(0, int)
        pos_b, k_balon, bal_todos = np.zeros((0, 2)), None, bal
        huella = None
        if estado.calibrado:
            H = estado.H_img2cancha
            huella = estado.huella(frame.shape)
            pos = aplicar_h(H, jug.pies) if len(jug) else np.zeros((0, 2))
            dentro = self.cancha.dentro(pos, margen=3.0) if len(pos) else np.zeros(0, bool)
            idx = _suprimir_duplicados(pos, jug.confianza, np.where(dentro)[0], radio=0.7)
            pares = self.seguidor.actualizar(t, pos[idx], jug.confianza[idx], [etiquetas[k] for k in idx], huella)
            for id_pista, k_local in pares:
                k = idx[k_local]
                x1, y1, x2, y2 = jug.cajas[k]
                r = {"cuadro": indice, "t": round(t, 3), "id": id_pista, "equipo_det": etiquetas[k],
                     "x": float(pos[k][0]), "y": float(pos[k][1]), "u": float((x1 + x2) / 2), "v": float(y2),
                     "alto_px": float(y2 - y1), "ancho_px": float(x2 - x1), "conf": float(jug.confianza[k]),
                     "calidad_camara": round(estado.calidad, 3)}
                if feats[k] is not None:
                    r["L"], r["a"], r["b"] = (float(v) for v in feats[k])
                self.registros.append(r)
                jugadores.append(r)
            for p in self.seguidor.fuera_de_cuadro(t):
                fuera.append({"id": p.id, "x": float(p.xy[0]), "y": float(p.xy[1]), "equipo": p.equipo()})
            # Balón
            if len(bal):
                pos_b = aplicar_h(H, bal.pies)
                # Fuera de la cancha: balones de repuesto, recogepelotas. Excepto dentro del arco
                # (detrás de la línea de meta, entre los postes): ahí termina un gol
                en_juego = self.cancha.dentro(pos_b, margen=1.5) | (
                    (np.abs(pos_b[:, 0]) <= self.cancha.L2 + 3.0) & (np.abs(pos_b[:, 1]) <= 5.0))
                bal = Detecciones(bal.cajas[en_juego], bal.confianza[en_juego], bal.es_balon[en_juego])
                pos_b = pos_b[en_juego]
            cerca = pos[idx] if len(idx) else None
            k = self.balon.actualizar(t, pos_b, bal.confianza, cerca)
            k_balon = k
            if k is not None:
                balon = {"cuadro": indice, "t": round(t, 3), "x": float(pos_b[k][0]), "y": float(pos_b[k][1]),
                         "u": float(bal.centros[k][0]), "v": float(bal.centros[k][1])}
                self.registros_balon.append(balon)
            for j in jugadores:
                eq = self.seguidor_equipo(j["id"])
                j["equipo"] = eq or j["equipo_det"]
        reg_cam = {"cuadro": indice, "t": round(t, 3), "calibrado": estado.calibrado,
                   "calidad": round(estado.calidad, 3), "corte": estado.corte,
                   "fuente": estado.fuente, "cesped": round(estado.fraccion_cesped, 3),
                   "huella": None if huella is None else np.round(huella, 2).tolist()}
        if estado.calibrado:
            # Homografía en píxeles del video original (para comparar o reutilizar)
            Ho = estado.H_img2cancha @ np.diag([self.escala_proc, self.escala_proc, 1.0])
            reg_cam.update({f"h{i}": float(v) for i, v in enumerate((Ho / Ho[2, 2]).ravel())})
        self.registros_camara.append(reg_cam)
        if self.guardar_diag:
            self.diag = {
                "indice": indice, "t": t, "frame": frame, "cesped": cesped_cam, "campo": campo_cam,
                "graficos": self._graf, "crudas": crudas, "jug": jug, "pos": pos,
                "dentro": dentro, "idx": idx, "feats": feats, "etiquetas": etiquetas, "estado": estado,
                "flujo": self.camara.ultimo_flujo if not self.camara_fija else None,
                "bal_todos": bal_todos, "bal": bal, "pos_b": pos_b, "k_balon": k_balon,
                "jugadores": jugadores, "balon": balon, "fuera": fuera, "excluir_balon": excluir,
            }
        return ResultadoCuadro(indice, t, frame, estado, jugadores, balon, fuera)

    def seguidor_equipo(self, id_pista):
        for p in self.seguidor.pistas:
            if p.id == id_pista:
                return p.equipo(minimo=2)
        return None

    # --------------------------------------------------------------- anotación
    def anotar(self, res: ResultadoCuadro) -> np.ndarray:
        colores = self._colores_bgr()
        mm = None
        if self.cfg.minimapa:
            if self.minimapa is None or self.minimapa.colores != colores:
                self.minimapa = Minimapa(self.lienzo, colores)
            huella = res.camara.huella(res.frame.shape)
            mm = self.minimapa.dibujar(res.jugadores, None if res.balon is None else (res.balon["x"], res.balon["y"]),
                                       huella, res.fuera)
        estados = {"clave": "calibrada a mano", "propagada": "siguiendo la camara", "refinada": "ajustada con lineas",
                   "reanclada": "re-anclada tras corte", "fija": "camara fija"}
        txt = (f"cancha: {estados.get(res.camara.fuente, res.camara.fuente)}" if res.camara.calibrado
               else "plano no tactico (repeticion / primer plano)")
        bu = None if res.balon is None else (res.balon["u"], res.balon["v"])
        return anotar_cuadro(res.frame, res.jugadores, bu, colores, txt, mm, self.esquina_minimapa)

    # --------------------------------------------------------------- video completo
    def correr(self, fuente, salida: Path, video_anotado: bool = True, mostrar: bool = False,
               max_cuadros: int | None = None, progreso: bool = True, al_cuadro=None, detener=None,
               grabar_entrada: Path | None = None) -> dict:
        """Procesa toda la fuente. `al_cuadro(res, anotado, n)` se llama en cada cuadro procesado
        (lo usa el panel web para el video en vivo); `detener()` permite cortar desde afuera.
        `grabar_entrada`: guarda la señal tal como llega (en vivo, para revisarla o reanalizarla)."""
        salida = Path(salida)
        salida.mkdir(parents=True, exist_ok=True)
        escritor = None
        escritor_entrada = None
        en_vivo = bool(getattr(fuente, "en_vivo", False))
        self.en_vivo = en_vivo
        if en_vivo:
            self.calentar()          # la primera inferencia de la red tarda: mejor antes que en el partido
        if grabar_entrada is not None and hasattr(fuente, "grabar"):
            fuente.grabar(grabar_entrada)          # en vivo graba el hilo de captura, a ritmo real
            grabar_entrada = None
        fps_salida = fuente.fps / self.cfg.paso
        t_primero, escritos = None, 0
        t_ini = time.time()
        n = 0
        try:
            for indice, t, frame in fuente:
                if grabar_entrada is not None:
                    if escritor_entrada is None:
                        Path(grabar_entrada).parent.mkdir(parents=True, exist_ok=True)
                        escritor_entrada = _escritor_h264(Path(grabar_entrada), frame.shape, fuente.fps)
                    escritor_entrada.stdin.write(np.ascontiguousarray(frame).tobytes())
                if indice % self.cfg.paso:
                    continue
                if detener is not None and detener():
                    break
                res = self.procesar(indice, t, frame)
                n += 1
                if video_anotado or mostrar or al_cuadro is not None:
                    anotado = self.anotar(res)
                    if al_cuadro is not None:
                        al_cuadro(res, anotado, n)
                    if video_anotado:
                        if escritor is None:
                            escritor = _escritor_h264(salida / "video_anotado.mp4", anotado.shape, fps_salida)
                            t_primero = t
                        veces = 1
                        if en_vivo:
                            # En vivo el análisis puede ir más lento que la señal: se repite el cuadro
                            # para que el video anotado dure lo mismo que el partido
                            veces = max(1, min(int((t - t_primero) * fps_salida) + 1 - escritos, int(2 * fps_salida)))
                        datos = anotado.tobytes()
                        for _ in range(veces):
                            escritor.stdin.write(datos)
                        escritos += veces
                    if mostrar:
                        cv2.imshow("Analista (q para salir)", anotado)
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                if progreso and n % 50 == 0:
                    vel = n / (time.time() - t_ini)
                    total = f"/{fuente.total // self.cfg.paso}" if fuente.total else ""
                    print(f"  cuadro {n}{total}  ({vel:.1f} cuadros/s)", flush=True)
                if max_cuadros and n >= max_cuadros:
                    break
        except KeyboardInterrupt:
            if not en_vivo:
                raise
            print("\nTerminando: se guarda lo analizado hasta ahora…")      # en vivo se corta con Ctrl+C
        for esc in (escritor, escritor_entrada):
            if esc is not None:
                esc.stdin.close()
                esc.wait()
        desfase = None
        if hasattr(fuente, "terminar_grabacion"):
            g = getattr(fuente, "grabadora", None)
            if g is not None and g.t0 is not None and getattr(fuente, "_t0", None):
                desfase = round(float(g.t0 - fuente._t0), 3)      # t del análisis = t de la grabación + desfase
            fuente.terminar_grabacion()
        if mostrar:
            cv2.destroyAllWindows()
        self.equipos.forzar_ajuste()
        segundos = time.time() - t_ini
        # En vivo el ritmo real es el que alcanzó el análisis, no el pedido
        fps = n / max(segundos, 1e-6) if en_vivo else fuente.fps
        extra = {"en_vivo": en_vivo, "forma_proceso": list(self.forma[:2]) if self.forma is not None else None}
        if desfase is not None:
            extra["grabacion_desfase_s"] = desfase
        if en_vivo and getattr(fuente, "_t0", None):
            extra["inicio_epoch"] = round(float(fuente._t0), 3)     # hora real del t = 0 (para alinear)
        return self.guardar(salida, n, segundos, round(fps, 2), extra)

    # --------------------------------------------------------------- post-proceso
    def tabla(self) -> pd.DataFrame:
        df = pd.DataFrame(self.registros)
        if df.empty:
            return df
        # Etiquetar retroactivamente lo visto antes de ajustar los colores
        if self.equipos.listo and {"L", "a", "b"} <= set(df.columns):
            falta = df["equipo_det"].isna() & df["L"].notna()
            if falta.any():
                df.loc[falta, "equipo_det"] = self.equipos.predecir(list(df.loc[falta, ["L", "a", "b"]].to_numpy()))
        # Equipo final por voto de mayoría de cada pista
        votos = df.dropna(subset=["equipo_det"]).groupby("id")["equipo_det"].agg(lambda s: s.value_counts().index[0])
        df["equipo"] = df["id"].map(votos).fillna("otro")
        # Arqueros: pistas "otro" que viven cerca de un arco pasan al equipo que defiende ese lado
        L2 = self.cancha.L2
        medias = df.groupby("id")[["x"]].mean()
        lado_equipo = df[df["equipo"].isin(["A", "B"])].groupby("equipo")["x"].mean()
        df["rol"] = "campo"
        if len(lado_equipo) == 2:
            for pid, fila in medias.iterrows():
                if votos.get(pid, "otro") == "otro" and abs(fila["x"]) > L2 - 22:
                    lado = np.sign(fila["x"])
                    eq = "A" if np.sign(lado_equipo["A"]) == lado and np.sign(lado_equipo["B"]) != lado \
                        else ("B" if np.sign(lado_equipo["B"]) == lado else None)
                    if eq is None:  # ambos en el mismo lado: el más retrasado hacia ese arco
                        eq = "A" if lado * lado_equipo["A"] > lado * lado_equipo["B"] else "B"
                    df.loc[df["id"] == pid, ["equipo", "rol"]] = [eq, "arquero"]
        df.loc[df["equipo"] == "otro", "rol"] = "arbitro/otro"
        # Suavizado temporal por pista (ruido de detección del pie)
        df = df.sort_values(["id", "cuadro"])
        for c in ("x", "y"):
            df[c + "_suave"] = df.groupby("id")[c].transform(
                lambda s: s.rolling(5, center=True, min_periods=1).median())
        return df.sort_values(["cuadro", "id"]).reset_index(drop=True)

    def guardar(self, salida: Path, n: int, segundos: float, fps: float, extra: dict | None = None) -> dict:
        salida = Path(salida)
        df = self.tabla()
        df.to_csv(salida / "tracking.csv", index=False)
        pd.DataFrame(self.registros_balon, columns=["cuadro", "t", "x", "y", "u", "v"]).to_csv(salida / "balon.csv", index=False)
        cam = pd.DataFrame(self.registros_camara)
        cam.drop(columns=["huella"], errors="ignore").to_csv(salida / "camara.csv", index=False)
        nombres = {k: v for k, v in zip(("A", "B"), list(self.cfg.equipos)[:2])} if self.cfg.equipos \
            else {"A": "Equipo A", "B": "Equipo B"}
        resumen = {
            "cuadros_procesados": n, "segundos_proceso": round(segundos, 1),
            "cuadros_por_segundo": round(n / max(segundos, 1e-6), 2),
            "fps_video": fps, "paso": self.cfg.paso,
            "porcentaje_calibrado": round(100 * float(cam["calibrado"].mean()), 1) if len(cam) else 0.0,
            "colores_equipos": self.equipos.colores_hex(),
            "nombres_equipos": nombres,
            "cancha": {"largo": self.cancha.largo, "ancho": self.cancha.ancho},
            "camara_fija": bool(self.camara_fija),
            "detector": self.cfg.detector, "imgsz": self.cfg.imgsz, "ancho_proceso": self.cfg.ancho_proceso,
            "conf": getattr(self.detector, "conf", None),
            **(extra or {}),
        }
        (salida / "resumen_proceso.json").write_text(json.dumps(resumen, indent=2, ensure_ascii=False), encoding="utf-8")
        return resumen


def _marcas_proyectadas(H_img2cancha, forma, cancha):
    try:
        H_c2i = np.linalg.inv(H_img2cancha)
    except np.linalg.LinAlgError:
        return None
    m = mascara_lineas_proyectadas(H_c2i, forma, cancha, grosor=3)
    for xy in [(0, 0), (cancha.L2 - cancha.distancia_penal, 0), (-cancha.L2 + cancha.distancia_penal, 0)]:
        u, v = aplicar_h(H_c2i, [xy])[0]
        if 0 <= u < forma[1] and 0 <= v < forma[0]:
            cv2.circle(m, (int(u), int(v)), 6, 255, -1)
    return cv2.dilate(m, np.ones((5, 5), np.uint8))


def _suprimir_duplicados(pos, conf, idx, radio=0.7):
    """Dos detecciones a menos de `radio` metros son el mismo jugador: queda la más confiable."""
    orden = sorted(idx, key=lambda k: -conf[k])
    elegidos = []
    for k in orden:
        if all(np.hypot(*(pos[k] - pos[j])) > radio for j in elegidos):
            elegidos.append(k)
    return np.array(sorted(elegidos), dtype=int)


def ruta_ffmpeg() -> str:
    """ffmpeg del sistema o, si no hay, el que trae el paquete imageio-ffmpeg (Windows sin instalar nada)."""
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("No se encontró ffmpeg: instala 'imageio-ffmpeg' (pip) o ffmpeg") from e


def _escritor_h264(ruta: Path, forma, fps: float, rapido: bool = False):
    """ffmpeg recibiendo cuadros BGR por stdin. rapido=True (grabar en vivo): casi sin CPU, archivo más grande."""
    h, w = forma[:2]
    return subprocess.Popen([ruta_ffmpeg(), "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
                             "-s", f"{w}x{h}", "-r", f"{fps:.3f}", "-i", "-",
                             "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",      # H.264 pide ancho y alto pares
                             "-c:v", "libx264", "-preset", "ultrafast" if rapido else "veryfast",
                             "-crf", "26" if rapido else "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(ruta)],
                            stdin=subprocess.PIPE)
