"""API de alto nivel: un `Partido` con sus eventos y/o su tracking, venga de donde venga.

    from analista import Partido

    final = Partido.statsbomb(3869685)          # eventos (StatsBomb Open Data)
    final.resumen()["por_equipo"]["A"]["xg"]
    nodos, aristas = final.red_de_pases("A")

    aleague = Partido.skillcorner(1886347)      # tracking (SkillCorner Open Data)
    aleague.fisico().sort_values("distancia_m").tail()
    xs, ys, P = aleague.control_de_cancha(t=5059.9)

    mio = Partido.video("resultados/")          # lo que salió de `analista analizar`
    mio.reporte("mi_partido.html")
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .analisis import eventos as ae
from .analisis import tracking as at
from .cancha import Cancha


class Partido:
    def __init__(self, info: dict, eventos: pd.DataFrame | None = None, tracking: pd.DataFrame | None = None,
                 balon: pd.DataFrame | None = None, camara: pd.DataFrame | None = None,
                 goles: pd.DataFrame | None = None):
        self.info = info
        self.eventos = eventos
        self.tracking = tracking
        self.balon = balon
        self.camara = camara
        self._goles = goles
        c = info.get("cancha", {})
        self.cancha = Cancha(largo=float(c.get("largo", 105)), ancho=float(c.get("ancho", 68)))
        self._metricas = None

    # ------------------------------------------------------------ constructores
    @classmethod
    def statsbomb(cls, partido_id: int, competicion_id: int | None = None, temporada_id: int | None = None) -> "Partido":
        from .datos import statsbomb as sb
        ev = sb.eventos(partido_id, nombres_cortos=True)
        meta = sb.info_partido(partido_id, competicion_id, temporada_id)
        nombres = ev.drop_duplicates("equipo").set_index("equipo")["equipo_nombre"].to_dict()
        g = ae.goles(ev)
        info = {"fuente": "StatsBomb Open Data", "partido_id": partido_id, "fecha": meta.get("fecha"),
                "competicion": meta.get("competicion"), "temporada": meta.get("temporada"), "fase": meta.get("fase"),
                "estadio": meta.get("estadio"),
                "equipos": {k: {"nombre": nombres.get(k, f"Equipo {k}"), "color": None} for k in ("A", "B")},
                "marcador": {k: int((g["equipo"] == k).sum()) for k in ("A", "B")},
                "cancha": {"largo": 105.0, "ancho": 68.0}}
        return cls(info, eventos=ev)

    @classmethod
    def skillcorner(cls, partido_id: int) -> "Partido":
        from .datos import skillcorner as sc
        d = sc.cargar(partido_id)
        p = cls(d["info"], tracking=d["tracking"], balon=d["balon"], goles=sc.goles(partido_id, d["info"]))
        return p

    @classmethod
    def video(cls, carpeta: str | Path) -> "Partido":
        from .datos import video as vd
        d = vd.cargar(carpeta)
        return cls(d["info"], tracking=d["tracking"], balon=d["balon"], camara=d["camara"])

    # ------------------------------------------------------------ utilidades
    @property
    def nombres(self) -> dict[str, str]:
        return {k: v.get("nombre", k) for k, v in self.info.get("equipos", {}).items()}

    @property
    def tiene_eventos(self) -> bool:
        return self.eventos is not None and len(self.eventos) > 0

    @property
    def tiene_tracking(self) -> bool:
        return self.tracking is not None and len(self.tracking) > 0

    def _params(self) -> at.ParametrosMetricas:
        p = at.ParametrosMetricas(largo=self.cancha.largo, ancho=self.cancha.ancho)
        if self.info.get("fuente") == "SkillCorner Open Data":
            p.ataca_derecha = "A"          # el cargador ya normalizó la dirección
        return p

    def __repr__(self):
        n = self.nombres
        partes = [f"{n.get('A')} vs {n.get('B')}", self.info.get("fuente", "")]
        if self.tiene_eventos:
            partes.append(f"{len(self.eventos)} eventos")
        if self.tiene_tracking:
            partes.append(f"{self.tracking['cuadro'].nunique()} cuadros de tracking")
        return "<Partido " + " · ".join(p for p in partes if p) + ">"

    # ------------------------------------------------------------ eventos
    def tiros(self) -> pd.DataFrame:
        return ae.tiros(self.eventos, self.cancha.largo) if self.tiene_eventos else pd.DataFrame()

    def goles(self) -> pd.DataFrame:
        if self.tiene_eventos:
            return ae.goles(self.eventos)
        if self._goles is not None:
            return self._goles
        ev = self.metricas()["eventos"]
        return ev[ev["tipo"] == "gol"] if len(ev) else ev

    def flujo_xg(self) -> pd.DataFrame:
        return ae.flujo_xg(self.eventos, self.cancha.largo)

    def red_de_pases(self, equipo: str, **kw):
        return ae.red_de_pases(self.eventos, equipo, **kw)

    # ------------------------------------------------------------ tracking
    def metricas(self, paso_forma: int | None = None) -> dict:
        """Todas las métricas de tracking (forma, posesión, pases, físico, eventos de balón)."""
        if self._metricas is None:
            if not self.tiene_tracking:
                raise ValueError("Este partido no tiene tracking")
            fps = float(self.info.get("fps") or 25)
            paso = paso_forma or max(1, int(round(fps / 2)))       # forma a ~2 cuadros por segundo
            self._metricas = at.calcular_todo(self.tracking, self.balon, self.camara, self._params(), paso)
        return self._metricas

    def forma(self) -> pd.DataFrame:
        return self.metricas()["forma"]

    def fisico(self) -> pd.DataFrame:
        f = self.metricas()["fisico"]
        if "nombre" not in f and "nombre" in self.tracking:
            f = f.merge(self.tracking.drop_duplicates("id")[["id", "nombre"]], on="id", how="left")
        return f

    def posesion(self) -> dict:
        return self.metricas()["resumen"].get("posesion_pct", {})

    def direccion(self) -> dict:
        return self.metricas()["direccion"]

    def instante(self, t: float, ventana: float = 0.5) -> pd.DataFrame:
        """Jugadores en el cuadro más cercano a `t` segundos, con velocidad estimada (vx, vy)."""
        cuadros = self.tracking[["cuadro", "t"]].drop_duplicates("cuadro").sort_values("t")
        k = cuadros.iloc[(cuadros["t"] - t).abs().argsort()[:1]]
        c0, t0 = int(k["cuadro"].iloc[0]), float(k["t"].iloc[0])
        ahora = self.tracking[self.tracking["cuadro"] == c0].copy()
        antes = self.tracking[(self.tracking["t"] >= t0 - ventana) & (self.tracking["t"] < t0 - ventana / 2)]
        if len(antes):
            a = antes.groupby("id")[["x", "y", "t"]].mean()
            ahora = ahora.join(a, on="id", rsuffix="_a")
            dt = (t0 - ahora["t_a"]).replace(0, np.nan)
            ahora["vx"] = ((ahora["x"] - ahora["x_a"]) / dt).fillna(0).clip(-9, 9)
            ahora["vy"] = ((ahora["y"] - ahora["y_a"]) / dt).fillna(0).clip(-9, 9)
            ahora = ahora.drop(columns=["x_a", "y_a", "t_a"])
        return ahora

    def balon_en(self, t: float) -> tuple[float, float] | None:
        if self.balon is None or self.balon.empty:
            return None
        b = self.balon.iloc[(self.balon["t"] - t).abs().argsort()[:1]]
        if abs(float(b["t"].iloc[0]) - t) > 1.0:
            return None
        return float(b["x"].iloc[0]), float(b["y"].iloc[0])

    def control_de_cancha(self, t: float, **kw):
        """Control de cancha en el instante `t` (s): (xs, ys, P_A). Ver `analisis.tracking.control_de_cancha`."""
        j = self.instante(t)
        j = j[j["equipo"].isin(["A", "B"])]
        return at.control_de_cancha(j, self.balon_en(t), self.cancha.largo, self.cancha.ancho, **kw)

    def formacion(self, equipo: str, desde: float | None = None, hasta: float | None = None) -> dict:
        df = self.tracking
        if desde is not None:
            df = df[df["t"] >= desde]
        if hasta is not None:
            df = df[df["t"] <= hasta]
        return at.formacion(df, equipo, self.direccion(), self.metricas()["poseedor"])

    # ------------------------------------------------------------ resumen y reportes
    def resumen(self) -> dict:
        out = {"partido": {k: self.info.get(k) for k in ("fuente", "fecha", "competicion", "estadio")},
               "equipos": self.nombres, "marcador": self.info.get("marcador")}
        if self.tiene_eventos:
            out.update(ae.resumen(self.eventos, self.cancha.largo))
        if self.tiene_tracking:
            out["tracking"] = self.metricas()["resumen"]
        return out

    def reporte(self, ruta: str | Path, titulo: str | None = None) -> Path:
        """Reporte HTML autocontenido del partido (eventos y/o tracking)."""
        ruta = Path(ruta)
        if self.tiene_eventos:
            from .reportes.eventos import reporte_eventos
            return reporte_eventos(self, ruta, titulo)
        from .reportes.tracking import reporte_tracking
        return reporte_tracking(self, ruta, titulo)
