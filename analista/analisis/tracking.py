"""Métricas tácticas y físicas a partir de datos de tracking (posiciones en metros).

Sirven igual para el tracking que sale del video, para SkillCorner o para cualquier
proveedor convertido al esquema común (ver `analista.datos.esquema`):

  tracking: cuadro, t, id, equipo ('A'/'B'/'otro'), rol, x, y  (opcional x_suave, y_suave, periodo)
  balon:    cuadro, t, x, y

Todas las métricas se calculan sobre los jugadores presentes en cada cuadro; con video
de TV eso son los que la cámara alcanza a ver.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull


@dataclass
class ParametrosMetricas:
    largo: float = 105.0
    ancho: float = 68.0
    radio_control: float = 2.0        # m: jugador que "tiene" el balón
    radio_presion: float = 5.0        # m: rivales que presionan al poseedor
    min_jugadores_forma: int = 5      # visibles por equipo para medir forma
    vel_sprint: float = 7.0           # m/s (25 km/h)
    ataca_derecha: str | None = None  # 'A' o 'B'; None = deducir


def _cols(df):
    return ("x_suave", "y_suave") if "x_suave" in df.columns else ("x", "y")


def direccion_ataque(df: pd.DataFrame, p: ParametrosMetricas) -> dict[str, int]:
    """+1 si el equipo ataca hacia x positivo, -1 si hacia x negativo."""
    if p.ataca_derecha in ("A", "B"):
        otro = "B" if p.ataca_derecha == "A" else "A"
        return {p.ataca_derecha: 1, otro: -1}
    if df is None or df.empty:
        return {"A": 1, "B": -1}
    cx, _ = _cols(df)
    if "rol" in df.columns:
        arq = df[(df["rol"] == "arquero") & df["equipo"].isin(["A", "B"])]
        if len(arq):
            lado = arq.groupby("equipo")[cx].mean()
            eq = lado.index[0]
            d = -int(np.sign(lado.iloc[0]) or 1)   # el arquero está en su propio arco
            return {eq: d, ("B" if eq == "A" else "A"): -d}
    medias = df[df["equipo"].isin(["A", "B"])].groupby("equipo")[cx].mean()
    if len(medias) == 2:
        a = 1 if medias["A"] < medias["B"] else -1
        return {"A": a, "B": -a}
    return {"A": 1, "B": -1}


# ------------------------------------------------------------------ forma de equipo
def _area_casco(g: pd.DataFrame) -> float:
    pts = g[["xs", "ys"]].to_numpy()
    if len(pts) < 3:
        return np.nan
    try:
        return float(ConvexHull(pts).volume)
    except Exception:  # noqa: BLE001 (puntos colineales)
        return np.nan


def forma_por_cuadro(df: pd.DataFrame, p: ParametrosMetricas, direccion: dict, paso: int = 1) -> pd.DataFrame:
    """Ancho, profundidad, área, compacidad y altura de la última línea por cuadro y equipo.

    Las coordenadas se normalizan para que cada equipo ataque hacia +x. `paso` permite
    calcular 1 de cada N cuadros (útil para partidos completos).
    """
    cols = ["cuadro", "t", "equipo", "n_visibles", "centroide_x", "centroide_y", "ancho", "profundidad",
            "area", "compacidad", "altura_bloque"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    cx, cy = _cols(df)
    campo = df[df["equipo"].isin(["A", "B"])]
    if "rol" in campo.columns:
        campo = campo[campo["rol"] != "arquero"]
    if paso > 1:
        cuadros = np.sort(campo["cuadro"].unique())[::paso]
        campo = campo[campo["cuadro"].isin(cuadros)]
    if campo.empty:
        return pd.DataFrame(columns=cols)
    d = campo["equipo"].map(direccion).fillna(1).to_numpy()
    c = pd.DataFrame({"cuadro": campo["cuadro"].to_numpy(), "t": campo["t"].to_numpy(),
                      "equipo": campo["equipo"].to_numpy(),
                      "xs": campo[cx].to_numpy() * d, "ys": campo[cy].to_numpy()})
    g = c.groupby(["cuadro", "equipo"], sort=True)
    agg = g.agg(t=("t", "first"), n_visibles=("xs", "size"), centroide_x=("xs", "mean"), centroide_y=("ys", "mean"),
                xmax=("xs", "max"), xmin=("xs", "min"), ymax=("ys", "max"), ymin=("ys", "min"))
    agg = agg[agg["n_visibles"] >= p.min_jugadores_forma]
    if agg.empty:
        return pd.DataFrame(columns=cols)
    c = c.join(agg[["centroide_x", "centroide_y"]], on=["cuadro", "equipo"], how="inner")
    c["dist"] = np.hypot(c["xs"] - c["centroide_x"], c["ys"] - c["centroide_y"])
    comp = c.groupby(["cuadro", "equipo"])["dist"].mean()
    # Altura de la última línea: penúltimo jugador más retrasado (el último suele ser un central o lateral)
    c = c.sort_values(["cuadro", "equipo", "xs"])
    c["orden"] = c.groupby(["cuadro", "equipo"]).cumcount()
    seg = c[c["orden"] == 1].set_index(["cuadro", "equipo"])["xs"]
    area = c.groupby(["cuadro", "equipo"])[["xs", "ys"]].apply(_area_casco)
    out = agg.assign(ancho=agg["ymax"] - agg["ymin"], profundidad=agg["xmax"] - agg["xmin"],
                     compacidad=comp, altura_bloque=seg + p.largo / 2, area=area)
    return out.reset_index()[cols]


# ------------------------------------------------------------------ posesión y toques
def poseedor_por_cuadro(df: pd.DataFrame, balon: pd.DataFrame, p: ParametrosMetricas) -> pd.DataFrame:
    """Jugador más cercano al balón en cada cuadro; poseedor si está a <= radio_control."""
    cols = ["cuadro", "t", "bx", "by", "dist", "id", "equipo", "presion_n", "presion_dist"]
    if balon is None or balon.empty or df is None or df.empty:
        return pd.DataFrame(columns=cols)
    cx, cy = _cols(df)
    j = df[df["equipo"].isin(["A", "B"])][["cuadro", "id", "equipo", cx, cy]].rename(columns={cx: "px", cy: "py"})
    b = balon[["cuadro", "t", "x", "y"]].rename(columns={"x": "bx", "y": "by"})
    m = j.merge(b, on="cuadro", how="inner")
    if m.empty:
        return pd.DataFrame(columns=cols)
    m["d"] = np.hypot(m["px"] - m["bx"], m["py"] - m["by"])
    m = m.dropna(subset=["d"])
    cerca = m.loc[m.groupby("cuadro")["d"].idxmin()].set_index("cuadro")
    out = pd.DataFrame({"cuadro": cerca.index, "t": cerca["t"].to_numpy(), "bx": cerca["bx"].to_numpy(),
                        "by": cerca["by"].to_numpy(), "dist": cerca["d"].to_numpy()})
    tiene = cerca["d"].to_numpy() <= p.radio_control
    out["id"] = np.where(tiene, cerca["id"].to_numpy(), np.nan)
    out["equipo"] = np.where(tiene, cerca["equipo"].to_numpy(), None)
    # Presión: rivales del poseedor a <= radio_presion
    pos = cerca[tiene][["px", "py", "equipo"]].rename(columns={"px": "qx", "py": "qy", "equipo": "eq_pos"})
    r = m.join(pos, on="cuadro", how="inner")
    r = r[r["equipo"] != r["eq_pos"]]
    r["dr"] = np.hypot(r["px"] - r["qx"], r["py"] - r["qy"])
    pres = r.groupby("cuadro").agg(presion_n=("dr", lambda s: int((s <= p.radio_presion).sum())),
                                   presion_dist=("dr", "min"))
    out = out.join(pres, on="cuadro")
    out.loc[~tiene, ["presion_n", "presion_dist"]] = np.nan
    return out[cols].reset_index(drop=True)


def secuencia_toques(poseedor: pd.DataFrame, max_hueco_s: float = 1.5, min_dur_s: float = 0.12) -> pd.DataFrame:
    """Agrupa cuadros consecutivos del mismo poseedor en 'toques' (controles)."""
    pos = poseedor.dropna(subset=["id"]).sort_values("t")
    toques, actual = [], None
    for r in pos.itertuples():
        if actual and r.id == actual["id"] and r.t - actual["t_fin"] <= max_hueco_s:
            actual["t_fin"] = r.t
            actual["x_fin"], actual["y_fin"] = r.bx, r.by
            continue
        if actual:
            toques.append(actual)
        actual = {"id": r.id, "equipo": r.equipo, "t_ini": r.t, "t_fin": r.t,
                  "x_ini": r.bx, "y_ini": r.by, "x_fin": r.bx, "y_fin": r.by}
    if actual:
        toques.append(actual)
    t = pd.DataFrame(toques)
    if t.empty:
        return t
    t["duracion"] = t["t_fin"] - t["t_ini"]
    return t[t["duracion"] >= min_dur_s - 1e-9].reset_index(drop=True) if len(t) > 1 else t


def pases(toques: pd.DataFrame, max_vuelo_s: float = 4.0) -> pd.DataFrame:
    """Pase = un toque seguido de otro de un jugador distinto; si es rival, pérdida."""
    filas = []
    if toques is None or len(toques) < 2:
        return pd.DataFrame(filas)
    for a, b in zip(toques.itertuples(), toques.iloc[1:].itertuples()):
        vuelo = b.t_ini - a.t_fin
        if vuelo > max_vuelo_s or a.id == b.id:
            continue
        filas.append({"t": a.t_fin, "equipo": a.equipo, "de": a.id, "a": b.id,
                      "completado": a.equipo == b.equipo, "x_ini": a.x_fin, "y_ini": a.y_fin,
                      "x_fin": b.x_ini, "y_fin": b.y_ini, "distancia": np.hypot(b.x_ini - a.x_fin, b.y_ini - a.y_fin)})
    return pd.DataFrame(filas)


# ------------------------------------------------------------------ eventos de balón
def eventos_balon(balon: pd.DataFrame, p: ParametrosMetricas, direccion: dict,
                  camara: pd.DataFrame | None = None) -> pd.DataFrame:
    """Remates, goles y saques de centro a partir de la trayectoria del balón.

    - gol: el balón cruza la línea de meta entre los postes (con tolerancia de proyección).
    - posible gol: balón rápido hacia el arco, junto a la boca del arco, y después se pierde
      (corte de cámara a la celebración, balón en la red tapado) sin volver a la cancha.
    - remate: balón a más de 12 m/s hacia el arco, en el último tercio.
    - saque de centro: balón quieto en el punto central (señal típica después de un gol).
    """
    cols = ["cuadro", "t", "tipo", "equipo_atacante", "x", "y", "velocidad"]
    if balon is None or len(balon) < 3:
        return pd.DataFrame(columns=cols)
    b = balon.sort_values("t").reset_index(drop=True).copy()
    b["xs"] = b["x"].rolling(3, center=True, min_periods=1).median()
    b["ys"] = b["y"].rolling(3, center=True, min_periods=1).median()
    dt = b["t"].diff()
    b["vx"] = (b["xs"].diff() / dt).rolling(3, min_periods=1).mean()
    b["vy"] = (b["ys"].diff() / dt).rolling(3, min_periods=1).mean()
    b["vel"] = np.hypot(b["vx"], b["vy"])
    L2 = p.largo / 2
    poste = 3.66
    cortes_t = []
    if camara is not None and len(camara) and "calibrado" in camara:
        cam = camara.sort_values("t")
        perdida = ~cam["calibrado"].astype(bool)
        cortes_t = cam.loc[perdida & ~perdida.shift(1, fill_value=False), "t"].tolist()
    evs, ultimo = [], {}

    def agregar(r, tipo, lado):
        clave = ("gol" if tipo in ("gol", "posible gol") else tipo, lado)
        if r.t - ultimo.get(clave, -99) < 6.0:
            return
        ultimo[clave] = r.t
        atacante = next((eq for eq, d in direccion.items() if d == lado), None)
        evs.append({"cuadro": int(r.cuadro), "t": round(float(r.t), 2), "tipo": tipo, "equipo_atacante": atacante,
                    "x": round(float(r.xs), 1), "y": round(float(r.ys), 1),
                    "velocidad": round(float(r.vel), 1) if np.isfinite(r.vel) else None})

    filas = list(b.itertuples())
    for n, r in enumerate(filas):
        lado = int(np.sign(r.xs) or 1)
        hacia_arco = np.isfinite(r.vx) and np.sign(r.vx) == lado
        if abs(r.xs) >= L2 - 0.3 and abs(r.ys) <= poste + 1.2:
            agregar(r, "gol", lado)
            continue
        if r.vel and 12 <= r.vel <= 45 and hacia_arco and abs(r.xs) > L2 - 30 and abs(r.ys) < 20:
            agregar(r, "remate", lado)
        # Posible gol: muy cerca de la boca del arco, yendo hacia él, y luego se pierde el balón
        if abs(r.xs) >= L2 - 4.0 and abs(r.ys) <= poste + 3.0 and r.vel and r.vel > 5 and hacia_arco:
            sig = filas[n + 1] if n + 1 < len(filas) else None
            hueco = (sig.t - r.t) if sig is not None else np.inf
            corte = any(0 <= c - r.t <= 4.0 for c in cortes_t)
            if hueco >= 1.5 or corte:
                agregar(r, "posible gol", lado)
    # Saques de centro (balón quieto en el centro >= 0.8 s)
    centro = (np.hypot(b["xs"], b["ys"]) <= 1.5) & (b["vel"].fillna(0) < 1.5)
    bloque = (centro != centro.shift()).cumsum()
    for _, g in b[centro].groupby(bloque[centro]):
        if g["t"].iloc[-1] - g["t"].iloc[0] >= 0.8:
            r = next(g.itertuples())
            evs.append({"cuadro": int(r.cuadro), "t": round(float(r.t), 2), "tipo": "saque de centro",
                        "equipo_atacante": None, "x": 0.0, "y": 0.0, "velocidad": 0.0})
    out = pd.DataFrame(evs, columns=cols).sort_values("t").reset_index(drop=True)
    # Un "posible gol" seguido de saque de centro en < 90 s se confirma como gol
    for i in out.index[out["tipo"] == "posible gol"]:
        if ((out["tipo"] == "saque de centro") & (out["t"] > out.at[i, "t"]) & (out["t"] - out.at[i, "t"] < 90)).any():
            out.at[i, "tipo"] = "gol"
    return out


# ------------------------------------------------------------------ físico
def fisico(df: pd.DataFrame, p: ParametrosMetricas, max_hueco_s: float = 0.6, ventana_s: float = 0.6) -> pd.DataFrame:
    """Distancia, velocidad máxima y sprint por pista (o jugador).

    La posición medida tiene ruido; derivarla directo infla la velocidad. Por eso se suaviza
    la trayectoria (~0.6 s) en cada tramo continuo y la velocidad se mide sobre ~0.3 s.
    """
    cols = ["id", "equipo", "rol", "segundos_visible", "distancia_m", "vel_max", "seg_sprint",
            "x_media", "y_media", "m_por_min"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    cx, cy = _cols(df)
    filas = []
    for pid, g in df.sort_values("t").groupby("id"):
        if len(g) < 5:
            continue
        t = g["t"].to_numpy(float)
        x, y = g[cx].to_numpy(float), g[cy].to_numpy(float)
        dt_med = float(np.median(np.diff(t))) if len(t) > 1 else 0.04
        dt_med = dt_med if dt_med > 0 else 0.04
        det = g["detectado"].to_numpy(bool) if "detectado" in g else np.ones(len(g), bool)
        tramo = np.concatenate([[0], np.cumsum(np.diff(t) > max_hueco_s)])
        w = max(3, int(round(ventana_s / dt_med)) | 1)
        k = max(1, int(round(0.3 / dt_med)))
        dist = sprint = visible = 0.0
        vs = []
        for _, idx in pd.Series(range(len(t))).groupby(tramo):
            ii = idx.to_numpy()
            if len(ii) < 3:
                continue
            xs = pd.Series(x[ii]).rolling(w, center=True, min_periods=1).mean().to_numpy()
            ys = pd.Series(y[ii]).rolling(w, center=True, min_periods=1).mean().to_numpy()
            tt = t[ii]
            visible += tt[-1] - tt[0]
            dist += float(np.hypot(np.diff(xs), np.diff(ys)).sum())
            if len(ii) > k:
                v = np.hypot(xs[k:] - xs[:-k], ys[k:] - ys[:-k]) / np.maximum(tt[k:] - tt[:-k], 1e-6)
                v = np.where(v <= 11, v, np.nan)   # más rápido es error de proyección
                # Velocidad máxima solo con posiciones observadas (no extrapoladas fuera de cuadro)
                v_obs = np.where(det[ii][k:] & det[ii][:-k], v, np.nan)
                # Velocidad sostenida: mediana móvil de ~1 s (un pico aislado no cuenta)
                vm = pd.Series(v_obs).rolling(max(3, int(round(1.0 / dt_med)) | 1), center=True, min_periods=2).median()
                vs.extend(vm.dropna().tolist())
                sprint += float((v >= p.vel_sprint).sum() * dt_med)
        if visible <= 0:
            continue
        vs = np.array(vs)
        fila = {"id": pid, "equipo": g["equipo"].iloc[0], "rol": g["rol"].iloc[0] if "rol" in g else "campo",
                "segundos_visible": float(visible), "distancia_m": dist,
                "vel_max": float(vs.max()) if len(vs) else np.nan,
                "seg_sprint": sprint, "x_media": float(np.mean(x)), "y_media": float(np.mean(y))}
        if "nombre" in g:
            fila["nombre"] = g["nombre"].iloc[0]
        filas.append(fila)
    f = pd.DataFrame(filas)
    if not f.empty:
        f["m_por_min"] = f["distancia_m"] / (f["segundos_visible"].clip(lower=1) / 60)
    return f


# ------------------------------------------------------------------ control de cancha
def control_de_cancha(jugadores: pd.DataFrame, balon_xy=None, largo: float = 105.0, ancho: float = 68.0,
                      resolucion: float = 1.5, vel: float = 5.0, reaccion: float = 0.7,
                      sigma: float = 0.45) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Control de cancha de un instante (modelo de tiempo de llegada, estilo Spearman simplificado).

    Para cada punto de una grilla se estima cuánto tarda en llegar el jugador más rápido de
    cada equipo (tiempo de reacción + distancia / velocidad, con la velocidad actual si viene en
    `vx`, `vy`). La probabilidad de que el equipo A controle el punto es una logística de la
    diferencia de tiempos. Devuelve (xs, ys, P_A) con P_A de forma (len(ys), len(xs)).
    """
    xs = np.arange(-largo / 2 + resolucion / 2, largo / 2, resolucion)
    ys = np.arange(-ancho / 2 + resolucion / 2, ancho / 2, resolucion)
    gx, gy = np.meshgrid(xs, ys)
    pts = np.stack([gx.ravel(), gy.ravel()], 1)
    tiempos = {}
    for eq in ("A", "B"):
        j = jugadores[jugadores["equipo"] == eq]
        if j.empty:
            tiempos[eq] = np.full(len(pts), np.inf)
            continue
        pos = j[["x", "y"]].to_numpy(float)
        if {"vx", "vy"} <= set(j.columns):
            pos = pos + reaccion * j[["vx", "vy"]].fillna(0).to_numpy(float)   # sigue su inercia al reaccionar
        d = np.linalg.norm(pts[:, None, :] - pos[None], axis=2)
        tiempos[eq] = reaccion + d.min(1) / vel
    dif = tiempos["B"] - tiempos["A"]
    P = 1 / (1 + np.exp(-dif / sigma))
    return xs, ys, P.reshape(gy.shape)


# ------------------------------------------------------------------ formación
def formacion(df: pd.DataFrame, equipo: str, direccion: dict, poseedor: pd.DataFrame | None = None) -> dict:
    """Formación aproximada del equipo a partir de su forma media.

    1. Si hay datos de posesión, se usan los cuadros en que el rival tiene el balón (la forma
       defensiva es la que define el sistema).
    2. Cada cuadro se centra en el centroide del equipo, así el bloque que sube y baja no borra
       las líneas; se promedia la posición relativa de cada jugador.
    3. Se ordena por profundidad y se busca la partición en 3 o 4 líneas que minimiza la
       varianza dentro de cada línea, con una línea de fondo de al menos 3 jugadores.
    """
    cx, cy = _cols(df)
    g = df[df["equipo"] == equipo]
    if "rol" in g.columns:
        g = g[g["rol"] != "arquero"]
    if g.empty:
        return {"formacion": None, "lineas": [], "posiciones": []}
    if poseedor is not None and len(poseedor):
        rival = "B" if equipo == "A" else "A"
        sin_balon = poseedor.loc[poseedor["equipo"] == rival, "cuadro"]
        if len(sin_balon) > 50:
            g = g[g["cuadro"].isin(sin_balon)]
    d = direccion.get(equipo, 1)
    g = g.assign(xs=g[cx] * d, ys=g[cy])
    g = g.assign(xr=g["xs"] - g.groupby("cuadro")["xs"].transform("mean"),
                 yr=g["ys"] - g.groupby("cuadro")["ys"].transform("mean"))
    medias = g.groupby("id").agg(x=("xr", "mean"), y=("yr", "mean"), n=("xr", "size"))
    medias = medias.sort_values("n", ascending=False).head(10)
    if len(medias) < 6:
        return {"formacion": None, "lineas": [], "posiciones": []}
    orden = medias.sort_values("x")
    v = orden["x"].to_numpy()
    n = len(v)
    from itertools import combinations
    mejor = None
    for k in (3, 4):
        for cortes in combinations(range(1, n), k - 1):
            grupos = np.split(v, cortes)
            if len(grupos[0]) < 3 or len(grupos[0]) > 5:
                continue
            costo = sum(((gr - gr.mean()) ** 2).sum() for gr in grupos)
            costo += 3.0 * sum(len(gr) == 1 for gr in grupos[:-1])       # un solo jugador entre líneas
            costo += 8.0 * (k - 3)                                        # preferir 3 líneas si da parecido
            if mejor is None or costo < mejor[0]:
                mejor = (costo, cortes)
    if mejor is None:
        return {"formacion": None, "lineas": [], "posiciones": []}
    grupos_idx = np.split(np.arange(n), mejor[1])
    lineas = [orden.index[gr].tolist() for gr in grupos_idx]
    return {"formacion": "-".join(str(len(gr)) for gr in grupos_idx), "lineas": lineas,
            "posiciones": orden[["x", "y"]].reset_index().to_dict(orient="records")}


# ------------------------------------------------------------------ resumen
def calcular_todo(df: pd.DataFrame, balon: pd.DataFrame, camara: pd.DataFrame | None,
                  p: ParametrosMetricas, paso_forma: int = 1) -> dict:
    # Posiciones faltantes (jugador sin ubicar en ese cuadro) no aportan a ninguna métrica
    if len(df):
        df = df.dropna(subset=["x", "y"])
    if balon is not None and len(balon):
        balon = balon.dropna(subset=["x", "y"])
    direccion = direccion_ataque(df, p)
    forma = forma_por_cuadro(df, p, direccion, paso=paso_forma)
    poseedor = poseedor_por_cuadro(df, balon, p)
    toques = secuencia_toques(poseedor) if len(poseedor) else pd.DataFrame()
    ps = pases(toques) if len(toques) > 1 else pd.DataFrame()
    evs = eventos_balon(balon, p, direccion, camara)
    fis = fisico(df, p)

    res = {"direccion_ataque": direccion}
    if len(poseedor) and poseedor["equipo"].notna().any():
        cont = poseedor["equipo"].value_counts()
        tot = cont.sum()
        res["posesion_pct"] = {eq: round(100 * float(cont.get(eq, 0)) / float(tot), 1) for eq in ("A", "B")}
        res["cuadros_con_balon"] = int(len(poseedor))
        res["cuadros_con_poseedor"] = int(tot)
    if len(poseedor) and poseedor["presion_n"].notna().any():
        pr = poseedor.dropna(subset=["equipo"]).groupby("equipo").agg(
            rivales_a_5m=("presion_n", "mean"), dist_rival_mas_cercano=("presion_dist", "median"))
        res["presion_sufrida"] = pr.round(2).to_dict(orient="index")
    if len(forma):
        res["forma"] = forma.groupby("equipo")[["n_visibles", "ancho", "profundidad", "area", "compacidad",
                                                "altura_bloque", "centroide_x"]].median().round(1).to_dict(orient="index")
    if len(toques):
        res["toques"] = {k: int(v) for k, v in toques.groupby("equipo").size().items()}
    if len(ps):
        res["pases"] = {eq: {"intentos": int(len(g)), "completados": int(g["completado"].sum()),
                             "precision_pct": round(100 * float(g["completado"].mean()), 1),
                             "distancia_media": round(float(g["distancia"].mean()), 1)}
                        for eq, g in ps.groupby("equipo")}
    if len(fis):
        res["fisico_equipo"] = fis[fis["equipo"].isin(["A", "B"])].groupby("equipo").agg(
            distancia_total_m=("distancia_m", "sum"), vel_max=("vel_max", "max")).round(1).to_dict(orient="index")
    res["eventos"] = evs.to_dict(orient="records")
    if camara is not None and len(camara) and "calibrado" in camara:
        cal = camara["calibrado"].astype(bool)
        res["calidad"] = {"pct_cuadros_calibrados": round(100 * float(cal.mean()), 1),
                          "calidad_media_lineas": round(float(camara.loc[cal, "calidad"].mean()), 3) if cal.any() else None,
                          "cortes_detectados": int(camara["corte"].sum()) if "corte" in camara else 0}
    return {"resumen": res, "forma": forma, "poseedor": poseedor, "toques": toques, "pases": ps,
            "eventos": evs, "fisico": fis, "direccion": direccion, "balon": balon}
