"""Compara lo que detectó la app con la verdad de la transmisión sintética.

    python demo/evaluar.py --resultados demo/salidas/completo --gt demo/clips/completo

Escribe evaluacion.json en la carpeta de resultados (el reporte lo muestra en "Validación").
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


def _emparejar(pred: np.ndarray, gt: np.ndarray, umbral: float):
    if not len(pred) or not len(gt):
        return []
    D = np.linalg.norm(pred[:, None] - gt[None], axis=2)
    f, c = linear_sum_assignment(np.where(D > umbral, 1e6, D))
    return [(i, j, D[i, j]) for i, j in zip(f, c) if D[i, j] <= umbral]


def evaluar_jugadores(tr: pd.DataFrame, gt: pd.DataFrame, largo: float, ancho: float, umbral: float) -> dict:
    gt = gt[gt["visible"] & (gt["x"].abs() <= largo / 2 + 1) & (gt["y"].abs() <= ancho / 2 + 1)]
    pares, n_gt, n_pred = [], 0, 0
    por_cuadro = dict(tuple(tr.groupby("cuadro")))
    for cuadro, g in gt.groupby("cuadro"):
        p = por_cuadro.get(cuadro)
        n_gt += len(g)
        if p is None:
            continue
        n_pred += len(p)
        for i, j, d in _emparejar(p[["x", "y"]].to_numpy(), g[["x", "y"]].to_numpy(), umbral):
            pares.append({"cuadro": cuadro, "pista": p["id"].iloc[i], "equipo_pred": p["equipo"].iloc[i],
                          "gt_id": g["id"].iloc[j], "equipo_gt": g["equipo"].iloc[j], "rol": g["rol"].iloc[j], "error": d})
    P = pd.DataFrame(pares)
    cuadros_pred = set(tr["cuadro"])
    n_gt_proc = int(gt["cuadro"].isin(cuadros_pred).sum())
    out = {"recall": round(len(P) / max(n_gt_proc, 1), 4), "precision": round(len(P) / max(n_pred, 1), 4),
           "error_mediana_m": round(float(P["error"].median()), 3) if len(P) else None,
           "error_p90_m": round(float(P["error"].quantile(0.9)), 3) if len(P) else None,
           "jugadores_gt": n_gt_proc, "detecciones": n_pred, "emparejadas": len(P)}
    if len(P):
        campo = P[(P["rol"] != "arquero") & P["equipo_pred"].isin(["A", "B"])]
        directo = (campo["equipo_pred"] == campo["equipo_gt"]).mean()
        out["equipo_correcto"] = round(float(max(directo, 1 - directo)), 4)
        out["equipo_invertido"] = bool(directo < 0.5)
        pureza = P.groupby("pista")["gt_id"].agg(lambda s: s.value_counts().iloc[0]).sum() / len(P)
        out["pureza_identidad"] = round(float(pureza), 4)
        out["pistas"] = int(P["pista"].nunique())
        out["jugadores_distintos"] = int(P["gt_id"].nunique())
    return out


def evaluar_balon(bal: pd.DataFrame, gt: pd.DataFrame, umbral: float = 2.0, cuadros=None) -> dict:
    gt = gt[gt["visible"]]
    if cuadros is not None:
        gt = gt[gt["cuadro"].isin(cuadros)]
    m = gt.merge(bal, on="cuadro", how="left", suffixes=("_gt", ""))
    d = np.hypot(m["x"] - m["x_gt"], m["y"] - m["y_gt"])
    bien = d <= umbral
    en_suelo = m["z"] < 1.0
    return {"cuadros_con_balon_visible": int(len(m)), "detectado_pct": round(100 * float(bien.mean()), 1),
            "detectado_en_suelo_pct": round(100 * float(bien[en_suelo].mean()), 1) if en_suelo.any() else None,
            "error_mediana_m": round(float(d[bien].median()), 3) if bien.any() else None,
            "falsos_pct": round(100 * float(((d > umbral) & m["x"].notna()).mean()), 1)}


def posesion_verdad(gt_j: pd.DataFrame, gt_b: pd.DataFrame, radio: float = 2.0) -> dict:
    """Mismo criterio que la app (jugador a ≤2 m del balón), solo con el balón en el suelo (z < 1 m)."""
    gt_b = gt_b[gt_b["z"] < 1.0] if "z" in gt_b else gt_b
    j = gt_j.merge(gt_b[["cuadro", "x", "y"]].rename(columns={"x": "bx", "y": "by"}), on="cuadro")
    j["d"] = np.hypot(j["x"] - j["bx"], j["y"] - j["by"])
    pos = j.loc[j.groupby("cuadro")["d"].idxmin()]
    pos = pos[pos["d"] <= radio]
    c = pos["equipo"].value_counts()
    return {eq: round(100 * float(c.get(eq, 0)) / max(c.sum(), 1), 1) for eq in ("A", "B")}


def evaluar_camara(cam: pd.DataFrame, gt: pd.DataFrame, ancho: int, alto: int, largo: float, ancho_c: float) -> dict:
    gt = gt[gt["tramo"] >= 0]
    m = gt.merge(cam, on="cuadro", how="inner", suffixes=("_gt", ""))
    if not len(m):
        return {}
    cal = m["calibrado"].astype(bool) if "calibrado" in m else pd.Series(False, index=m.index)
    us, vs = np.meshgrid(np.linspace(0.05, 0.95, 12) * ancho, np.linspace(0.3, 0.95, 8) * alto)
    uv = np.column_stack([us.ravel(), vs.ravel(), np.ones(us.size)])
    errores = []
    for _, r in m[cal].iterrows():
        Hg = np.array([r[f"h{i}_gt"] for i in range(9)]).reshape(3, 3)
        Hp = np.array([r[f"h{i}"] for i in range(9)]).reshape(3, 3)
        a, b = uv @ Hg.T, uv @ Hp.T
        a, b = a[:, :2] / a[:, 2:3], b[:, :2] / b[:, 2:3]
        ok = (np.abs(a[:, 0]) <= largo / 2) & (np.abs(a[:, 1]) <= ancho_c / 2)
        if ok.any():
            errores.append(float(np.median(np.linalg.norm(a[ok] - b[ok], axis=1))))
    e = np.array(errores)
    return {"calibrado_pct": round(100 * float(cal.mean()), 1),
            "error_mediana_m": round(float(np.median(e)), 3) if len(e) else None,
            "error_p90_m": round(float(np.quantile(e, 0.9)), 3) if len(e) else None,
            "cuadros_error_menor_1m_pct": round(100 * float((e < 1).mean()), 1) if len(e) else None}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--resultados", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--umbral", type=float, default=2.0, help="distancia máxima para emparejar (m)")
    ap.add_argument("--salida", default=None)
    a = ap.parse_args()
    res, gtd = Path(a.resultados), Path(a.gt)
    info = json.loads((gtd / "info.json").read_text(encoding="utf-8"))
    largo, ancho_c = info["cancha"]["largo"], info["cancha"]["ancho"]
    tr = pd.read_csv(res / "tracking.csv")
    bal = pd.read_csv(res / "balon.csv")
    cam = pd.read_csv(res / "camara.csv")
    proc = json.loads((res / "resumen_proceso.json").read_text(encoding="utf-8"))
    met = json.loads((res / "metricas.json").read_text(encoding="utf-8")) if (res / "metricas.json").exists() else {}
    gt_j, gt_b, gt_c = (pd.read_csv(gtd / f) for f in ("gt_jugadores.csv", "gt_balon.csv", "gt_camara.csv"))
    ev = {"clip": info["camara"], "cuadros": info["cuadros"], "fuente": info["fuente"]}
    ev["jugadores"] = evaluar_jugadores(tr, gt_j, largo, ancho_c, a.umbral)
    ev["balon"] = evaluar_balon(bal, gt_b, cuadros=set(cam["cuadro"]))
    ev["camara"] = evaluar_camara(cam, gt_c, info["resolucion"][0] / 1, info["resolucion"][1] / 1, largo, ancho_c)
    # posesión: verdad con los jugadores y el balón reales, en los cuadros procesados
    procesados = set(cam["cuadro"])
    verdad = posesion_verdad(gt_j[gt_j["cuadro"].isin(procesados)], gt_b[gt_b["cuadro"].isin(procesados)])
    est = met.get("posesion_pct") or {}
    if ev["jugadores"].get("equipo_invertido") and est:
        est = {"A": est.get("B"), "B": est.get("A")}
    ev["posesion"] = {"real": verdad, "estimada": est}
    # El gol "real" es cuando el balón cruza la línea de meta entre los postes (no el instante del remate)
    for g in info["goles"]:
        cruce = gt_b[(gt_b["t"] >= g["t"] - 0.5) & (gt_b["x"].abs() >= largo / 2) & (gt_b["y"].abs() <= 3.66 + 0.3)]
        g["t_remate"] = g["t"]
        if len(cruce):
            g["t"] = round(float(cruce["t"].iloc[0]), 2)
    goles_app = [e for e in met.get("eventos", []) if e.get("tipo") in ("gol", "posible gol")]
    ev["goles"] = {"reales": info["goles"],
                   "detectados": [{"t": e.get("t"), "tipo": e.get("tipo"), "equipo": e.get("equipo_atacante")} for e in goles_app],
                   "aciertos": sum(any(abs((d.get("t") or -99) - g["t"]) <= 2.0 for d in goles_app) for g in info["goles"])}
    ev["velocidad"] = {"cuadros_por_segundo": proc.get("cuadros_por_segundo"), "ancho_proceso": proc.get("ancho_proceso"),
                       "detector": proc.get("detector"), "imgsz": proc.get("imgsz")}
    salida = Path(a.salida) if a.salida else res / "evaluacion.json"
    salida.write_text(json.dumps(ev, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    j = ev["jugadores"]
    print(f"Jugadores: recall {100 * j['recall']:.1f}% · precisión {100 * j['precision']:.1f}% · error {j['error_mediana_m']} m "
          f"(p90 {j['error_p90_m']}) · equipo {100 * j.get('equipo_correcto', 0):.1f}% · identidad {100 * j.get('pureza_identidad', 0):.1f}%")
    print(f"Balón: {ev['balon']} · Cámara: {ev['camara']}")
    print(f"Posesión real {verdad} · estimada {est} · Goles {ev['goles']['aciertos']}/{len(info['goles'])} "
          f"{[round(g['t'], 1) for g in goles_app]} · {proc.get('cuadros_por_segundo')} cuadros/s")


if __name__ == "__main__":
    main()
