"""Entrena el modelo xG del paquete con tiros de StatsBomb Open Data.

Entrenamiento: Premier League 2015/16 (380 partidos).
Prueba (fuera de muestra): Mundial 2022, Euro 2020, Euro 2024 y Copa América 2024.

Uso:
  python demo/entrenar_xg.py --eventos datos/statsbomb/events --partidos datos/statsbomb/matches \
      --salida analista/modelos/xg.json --figura docs/media/xg_calibracion.png
Los archivos de eventos se descargan con `analista datos statsbomb --descargar ...` o desde
https://github.com/statsbomb/open-data.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analista.analisis.xg import rasgos  # noqa: E402
from analista.datos.statsbomb import eventos  # noqa: E402

RASGOS = {
    "basico": ["log_distancia", "angulo", "cabeza", "tiro_libre"],
    "contexto": ["log_distancia", "angulo", "cabeza", "tiro_libre", "balon_parado", "bajo_presion",
                 "primer_toque", "defensores_en_cono", "log_dist_arquero"],
}
ENTRENAMIENTO = {"2_27"}                       # Premier League 2015/16
PRUEBA = {"43_106", "55_43", "55_282", "223_282"}


def tiros_de(carpeta_eventos: Path, carpeta_partidos: Path, grupo: set[str]) -> pd.DataFrame:
    filas = []
    for f in sorted(carpeta_partidos.glob("*.json")):
        if f.stem not in grupo:
            continue
        for m in json.loads(f.read_text(encoding="utf-8")):
            ruta = carpeta_eventos / f"{m['match_id']}.json"
            if not ruta.exists():
                continue
            ev = eventos(crudos=json.loads(ruta.read_text(encoding="utf-8")))
            t = ev[(ev["tipo"] == "tiro") & (ev["periodo"] < 5)].copy()
            t["partido_id"] = m["match_id"]
            t["competicion"] = f.stem
            filas.append(t)
    return pd.concat(filas, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eventos", required=True)
    ap.add_argument("--partidos", required=True)
    ap.add_argument("--salida", default="analista/modelos/xg.json")
    ap.add_argument("--figura")
    a = ap.parse_args()
    ce, cp = Path(a.eventos), Path(a.partidos)
    print("Leyendo tiros…")
    tr = tiros_de(ce, cp, ENTRENAMIENTO)
    te = tiros_de(ce, cp, PRUEBA)
    tr = tr[tr["subtipo"] != "Penalty"]
    te_np = te[te["subtipo"] != "Penalty"]
    print(f"Entrenamiento: {len(tr)} tiros ({tr['exito'].mean():.3f} goles) · prueba: {len(te_np)} tiros")
    Rtr, Rte = rasgos(tr), rasgos(te_np)
    ytr, yte = tr["exito"].astype(int).to_numpy(), te_np["exito"].astype(int).to_numpy()
    salida = {"version": 1, "fecha": date.today().isoformat(),
              "datos": {"entrenamiento": "StatsBomb Open Data: Premier League 2015/16",
                        "prueba": "StatsBomb Open Data: Mundial 2022, Euro 2020, Euro 2024, Copa América 2024",
                        "tiros_entrenamiento": int(len(tr)), "tiros_prueba": int(len(te_np))},
              "penal": 0.78, "modelos": {}, "referencia_statsbomb": {}}
    calib = {}
    for nombre, cols in RASGOS.items():
        lr = LogisticRegression(C=1.0, max_iter=2000).fit(Rtr[cols].to_numpy(float), ytr)
        p = lr.predict_proba(Rte[cols].to_numpy(float))[:, 1]
        salida["modelos"][nombre] = {
            "rasgos": cols, "intercepto": float(lr.intercept_[0]), "coeficientes": [float(c) for c in lr.coef_[0]],
            "prueba": {"auc": round(roc_auc_score(yte, p), 4), "log_loss": round(log_loss(yte, p), 4),
                       "brier": round(brier_score_loss(yte, p), 4),
                       "xg_total": round(float(p.sum()), 1), "goles": int(yte.sum())}}
        calib[nombre] = p
        print(nombre, salida["modelos"][nombre]["prueba"])
    psb = te_np["xg"].to_numpy(float)
    salida["referencia_statsbomb"] = {"auc": round(roc_auc_score(yte, psb), 4), "log_loss": round(log_loss(yte, psb), 4),
                                      "brier": round(brier_score_loss(yte, psb), 4), "xg_total": round(float(psb.sum()), 1)}
    print("StatsBomb", salida["referencia_statsbomb"])
    Path(a.salida).parent.mkdir(parents=True, exist_ok=True)
    Path(a.salida).write_text(json.dumps(salida, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Modelo guardado en", a.salida)

    if a.figura:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
        bins = np.array([0, .03, .06, .1, .15, .2, .3, .45, .6, 1])
        for nombre, p, color in (("Modelo básico", calib["basico"], "#2a78d6"),
                                 ("Modelo con contexto", calib["contexto"], "#eb6834"),
                                 ("xG de StatsBomb", psb, "#7d877e")):
            idx = np.digitize(p, bins) - 1
            xs, ys, ns = [], [], []
            for k in range(len(bins) - 1):
                m = idx == k
                if m.sum() >= 15:
                    xs.append(p[m].mean())
                    ys.append(yte[m].mean())
                    ns.append(m.sum())
            ax.plot(xs, ys, "o-", color=color, label=nombre, markersize=4, linewidth=1.6)
        ax.plot([0, .7], [0, .7], color="#bbbbbb", linewidth=1, linestyle="--", zorder=0)
        ax.set_xlabel("xG predicho (promedio del grupo)")
        ax.set_ylabel("Proporción real de goles")
        ax.set_title(f"Calibración en {len(te_np):,} tiros fuera de muestra".replace(",", "."), fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=9)
        ax.set_xlim(0, .7)
        ax.set_ylim(0, .7)
        fig.tight_layout()
        Path(a.figura).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(a.figura)
        print("Figura en", a.figura)


if __name__ == "__main__":
    main()
