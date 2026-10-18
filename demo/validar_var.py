"""Valida el VAR de la app contra decisiones reales de árbitros y contra la verdad sintética.

    python demo/validar_var.py skillcorner          # 20 partidos de la A-League (tracking 10 Hz)
    python demo/validar_var.py statsbomb            # Mundial 2022 y Euro 2024 (capturas 360)
    python demo/validar_var.py video                # clips sintéticos: video vs posiciones reales
    python demo/validar_var.py tiros                # remates de StatsBomb vs fueras de lugar pitados tras el remate

Escribe demo/salidas/validacion_var.json (lo leen el README y el sitio).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parent
SALIDA = RAIZ / "salidas" / "validacion_var.json"


def _guardar(clave: str, datos: dict):
    todo = json.loads(SALIDA.read_text(encoding="utf-8")) if SALIDA.exists() else {}
    todo[clave] = datos
    SALIDA.parent.mkdir(parents=True, exist_ok=True)
    SALIDA.write_text(json.dumps(todo, indent=2, ensure_ascii=False), encoding="utf-8")


def _tabla(filas: list[dict], columna_real: str) -> dict:
    df = pd.DataFrame(filas)
    if df.empty:
        return {}
    t = df.groupby([columna_real, "veredicto"]).size().unstack(fill_value=0)
    return {str(k): {str(c): int(v) for c, v in fila.items()} for k, fila in t.iterrows()}



def _tasas(df: pd.DataFrame) -> dict:
    """Entre las jugadas que se pudieron medir: qué parte tiene al receptor por delante de la línea."""
    m = df[df["veredicto"].isin(["fuera de lugar", "dudoso", "habilitado", "sin intervención"]) & df["margen_m"].notna()]
    out = {}
    for real, g in m.groupby("real"):
        out[str(real)] = {"medidos": int(len(g)), "adelantado_pct": round(100 * float((g["margen_m"] > 0).mean()), 1),
                          "margen_mediano_m": round(float(g["margen_m"].median()), 2)}
    return out

def skillcorner():
    from analista import Partido
    from analista.datos import skillcorner as sc
    filas, ejemplos, t0 = [], [], time.time()
    for pid in sc.partidos()["partido_id"]:
        p = Partido.skillcorner(int(pid))
        rev = p.var()
        ev = sc.eventos_dinamicos(int(pid)).set_index(sc.eventos_dinamicos(int(pid))["event_id"].astype(str))
        for r in rev.fueras_de_lugar:
            res = r.get("resultado_evento") or ""
            real = "sancionado" if res == "offside" else ("completado" if res == "successful" else "no llegó")
            # La línea que calcula SkillCorner (último defensor sin arquero) y la x del receptor buscado,
            # en el sentido de ataque del equipo: sirven para comparar la geometría
            e = ev.loc[r["evento"]] if r.get("evento") in ev.index else None
            m_sc = float(e["player_targeted_x_pass"] - e["last_defensive_line_x_end"]) if e is not None and \
                pd.notna(e["player_targeted_x_pass"]) and pd.notna(e["last_defensive_line_x_end"]) else None
            filas.append({"partido": int(pid), "real": real, "veredicto": r.get("posicion", r["veredicto"]),
                          "margen_m": r.get("margen_m"), "sigma_m": r.get("sigma_m"), "t": r["t"],
                          "margen_sc": m_sc, "linea_por": r.get("linea_por")})
            if real == "sancionado":
                ejemplos.append({"partido": int(pid), "t": r["t"], "veredicto": r.get("posicion", r["veredicto"]),
                                 "margen_m": r.get("margen_m"), "sigma_m": r.get("sigma_m"),
                                 "avisos": r.get("avisos", [])})
        print(f"{pid}: {len(rev.fueras_de_lugar)} pases revisados · {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(filas)
    sanc = df[df["real"] == "sancionado"]
    comp = df[df["real"] == "completado"]
    out = {"partidos": int(df["partido"].nunique()), "pases_revisados": int(len(df)),
           "tabla": _tabla(filas, "real"),
           "sancionados": int(len(sanc)),
           "sancionados_adelantados_o_dudosos": int(sanc["veredicto"].isin(["fuera de lugar", "dudoso"]).sum()),
           "sancionados_margen_mediano_m": round(float(sanc["margen_m"].median()), 2) if len(sanc) else None,
           "completados": int(len(comp)),
           "completados_fuera_de_lugar": int((comp["veredicto"] == "fuera de lugar").sum()),
           "completados_dudosos": int((comp["veredicto"] == "dudoso").sum()),
           "sigma_mediana_m": round(float(df["sigma_m"].median()), 2),
           "tasas": _tasas(df),
           "ejemplos_sancionados": ejemplos}
    geo = df.dropna(subset=["margen_m", "margen_sc"])
    geo = geo[geo["linea_por"] == "penúltimo defensor"]
    dif = (geo["margen_m"] - geo["margen_sc"]).abs()
    out["geometria_vs_skillcorner"] = {
        "pases": int(len(geo)), "diferencia_mediana_m": round(float(dif.median()), 3),
        "diferencia_p90_m": round(float(dif.quantile(0.9)), 3),
        "mismo_lado_pct": round(100 * float(((geo["margen_m"] > 0) == (geo["margen_sc"] > 0)).mean()), 1)}
    _guardar("skillcorner", out)
    print(json.dumps({k: v for k, v in out.items() if k != "ejemplos_sancionados"}, indent=2, ensure_ascii=False))


def statsbomb(competiciones=((43, 106), (55, 282)), borrar_360: bool = True):
    from analista import Partido
    from analista.datos import statsbomb as sb
    from analista.var_partido import var_statsbomb
    filas, ejemplos, faltas, t0 = [], [], [], time.time()
    for comp, temp in competiciones:
        for pid in sb.partidos(comp, temp)["partido_id"]:
            tres = sb.tres_sesenta(int(pid))
            if not tres:
                continue
            p = Partido.statsbomb(int(pid), comp, temp)
            rev, _ = var_statsbomb(p, tres=tres)
            for f in rev.faltas:
                oc = (f.get("ocasion") or {}).get("veredicto")
                if oc is not None:
                    faltas.append({"partido": int(pid), "ocasion": oc, "tarjeta": f.get("tarjeta") or "ninguna",
                                   "area": f["veredicto"], "sancion": f.get("sancion"), "mano": f.get("motivo") == "mano"})
            for r in rev.fueras_de_lugar:
                res = r.get("resultado_evento") or ""
                real = "sancionado" if res == "Pass Offside" else ("completado" if res == "Completo" else "no llegó")
                filas.append({"partido": int(pid), "real": real, "veredicto": r.get("posicion", r["veredicto"]),
                              "margen_m": r.get("margen_m"), "sigma_m": r.get("sigma_m")})
                if real == "sancionado" and len(ejemplos) < 400:
                    ejemplos.append({"partido": int(pid), "minuto": r.get("minuto"), "segundo": r.get("segundo"),
                                     "pasador": r.get("pasador_nombre"), "veredicto": r.get("posicion", r["veredicto"]),
                                     "margen_m": r.get("margen_m")})
            if borrar_360:          # 9 MB por partido: no se guardan
                sb._cache("three-sixty", f"{pid}.json").unlink(missing_ok=True)
            print(f"{pid}: {len(rev.fueras_de_lugar)} pases · {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(filas)
    sanc = df[df["real"] == "sancionado"]
    comp_ = df[df["real"] == "completado"]
    out = {"partidos": int(df["partido"].nunique()), "pases_revisados": int(len(df)), "tabla": _tabla(filas, "real"),
           "sancionados": int(len(sanc)),
           "sancionados_adelantados_o_dudosos": int(sanc["veredicto"].isin(["fuera de lugar", "dudoso"]).sum()),
           "sancionados_margen_mediano_m": round(float(sanc["margen_m"].median()), 2) if len(sanc) else None,
           "completados": int(len(comp_)),
           "completados_fuera_de_lugar": int((comp_["veredicto"] == "fuera de lugar").sum()),
           "completados_dudosos": int((comp_["veredicto"] == "dudoso").sum()),
           "sigma_mediana_m": round(float(df["sigma_m"].median()), 2), "tasas": _tasas(df),
           "ejemplos_sancionados": ejemplos[:40]}
    fa = pd.DataFrame(faltas)
    if len(fa):
        out["faltas"] = {"revisadas": int(len(fa)),
                         "ocasion_vs_tarjeta": {str(k): {str(c): int(v) for c, v in fila.items()} for k, fila in
                                                fa.groupby(["ocasion", "tarjeta"]).size().unstack(fill_value=0).iterrows()},
                         "penales_por_posicion": {str(k): {str(c): int(v) for c, v in fila.items()} for k, fila in
                                                  fa.groupby(["sancion", "area"]).size().unstack(fill_value=0).iterrows()},
                         "manos": int(fa["mano"].sum())}
    _guardar("statsbomb360", out)
    print(json.dumps({k: v for k, v in out.items() if k != "ejemplos_sancionados"}, indent=2, ensure_ascii=False))


def video(clips=("completo", "tv", "fuera_de_lugar")):
    """Decisiones sacadas del video vs las mismas jugadas con las posiciones reales (verdad sintética)."""
    import numpy as np

    from analista import Partido
    from analista.analisis import var as av
    out, filas = {}, []
    for clip in clips:
        res, gt = RAIZ / "salidas" / clip, RAIZ / "clips" / clip
        if not (res / "tracking.csv").exists() or not (gt / "gt_jugadores.csv").exists():
            continue
        p = Partido.video(res)
        rev = p.var()
        gj = pd.read_csv(gt / "gt_jugadores.csv")
        gj = gj.assign(detectado=True)
        gb = pd.read_csv(gt / "gt_balon.csv")
        pos_gt = av.Posiciones(gj, gb)
        exacto = av.ParametrosVar(p.cancha.largo, p.cancha.ancho, error_posicion_m=0.0, error_estimado_m=0.0,
                                  error_cuerpo_m=0.0, error_tiempo_s=0.0, error_balon_m=0.0)
        for r in rev.fueras_de_lugar:
            if r["veredicto"] not in ("fuera de lugar", "habilitado", "dudoso") or not r.get("receptor_xy"):
                continue
            ins = pos_gt.instante(r["t"])
            j = ins.jugadores
            def cercano(xy, eq):
                c = j[j["equipo"] == eq]
                if not len(c) or xy is None:
                    return None
                d = np.hypot(c["x"] - xy[0], c["y"] - xy[1])
                return c["id"].iloc[int(np.argmin(d))] if d.min() < 3.0 else None
            rec = cercano(r["receptor_xy"], r["equipo"])
            pj = [x for x in r.get("jugadores", []) if x["id"] == r.get("pasador")]
            pas = cercano((pj[0]["x"], pj[0]["y"]) if pj else None, r["equipo"])
            if rec is None or rec == pas:
                continue
            g_mismo = av.revisar_fuera_de_lugar(ins, r["equipo"], r["direccion"], exacto, pas, rec)
            t_gt = av.momento_del_pase(pos_gt, r["t_toque"], pas) if pas is not None else r["t"]
            g_real = av.revisar_fuera_de_lugar(pos_gt.instante(t_gt), r["equipo"], r["direccion"], exacto, pas, rec)
            filas.append({"clip": clip, "t": r["t"], "veredicto_video": r["veredicto"],
                          "margen_video": r.get("margen_centro_m", r["margen_m"]), "margen_cuerpo": r["margen_m"],
                          "sigma_video": r["sigma_m"], "margen_gt_mismo_t": g_mismo.get("margen_m"),
                          "margen_gt": g_real.get("margen_m"), "dt_pase": round(r["t"] - t_gt, 3)})
    df = pd.DataFrame(filas)
    if df.empty:
        print("Sin clips analizados")
        return
    df["error_pos"] = (df["margen_video"] - df["margen_gt_mismo_t"]).abs()
    df["error_total"] = (df["margen_video"] - df["margen_gt"]).abs()
    def correcto(r):
        if r.veredicto_video == "fuera de lugar":
            return r.margen_gt > 0
        if r.veredicto_video == "habilitado":
            return r.margen_gt <= 0
        return None
    df["correcto"] = df.apply(correcto, axis=1)
    dec = df.dropna(subset=["correcto"])
    out = {"pases": int(len(df)), "decididos": int(len(dec)), "decisiones_correctas": int(dec["correcto"].sum()),
           "dudosos": int((df["veredicto_video"] == "dudoso").sum()),
           "error_margen_mediana_m": round(float(df["error_total"].median()), 2),
           "error_margen_p90_m": round(float(df["error_total"].quantile(0.9)), 2),
           "error_posicion_mediana_m": round(float(df["error_pos"].median()), 2),
           "error_instante_mediana_s": round(float(df["dt_pase"].abs().median()), 3),
           "sigma_mediana_m": round(float(df["sigma_video"].median()), 2),
           "dentro_de_2sigma_pct": round(100 * float((df["error_total"] <= 2 * df["sigma_video"]).mean()), 1),
           "detalle": df.round(3).to_dict(orient="records")}
    _guardar("video", out)
    print(json.dumps({k: v for k, v in out.items() if k != "detalle"}, indent=2, ensure_ascii=False))
    print(df.round(2).to_string())


def tiros():
    """Remates de StatsBomb (todas las competiciones en caché, foto del tiro con nombres) contra los
    fueras de lugar que el árbitro pitó justo después de un remate (rebote o interferencia)."""
    import glob

    from analista.analisis.var import parametros_para
    from analista.datos.cache import carpeta_cache
    from analista.var_partido import _tiro_statsbomb
    p = parametros_para("statsbomb360", 1.0)
    filas, t0 = [], time.time()
    archivos = sorted(glob.glob(str(carpeta_cache() / "statsbomb" / "events" / "*.json")))
    for k, f in enumerate(archivos):
        crudos = json.loads(Path(f).read_text(encoding="utf-8"))
        equipos = []
        for e in crudos:
            if e["team"]["id"] not in equipos:
                equipos.append(e["team"]["id"])
        for e in crudos:
            sh = e.get("shot") or {}
            if e["type"]["name"] != "Shot" or not sh.get("freeze_frame") or (sh.get("type") or {}).get("name") != "Open Play" \
                    or e["period"] >= 5:
                continue
            t = e["minute"] * 60 + e["second"]
            pit = [g for g in crudos[e["index"]: e["index"] + 12] if g["type"]["name"] == "Offside"
                   and g["team"]["id"] == e["team"]["id"] and g["minute"] * 60 + g["second"] - t <= 10]
            eq = "A" if e["team"]["id"] == equipos[0] else "B"
            r = _tiro_statsbomb(e, crudos, eq, p)
            ade = {a["id"] for a in (r or {}).get("adelantados", [])}
            jugador_pitado = (pit[0].get("player") or {}).get("id") if pit else None
            filas.append({"pitado": bool(pit), "veredicto": (r or {}).get("veredicto", "nadie adelantado"),
                          "motivo": (r or {}).get("motivo"),
                          "pitado_estaba_adelantado": (jugador_pitado in ade) if pit else None})
        if k % 50 == 0:
            print(f"{k}/{len(archivos)} partidos · {len(filas)} remates · {time.time() - t0:.0f} s", flush=True)
    df = pd.DataFrame(filas)
    pit = df[df["pitado"]]
    out = {"partidos": len(archivos), "remates": int(len(df)), "pitados_tras_remate": int(len(pit)),
           "tabla": _tabla([{**r, "real": "pitado" if r["pitado"] else "no pitado"} for r in filas], "real"),
           "pitados_con_el_jugador_adelantado_pct": round(100 * float(pit["pitado_estaba_adelantado"].mean()), 1) if len(pit) else None,
           "pitados_marcados_pct": round(100 * float(pit["veredicto"].isin(["fuera de lugar", "revisar", "dudoso"]).mean()), 1) if len(pit) else None,
           "no_pitados_marcados_pct": round(100 * float(df[~df["pitado"]]["veredicto"].isin(["fuera de lugar", "revisar"]).mean()), 2)}
    _guardar("tiros", out)
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    que = sys.argv[1] if len(sys.argv) > 1 else "skillcorner"
    {"skillcorner": skillcorner, "statsbomb": statsbomb, "video": video, "tiros": tiros}[que]()
