"""Datos de prueba sintéticos: no hace falta internet ni GPU para correr las pruebas."""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from analista.cancha import Cancha

_ids = itertools.count()


def _ev(tipo, equipo, minuto, segundo, loc, jugador="J1", periodo=1, **extra):
    e = {"id": f"e{next(_ids)}", "index": next(_ids), "period": periodo, "minute": minuto, "second": segundo,
         "timestamp": f"00:{minuto:02d}:{segundo:02d}.000", "type": {"name": tipo},
         "team": {"name": equipo}, "possession_team": {"name": equipo}, "play_pattern": {"name": "Regular Play"},
         "player": {"id": hash(jugador) % 1000, "name": jugador}, "position": {"name": "Center Forward"},
         "location": loc}
    e.update(extra)
    return e


@pytest.fixture
def crudos_statsbomb():
    """Un mini partido en el formato de StatsBomb: Local 2 - 1 Visita."""
    L, V = "Local", "Visita"
    ev = [_ev("Starting XI", L, 0, 0, None), _ev("Starting XI", V, 0, 0, None)]
    rng = np.random.default_rng(0)
    nombres = {L: [f"L{i}" for i in range(1, 8)], V: [f"V{i}" for i in range(1, 8)]}
    for m in range(1, 60):
        eq = L if m % 3 else V
        a, b = rng.choice(nombres[eq], 2, replace=False)
        x0, y0 = rng.uniform(20, 100), rng.uniform(5, 75)
        ok = rng.random() < 0.8
        ev.append(_ev("Pass", eq, m, 5, [x0, y0], a, **{"pass": {
            "end_location": [min(119, x0 + rng.uniform(-5, 25)), rng.uniform(5, 75)], "recipient": {"name": b},
            **({} if ok else {"outcome": {"name": "Incomplete"}})}}))
        ev.append(_ev("Pressure", V if eq == L else L, m, 8, [rng.uniform(30, 110), rng.uniform(5, 75)], "P1"))
    def tiro(eq, m, loc, gol, xg, quien):
        return _ev("Shot", eq, m, 30, loc, quien, shot={
            "outcome": {"name": "Goal" if gol else "Saved"}, "statsbomb_xg": xg, "end_location": [120, 40, 1],
            "body_part": {"name": "Right Foot"}, "type": {"name": "Open Play"}, "technique": {"name": "Normal"},
            "freeze_frame": [{"location": [118, 40], "teammate": False, "position": {"name": "Goalkeeper"}},
                             {"location": [110, 42], "teammate": False, "position": {"name": "Center Back"}}]})
    ev += [tiro(L, 10, [108, 38], True, 0.35, "L9"), tiro(L, 30, [100, 50], False, 0.05, "L10"),
           tiro(V, 50, [109, 41], True, 0.45, "V9"), tiro(L, 70, [111, 40], True, 0.6, "L9"),
           _ev("Substitution", L, 30, 0, None, "L1", substitution={"replacement": {"name": "L12"}})]
    return ev


@pytest.fixture
def eventos(crudos_statsbomb):
    from analista.datos import statsbomb as sb
    return sb.eventos(crudos=crudos_statsbomb, local="Local")


@pytest.fixture
def tracking_sintetico():
    """20 s a 10 Hz: A (ataca a +x) avanza, B defiende; el balón va de un A a otro."""
    rng = np.random.default_rng(1)
    filas, bal = [], []
    base_A = [(-45, 0)] + [(-20 + 8 * (i % 4), -25 + 12 * (i // 4) + 2 * (i % 2)) for i in range(10)]
    base_B = [(45, 0)] + [(10 + 8 * (i % 4), -25 + 12 * (i // 4)) for i in range(10)]
    for k in range(200):
        t = k / 10
        for eq, base in (("A", base_A), ("B", base_B)):
            for i, (x, y) in enumerate(base):
                avance = 0.5 * t if eq == "A" and i else 0.0
                filas.append({"cuadro": k, "t": t, "id": f"{eq}{i}", "equipo": eq, "rol": "arquero" if i == 0 else "campo",
                              "x": x + avance + rng.normal(0, 0.05), "y": y + rng.normal(0, 0.05)})
        portador = base_A[1 + (k // 40) % 10]
        bal.append({"cuadro": k, "t": t, "x": portador[0] + 0.5 * t + 0.5, "y": portador[1]})
    return pd.DataFrame(filas), pd.DataFrame(bal)


@pytest.fixture
def cancha():
    return Cancha()
