"""Esquema común de datos.

Todo el paquete trabaja con tres tablas, vengan del video, de StatsBomb, de SkillCorner
o de cualquier otro proveedor que se convierta a este formato:

**tracking** (una fila por jugador y cuadro)
    cuadro, t (s), id, equipo ('A' local / 'B' visitante / 'otro'), rol ('campo', 'arquero',
    'arbitro/otro'), x, y (m) — opcionales: periodo, nombre, numero, x_suave, y_suave, detectado

**balon** (una fila por cuadro con balón)
    cuadro, t, x, y — opcional z, detectado

**eventos** (una fila por acción)
    id, periodo, minuto, segundo, t (s), equipo ('A'/'B'), equipo_nombre, jugador, jugador_id,
    posicion, tipo (en español: pase, tiro, conduccion, presion, recuperacion, duelo, ...),
    tipo_original, subtipo, x, y, x_fin, y_fin (m), exito (bool), resultado, xg, xg_modelo,
    cuerpo, bajo_presion, receptor, jugada, posesion, equipo_posesion, duracion

Coordenadas en metros con origen en el centro de la cancha:
  - tracking y balón: el equipo A ataca siempre hacia +x (se voltea el segundo tiempo);
  - eventos: cada acción está en la perspectiva del equipo que la hace (ataca hacia +x),
    que es como las publican StatsBomb y la mayoría de proveedores de eventos.
"""
from __future__ import annotations

import pandas as pd

COLUMNAS_TRACKING = ["cuadro", "t", "id", "equipo", "rol", "x", "y"]
COLUMNAS_BALON = ["cuadro", "t", "x", "y"]
COLUMNAS_EVENTOS = ["id", "periodo", "minuto", "segundo", "t", "equipo", "jugador", "tipo",
                    "x", "y", "x_fin", "y_fin", "exito"]

TIPOS = {
    "Pass": "pase", "Shot": "tiro", "Carry": "conduccion", "Pressure": "presion",
    "Ball Recovery": "recuperacion", "Duel": "duelo", "Interception": "intercepcion",
    "Clearance": "despeje", "Block": "bloqueo", "Dribble": "regate", "Dribbled Past": "regateado",
    "Foul Committed": "falta", "Foul Won": "falta_recibida", "Miscontrol": "mal_control",
    "Dispossessed": "desposesion", "Goal Keeper": "arquero", "Ball Receipt*": "recepcion",
    "50/50": "dividida", "Offside": "fuera_de_juego", "Own Goal Against": "autogol_en_contra",
    "Own Goal For": "autogol_a_favor", "Substitution": "cambio", "Tactical Shift": "ajuste_tactico",
    "Starting XI": "alineacion", "Half Start": "inicio_periodo", "Half End": "fin_periodo",
    "Error": "error", "Shield": "proteccion", "Injury Stoppage": "pausa_lesion",
    "Referee Ball-Drop": "bote_neutral", "Bad Behaviour": "conducta", "Player On": "jugador_entra",
    "Player Off": "jugador_sale",
}


def validar(df: pd.DataFrame, columnas: list[str], nombre: str) -> pd.DataFrame:
    faltan = [c for c in columnas if c not in df.columns]
    if faltan:
        raise ValueError(f"La tabla de {nombre} no tiene las columnas {faltan}")
    return df


def validar_tracking(df: pd.DataFrame) -> pd.DataFrame:
    return validar(df, COLUMNAS_TRACKING, "tracking")


def validar_eventos(df: pd.DataFrame) -> pd.DataFrame:
    return validar(df, COLUMNAS_EVENTOS, "eventos")
