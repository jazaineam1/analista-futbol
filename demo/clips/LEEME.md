# Clips de ejemplo

| Carpeta | Qué es | Origen |
|---|---|---|
| `completo/` | Transmisión sintética de 28 s con cámara fija que ve toda la cancha (1920×1080). El gol de N. Moreno. | Posiciones reales de SkillCorner Open Data (Auckland FC – Newcastle Jets, A-League 2024/25, MIT) |
| `tv/` | Transmisión sintética de 38 s con la cámara de TV (1280×720): paneo y zoom reales, marcador, una repetición y dos goles. | Ídem; la cámara repite el movimiento de la transmisión original |
| `real/` | 10 s de un partido universitario real con cámara fija gran angular (3072×1728). | SoccerTrack (MIT, Atom Scott) |

Cada clip sintético trae la verdad de cada cuadro: `gt_jugadores.csv`, `gt_balon.csv` y `gt_camara.csv`, más
`calibracion.json` (clics simulados con 2 px de error) e `info.json` (tramos y goles). Los jugadores son recortes de
jugadores reales de SoccerTrack (`demo/sprites`).

Se regeneran con `demo/generar_transmision.py` y `demo/soccertrack_calibracion.py`.
