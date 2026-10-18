#!/usr/bin/env bash
# Prueba de punta a punta de la línea de comandos: cada comando debe terminar bien y dejar sus archivos.
#   bash demo/probar_cli.sh /tmp/prueba_cli
set -u
DEST=${1:-/tmp/prueba_cli}
RAIZ=$(cd "$(dirname "$0")/.." && pwd)
MODELO=${MODELO:-$RAIZ/modelos/yolo11n.pt}
rm -rf "$DEST"; mkdir -p "$DEST"; cd "$DEST"
fallas=0
paso() {  # paso "nombre" archivo_esperado comando...
  local nombre=$1 esperado=$2; shift 2
  local t0=$SECONDS
  if "$@" > "$DEST/log_$(echo "$nombre" | tr ' /' '__').txt" 2>&1 && { [ -z "$esperado" ] || [ -s "$esperado" ]; }; then
    echo "  ok    $nombre ($((SECONDS - t0)) s)"
  else
    echo "  FALLA $nombre"; tail -5 "$DEST/log_$(echo "$nombre" | tr ' /' '__').txt" | sed 's/^/        /'; fallas=$((fallas + 1))
  fi
}
CLIP=$RAIZ/demo/clips/tv
paso "version"                     ""                       analista --version
paso "ayuda"                       ""                       analista --help
paso "statsbomb competiciones"     ""                       analista datos statsbomb competiciones
paso "statsbomb partidos"          ""                       analista datos statsbomb partidos 43 106
paso "statsbomb reporte"           statsbomb_3869685.html   analista datos statsbomb reporte 3869685 --competicion 43 --temporada 106
paso "statsbomb eventos csv"       statsbomb_3869685_eventos.csv analista datos statsbomb eventos 3869685
paso "skillcorner partidos"        ""                       analista datos skillcorner partidos
paso "skillcorner reporte"         skillcorner_1886347.html analista datos skillcorner reporte 1886347
PUNTOS=$(python - "$CLIP/calibracion.json" <<'EOF'
import json, sys
k = json.load(open(sys.argv[1]))["claves"][0]["puntos"]
print(";".join(f"{p['imagen'][0]},{p['imagen'][1]}={p['cancha'][0]},{p['cancha'][1]}" for p in k))
EOF
)
paso "calibrar"                    cal.json                 analista calibrar "$CLIP/video.mp4" --puntos "$PUNTOS" --largo 104 -o cal.json
paso "verificar"                   verificacion.jpg         analista verificar "$CLIP/video.mp4" -c cal.json -t 0
paso "analizar (4 s)"              res/reporte.html         analista analizar "$CLIP/video.mp4" -c cal.json -o res --fin 4 --modelo "$MODELO" \
                                                            --equipos "Auckland=#1f6fd1,Newcastle=#f2f2f2"
paso "analizar sin YOLO"           res2/tracking.csv        analista analizar "$CLIP/video.mp4" -c cal.json -o res2 --fin 3 --detector cesped --sin-video
paso "reporte"                     res/reporte.html         analista reporte res --titulo "Prueba"
paso "explicar"                    exp/pagina.html          analista explicar "$CLIP/video.mp4" -c cal.json -t 2 -o exp/pagina.html --modelo "$MODELO"
paso "var del video"               res/var.html             analista var res
paso "var skillcorner"             var_sk.html              analista var skillcorner:1886347 -o var_sk.html
paso "var statsbomb"               var_sb.html              analista var statsbomb:3869685 -o var_sb.html
paso "fuentes"                     ""                       analista fuentes
if [ -n "${DISPLAY:-}" ]; then
  # en vivo desde la pantalla, 6 s, grabando la señal (termina solo con timeout -s INT, como Ctrl+C)
  paso "vivo pantalla y grabar"    vivo/reporte.html        timeout --preserve-status -s INT 25 analista vivo "pantalla:1:0,0,1280,720" -c cal.json -o vivo \
                                                            --fps 4 --grabar vivo.mp4 --sin-video --sin-ventana --modelo "$MODELO"
fi
paso "error claro sin calibración" ""                       bash -c "! analista analizar '$CLIP/video.mp4' -c no_existe.json -o x 2>/dev/null"
echo
echo "$fallas fallas"
exit $fallas
