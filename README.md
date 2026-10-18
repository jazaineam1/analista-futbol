# Analista de fútbol

**Del partido a los datos.** Analista lee un partido desde el video (la señal del televisor por una capturadora HDMI,
la página donde lo ves en vivo en el PC o un video que subes) y ubica a cada jugador y el balón en metros. Con eso, y con
datos abiertos de eventos (StatsBomb) y de tracking (SkillCorner), calcula xG, posesión, presión, formación y control de
cancha, **revisa las jugadas como un VAR** (fuera de lugar con su margen y su error, goles, balón fuera, faltas en el área)
y entrega reportes listos para compartir.

[![pruebas](https://github.com/jazaineam1/analista-futbol/actions/workflows/pruebas.yml/badge.svg)](https://github.com/jazaineam1/analista-futbol/actions/workflows/pruebas.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-1f6f4a)
![Licencia](https://img.shields.io/badge/licencia-MIT-1f6f4a)

![El panel: datos de la final del Mundial 2022, tracking de la A-League y el análisis de una transmisión](docs/media/demo.gif)

[Sitio](https://jazaineam1.github.io/analista-futbol/) ·
[Guía de uso](https://jazaineam1.github.io/analista-futbol/guia.html) ·
[VAR: jugadas revisadas](https://jazaineam1.github.io/analista-futbol/var/) ·
[Cómo detecta, etapa por etapa](https://jazaineam1.github.io/analista-futbol/deteccion/) ·
[Video de la demo](docs/media/demo.mp4)

## Qué analiza

| Fuente | Qué trae | Qué sale |
|---|---|---|
| **Tu video** (subido, capturadora HDMI del televisor, la página en vivo en tu pantalla, `rtsp://`) | Lo que se ve en la imagen | Posiciones en metros de los jugadores y el balón, equipos por color, posesión, pases, presión, bloque, físico, goles, VAR, video anotado con minimapa |
| **Eventos · StatsBomb Open Data** | Casi 4.000 partidos: cada pase, tiro, presión y duelo | xG (de StatsBomb y de un modelo propio), flujo de xG, mapa de tiros, redes de pases, PPDA, field tilt, pases progresivos |
| **Tracking · SkillCorner Open Data** | 20 partidos de la A-League 2024/25, 22 jugadores y balón a 10 Hz | Posesión, altura del bloque, ancho, formación sin balón, control de cancha en los goles, distancia y velocidad máxima |

Todo llega al mismo esquema (metros, origen en el centro de la cancha, `A` local y `B` visitante), así que el mismo
código analiza cualquiera de las tres fuentes.

## Empezar

```bash
pip install "analista-futbol[todo] @ git+https://github.com/jazaineam1/analista-futbol"
analista panel
```

El panel se abre en el navegador (`http://localhost:8765`) con siete pestañas: **Datos de partidos**, **Fuente**,
**Calibrar**, **Analizar**, **Resultados**, **VAR** y **Cómo detecta**. No sube nada a internet.

En **Fuente** eliges de dónde viene el partido:

- **Subir un video**: arrastrar y soltar, con barra de progreso; quedan en *Videos del proyecto*.
- **Pantalla del PC**: la página (o app) donde ves el partido en vivo. Eliges el monitor, la app saca una foto de la
  pantalla y marcas el reproductor arrastrando un rectángulo, o pulsas *Detectar el reproductor* (busca la zona que se
  mueve). Con un solo monitor, la foto espera unos segundos para que pases a la ventana del partido.
- **Televisor**: la salida HDMI del decodificador o TV box a una capturadora HDMI-USB; la app lista las capturadoras
  con su nombre y una miniatura.
- **Ruta o URL**: un archivo del disco, `disp:N`, `pantalla:M:x,y,ancho,alto` o un stream `rtsp://`.

En vivo se calibra sobre un cuadro congelado y, si quieres, la señal se graba mientras se analiza para revisarla
después (también en el VAR). La app no evita protecciones de contenido: una señal con HDCP o DRM se ve negra.

Probado de verdad (`demo/probar_captura.py`): un navegador reproduce el clip de TV en una pantalla, la app encuentra el
reproductor sola (a 7 px del real), se calibra sobre el cuadro congelado (0,06 m) y analiza 30 s en vivo mientras graba
la señal: 3,2 cuadros/s, 66% de los jugadores detectados y 1,1 m de error, con el navegador, la grabación y el análisis
compartiendo los mismos 2 núcleos. Con GPU o analizando la grabación después se llega a las cifras de la tabla de abajo.

Desde la línea de comandos:

```bash
analista datos statsbomb reporte 3869685 --competicion 43 --temporada 106   # final del Mundial 2022
analista datos skillcorner reporte 1886347                                    # Auckland FC – Newcastle Jets
analista analizar partido.mp4 -c calibracion.json -o resultados               # tu video
analista fuentes --sugerir 1                                                  # monitores, capturadoras y dónde está el reproductor
analista vivo pantalla:1:160,120,1280,720 -c calibracion.json --grabar vivo.mp4   # la página del partido, en vivo
analista vivo disp:0 -c calibracion.json                                      # en vivo, desde la capturadora
analista var resultados/                                                      # VAR de tu video (también skillcorner:ID, statsbomb:ID)
```

Y desde Python:

```python
from analista import Partido
from analista import graficos as gr

final = Partido.statsbomb(3869685)
final.resumen()["por_equipo"]["A"]["xg"]      # 2.76
gr.red_de_pases(final, "A")

aleague = Partido.skillcorner(1886347)
aleague.formacion("A")["formacion"]
gr.control_de_cancha(aleague, t=5059.9)

mio = Partido.video("resultados/")
mio.reporte("mi_partido.html")
```

Hay notebooks de ejemplo en [`notebooks/`](notebooks) y la [guía](docs/guia.html) explica cómo conectar el televisor,
calibrar bien y qué mide cada métrica.

## VAR

La app revisa las jugadas como un asistente de video, con los datos que tiene (pestaña **VAR** del panel, sección VAR de
los reportes y `analista var`):

| Revisión | Cómo |
|---|---|
| Fuera de lugar en cada pase | El instante exacto en que el balón sale del pie (entre dos cuadros si hace falta), la línea del penúltimo defensor o del balón, y el margen del receptor con su incertidumbre. Con video, medido con la parte del cuerpo más adelantada (la silueta, sin los brazos). Reglas de la IFAB: en línea es habilitado, nada en campo propio ni en saques de banda, de meta o de esquina, y si el pase no le llega no intervino. |
| Fuera de lugar en remates | El que estaba adelantado cuando su compañero remató y juega el rebote tras una atajada, un palo o un desvío; y el que interfiere con el arquero (le tapa la visión, está en la trayectoria o encima de él). |
| Goles | Si el balón cruzó entero la línea entre los postes, y la jugada previa desde que el equipo recuperó el balón. |
| Balón fuera | Balones que cruzaron la línea un instante y volvieron sin que se parara el juego. |
| Faltas | Dentro o fuera del área (la línea es parte del área), manos y tarjetas, y si cortaron una **ocasión manifiesta de gol** (criterios de la Ley 12 medidos con posiciones). En tu video se marcan con un clic. |
| Lo que no se ve | Arquero fuera de imagen: se supone en su arco. Defensor que podría estar fuera de cámara: "fuera de lugar" pasa a dudoso. Pase hacia donde la cámara no mira: no revisable. |

Si el margen cae dentro de ±2σ la jugada es **dudosa**. Cada jugada tiene una **repetición sobre el video** (cámara lenta,
imagen congelada con zoom y las líneas trazadas en perspectiva sobre el césped, como en la TV) y una **reconstrucción 2D**
animada; la revisión manual va **cuadro a cuadro** sobre tu video.

<!-- VALIDACION_VAR -->
Validación del VAR (ver la [página del VAR](https://jazaineam1.github.io/analista-futbol/var/#validacion)):

| Medida | Resultado | Detalle |
|---|---|---|
| Video contra la verdad (clips sintéticos) | 14 de 14 decisiones correctas, 4 dudosas | error mediano del margen 0,29 m |
| Fuera de lugar pitado, desde el video | +0,04 m (verdad −0,01 m) | ±0,31 m: dudoso, como corresponde a una jugada en línea |
| Línea contra la de SkillCorner | 99,6 % del mismo lado | 8543 pases, diferencia mediana 0,00 m |
| Remates de StatsBomb con fuera de lugar pitado después | 76 % marcados | 96 casos; el sancionado estaba adelantado en la foto del tiro en el 77 %; en los otros 13.860 remates marca el 1,3 % |
| Faltas: dentro o fuera del área (StatsBomb) | 33 de 34 penales dentro | ningún tiro libre ubicado dentro del área; los que caen sobre la línea salen como dudosos |
| Pitados por el árbitro, SkillCorner | 11,1 % adelantados | contra 1,4 % en pases completados (54 pitados) |
| Pitados por el árbitro, StatsBomb 360 | 77,4 % adelantados | contra 1,6 % en pases completados (325 pitados) |
<!-- /VALIDACION_VAR -->

## Resultados medidos

<!-- VALIDACION -->
Medido contra la verdad de cada cuadro en dos transmisiones generadas con el tracking real de SkillCorner (ver [validación](https://jazaineam1.github.io/analista-futbol/#validacion)):

| Medida | Plano completo (cámara fija) | Cámara de TV (paneo y zoom) | Qué significa |
|---|---|---|---|
| Jugadores detectados | 97,4% | 91,1% | de los que se ven en cada cuadro |
| Detecciones correctas | 99,4% | 86,7% | lo detectado que era un jugador real, a ≤2 m |
| Error de posición | 0,13 m | 0,55 m | mediana; el 90% bajo 0,22 m y 1,59 m |
| Equipo correcto | 99,5% | 95,4% | jugadores de campo, por el color de la camiseta |
| Identidad estable | 99,0% | 91,4% | una pista sigue siempre al mismo jugador |
| Balón ubicado | 36% | 44% | cuadros con balón visible en que la app lo ubica a ≤2 m |
| Cámara calibrada | 100% | 100% | error mediano en la cancha: 0,06 m y 0,97 m |
| Posesión de Auckland | 57% / 90% | 61% / 84% | estimada / real, en los cuadros analizados |
| Goles detectados | 1 de 1 | 2 de 2 | el balón cruza la línea entre los postes (±2 s) |
| Velocidad | 1,3 cuadros/s | 3,1 cuadros/s | CPU de 2 núcleos sin GPU, a 1920 y 1280 px |

Con una sola calibración (sin calibrar la toma que sigue a la repetición), la cámara de TV queda en 0,59 m de error de posición y 91,2% de jugadores detectados: tras el corte la app reubica la cámara comparando el césped y las líneas con el cuadro calibrado (94% de los cuadros calibrados, error de la cámara 1,13 m).
<!-- /VALIDACION -->

## Cómo funciona con video

1. **Césped y zona de juego**: lo verde forma la cancha; tribuna, bancos y gráficos de TV quedan fuera.
2. **Líneas**: las líneas blancas pintadas, con un filtro morfológico.
3. **Calibración**: una homografía de 4 o más clics, ajustada contra las líneas; con cámara de TV se sigue el paneo y el
   zoom con flujo óptico y se re-ancla tras un corte.
4. **Personas**: YOLO marca a cada persona; las que están fuera de la cancha se descartan.
5. **Equipos**: color del torso en espacio Lab y k-means, sin entrenar nada; árbitro y arqueros quedan como "otro".
6. **Balón**: candidatos de la red y manchas claras y redondas sobre el césped, con un seguidor que exige coherencia.
7. **A metros**: el pie de cada jugador se proyecta a la cancha.
8. **Seguimiento**: Kalman y asociación en dos etapas en metros, para que cada jugador conserve su número.
9. **Métricas**: posesión por proximidad, pases, presión, bloque, forma, físico, eventos de gol.
10. **Reporte**: HTML autocontenido y tablas CSV.

La pestaña **Cómo detecta** muestra cada etapa sobre el cuadro que elijas.

## Estructura

```
analista/
  vision/      del cuadro de video a posiciones en metros (fuentes, cámara, detección, equipos, seguimiento)
  datos/       StatsBomb, SkillCorner y resultados de video en el esquema común
  analisis/    métricas de eventos y de tracking, modelo xG, VAR
  reportes/    reportes HTML
  panel/       panel web local (Flask)
  partido.py   API de alto nivel: Partido.statsbomb / .skillcorner / .video (y .var())
  graficos.py  gráficos con matplotlib
demo/          scripts que generan los ejemplos, la validación, el sitio y el video de la demo
docs/          sitio (GitHub Pages)
notebooks/     ejemplos en Jupyter
tests/         pruebas (pytest)
```

## Desarrollo

```bash
pip install -e ".[todo,dev]"
pytest            # pruebas sin internet ni GPU (las de pantalla se saltan si no hay una)
ruff check analista tests demo
```

Pruebas de punta a punta en un navegador real (Playwright + Chromium; las de pantalla necesitan un `DISPLAY`):

```bash
analista panel --carpeta /tmp/p --puerto 8899 --sin-navegador &
python demo/probar_panel.py --url http://localhost:8899 --video demo/clips/tv/video.mp4 --calibracion demo/clips/tv/calibracion.json
python demo/probar_captura.py --url http://localhost:8899 --clip demo/clips/tv   # la página del partido en pantalla, en vivo
bash demo/probar_cli.sh /tmp/prueba_cli                                          # todos los comandos
```

Para regenerar los ejemplos y la validación:

```bash
python demo/extraer_sprites.py --videos left_camera.mp4 right_camera.mp4 --modelo yolo11s.pt
python demo/generar_transmision.py --camara completa --tramos 50377-50657 -o demo/clips/completo
python demo/generar_transmision.py --camara tv --tramos 50377-50657,53770-53858 --marcadores 0-0,1-0 -o demo/clips/tv
analista analizar demo/clips/tv/video.mp4 -c demo/clips/tv/calibracion.json -o demo/salidas/tv
python demo/evaluar.py --resultados demo/salidas/tv --gt demo/clips/tv
python demo/validar_var.py skillcorner|statsbomb|tiros|video
python demo/construir_sitio.py && python demo/construir_var.py
```

## Límites

- **Protección de contenido**: la app lee solo lo que tu equipo puede reproducir. Una señal con HDCP se ve negra en la
  capturadora y la app no la evita.
- **Detector general**: YOLO entrenado con COCO no distingue arquero, árbitro y jugador. Un modelo afinado para fútbol
  mejoraría la detección del balón y de los jugadores lejanos.
- **Balón en el aire**: con una sola cámara no se puede ubicar en metros.
- **Deriva de la cámara de TV**: siguiendo el paneo, la calibración se desvía con el tiempo (en la prueba, 1 m de error mediano en tomas de hasta 28 s). Tras un corte la app reubica la cámara comparando con el cuadro calibrado; calibrar cada toma sigue siendo lo más preciso.
- **Posesión con balones aéreos**: un balón alto se proyecta al suelo lejos de donde está; en centros y remates desvía la posesión.
- **VAR con una cámara**: la cámara de TV no ve a todos; la parte del cuerpo más adelantada es una aproximación por la
  silueta; el contacto de una falta no se ve en las posiciones, así que las faltas vienen de los eventos o se marcan a mano
  (detectar posibles penales por frenadas con un rival encima acertaba menos del 10 % y se descartó).

## Datos y licencias

- Código: MIT. El detector opcional usa [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0).
- [StatsBomb Open Data](https://github.com/statsbomb/open-data): uso educativo y de investigación, citando a Hudl StatsBomb.
- [SkillCorner Open Data](https://github.com/skillcorner/opendata): MIT.
- [SoccerTrack](https://github.com/keisuke198619/SoccerTrack): clip real de ejemplo y recortes de jugadores para las
  transmisiones sintéticas.
