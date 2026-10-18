# Cambios

## 0.4.0 — 2026-10-18

Captura integral y revisión tipo VAR.

- **Fuente de video en el panel, rediseñada**: subir un video desde el navegador (arrastrar y soltar, con
  barra de progreso y lista de videos del proyecto); **pantalla del PC** para la página donde se ve el partido
  en vivo (elegir monitor, foto de la pantalla, marcar el reproductor arrastrando o **detectarlo solo** por lo que
  se mueve); **televisor** por capturadora HDMI con la lista de dispositivos, su nombre y una miniatura; y ruta o URL.
- **En vivo**: la captura corre en su propio hilo y, si se pide, **graba la señal** en tiempo real mientras se
  analiza (queda en *Videos del proyecto* para revisarla o volver a analizarla). El video anotado dura lo mismo que
  el partido aunque el análisis vaya más lento. `Ctrl+C` en `analista vivo` guarda lo analizado.
- **Calibración en vivo** sobre un cuadro congelado, que queda como referencia: la cámara se reubica sola al empezar
  y después de cada corte comparando el césped y las líneas con ese cuadro (ORB + RANSAC).
- **En vivo más rápido y estable**: el detector se precalienta antes del primer cuadro, la cámara se reubica sola
  tras un salto de tiempo o si la calidad cae, guarda referencias recientes, la grabación usa un códec más liviano y el
  panel analiza a 960 px por defecto en vivo. Prueba real con la página del partido en pantalla
  (`demo/probar_captura.py`): de 0,6 a 3,2 cuadros/s y de 8–13% a 66% de jugadores detectados.
- La instalación incluye los archivos de la repetición 2D (`reportes/activos`), con una prueba que lo vigila.
- **VAR** (`analista.analisis.var`, pestaña *VAR*, `analista var`, sección en los reportes):
  - fuera de lugar en el instante exacto del pase (línea del penúltimo defensor o del balón, margen ± incertidumbre,
    reglas de la IFAB), con video medido con la **parte del cuerpo más adelantada** (silueta, sin los brazos);
  - **fuera de lugar en remates**: el adelantado que juega el rebote tras una atajada, palo o desvío, y el que
    interfiere con el arquero (le tapa la visión, está en la trayectoria o encima de él);
  - línea de gol y **jugada previa** de cada gol (pases y remates desde la recuperación), balones que salieron sin
    que se detuviera el juego;
  - faltas dentro o fuera del área, **manos** y tarjetas (StatsBomb), y **ocasión manifiesta de gol** con los
    criterios de la Ley 12 medidos con posiciones;
  - revisión manual **cuadro a cuadro** sobre el video (fuera de lugar en el cuadro elegido, y falta marcada con un
    clic: dentro o fuera del área y ocasión manifiesta).
  Funciona con el video (líneas dibujadas en perspectiva sobre la imagen), con SkillCorner y con StatsBomb (capturas
  360 y fotos de cada remate).
- **Clips VAR sobre el video**, como en la TV: cámara lenta hasta el instante revisado, imagen congelada con zoom,
  líneas que se trazan de banda a banda en perspectiva, el margen que cuenta hasta su valor y el veredicto (MP4 en la
  carpeta de resultados, en el panel y en el reporte).
- **Repetición 2D estilo transmisión**: césped con franjas, cámara que sigue al balón y hace zoom a la jugada, imagen
  congelada con destello, líneas que se dibujan, contador del margen, sello del veredicto, mapa de la cancha con la
  zona que se ve, vista inclinada como la cámara de TV, altura del balón con su sombra, estelas, cuadro a cuadro y
  atajos de teclado. Respeta *movimiento reducido*.
- **Validación del VAR** (`demo/validar_var.py`): video contra la verdad, geometría contra SkillCorner, fueras de lugar
  pitados en SkillCorner y StatsBomb 360, y 13.956 remates de StatsBomb contra los fueras de lugar pitados después.
- `analista fuentes` lista monitores y capturadoras (`--sugerir N` encuentra el reproductor en el monitor N).
- `mss` y `pillow` pasan a ser dependencias base.

## 0.3.0 — 2026-10-17

Primera versión como producto instalable.

- **Datos de partidos**: StatsBomb Open Data (eventos) y SkillCorner Open Data (tracking) con un
  esquema común en metros; `Partido.statsbomb(...)`, `Partido.skillcorner(...)`, `Partido.video(...)`.
- **Análisis**: xG con modelo propio validado fuera de muestra, flujo de xG, mapa de tiros, PPDA,
  field tilt, pases progresivos, redes de pases, recuperaciones altas; posesión por proximidad,
  presión, altura del bloque, ancho, formación, físico y control de cancha a partir del tracking.
- **Panel web** con pestaña *Datos de partidos* para explorar competiciones y generar reportes.
- **CLI** `analista` con `panel`, `analizar`, `vivo`, `calibrar`, `verificar`, `explicar`,
  `reporte` y `datos`.
- **Gráficos** con matplotlib (`analista.graficos`) y notebooks de ejemplo.
- Reportes HTML autocontenidos con tema claro y oscuro.
- Pruebas automáticas (pytest) e integración continua en Linux y Windows.

## 0.2.0

- Vista paso a paso de la detección (10 etapas) en el panel y como página independiente.
- Validación contra transmisiones sintéticas con verdad conocida.

## 0.1.0

- Prototipo: detección con YOLO, equipos por color, calibración por clics, seguimiento con
  Kalman en metros, minimapa y métricas básicas.
