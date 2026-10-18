# Cambios

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
