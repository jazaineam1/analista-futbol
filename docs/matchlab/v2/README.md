# MatchLab 2.0 · Prototipo multiliga funcional

## Qué se entrega

- `MatchLab_2_Multiliga.html`: aplicación **autocontenida**, sin instalación, con modo demostrativo de Colombia incorporado. También `index.html` para hosting estático.
- `demo_colombia.json`: foto fija de Liga BetPlay II 2026 actualizada con cuatro marcadores del 20/10/2026, con 20 clubes, 50 encuentros pendientes y rutas testigo precalculadas para los 20 clubes bajo un modelo de marcadores limitado.
- `app_template.html` + `build.py`: fuentes de compilación para insertar una instantánea sin dependencias del navegador.
- `test_ui.py` + `test_historical.py`: pruebas de interacción mediante Playwright, con casos de selección de equipos, rutas, puntajes, edición, importación genérica y parsing histórico.

## Producto disponible

1. Selecciona una **competición/temporada**. Colombia funciona sin Internet con un corte de octubre de 2026.
2. Inglaterra (Premier League), España (LaLiga), Italia (Serie A), Alemania (Bundesliga), Francia (Ligue 1) y Portugal (Primeira Liga) se pueden cargar, con **conexión a Internet**, desde los resultados históricos públicos de OpenFootball 2024/25, recreando la clasificación a mitad/final de temporada. No son ligas actuales ni feeds en vivo.
3. Selecciona **cualquier equipo** de la competición. La tabla, el calendario, las métricas, los escenarios y las estimaciones cambiarán al club elegido.
4. Escoge Tabla actual, Gana lo tuyo, Mejor camino, Peor camino, Todos empatan o Pronóstico modal. Los resultados se reparten **por partido**, nunca por club de manera independiente.
5. Ajusta los marcadores, filtra por equipo o rivales directos y recalcula inmediatamente los puntos de todos.
6. Simula de 2.000 a 10.000 temporadas condicionales a los marcadores ya fijados. Estas estimaciones usan un **modelo experimental de goles regularizado**; no son un modelo ML calibrado ni probabilidades de casas de apuestas.
7. Importa otras ligas en formato JSON de MatchLab o el histórico estándar `openfootball/football.json`. Para un histórico puedes escoger la jornada de corte; para un partido pendiente en la jornada 38 no hace falta marcador.
8. Descarga la plantilla JSON, exporta la liga o los resultados de tu escenario.

## Límites que aparecen explícitos en la UI

- No está conectada a resultados en vivo. Colombia usa la instantánea de resultados al 20/10/2026.
- Los extremos de Colombia usan **soluciones matemáticas verificadas** bajo victorias con márgenes fijos, empates 0-0 y desempates simplificados; no cubren la totalidad de todos los marcadores exactos reglamentarios.
- En ligas importadas/históricas la función mejor/peor muestra **rutas testigo encontradas por muestreo**, **no** óptimos matemáticos certificados. Los paneles de extremos dicen «sin certificación exacta».
- Las copas de formato suizo, playoffs, cruces de eliminación directa, reglas de descenso por promedio y campeonatos con estructura múltiple requieren plugins de reglamento específicos; no afirmamos soportarlos como si fueran ligas sencillas.
- No existen cuotas de apuestas ni modelos por liga calibrados en este prototipo. Se respeta la distinción entre «posible matemáticamente», «ruta testigo» y «probabilidad estimada».
- Todo el análisis ocurre localmente en el navegador; importar un archivo JSON no lo envía a ningún servidor. La selección de ligas históricas sí descarga datos JSON públicos de GitHub.

## Formato de importación

```json
{
  "teams": [
    {"id":"A","name":"Equipo A","pj":1,"w":1,"d":0,"l":0,"gf":2,"ga":1,"pts":3},
    {"id":"B","name":"Equipo B","pj":1,"w":0,"d":0,"l":1,"gf":1,"ga":2,"pts":0},
    {"id":"C","name":"Equipo C","pj":0,"w":0,"d":0,"l":0,"gf":0,"ga":0,"pts":0}
  ],
  "fixtures": [
    {"home":"A","away":"C","date":"2026-11-01","round":2},
    {"home":"B","away":"C","date":"2026-11-02","round":2}
  ],
  "meta": {"name":"Mi liga","topN":2,"season":"2026","source":"usuario","status":"custom"}
}
```

Validaciones: IDs únicos, estadísticas de puntos consistentes (3V+E), emparejamientos únicos, clubes existentes y cantidad válida de clasificados. Para torneos nacionales, la configuración `topN` indica el **objetivo principal del simulador**, no necesariamente todos los formatos de clasificación a torneos continentales.

## Cómo probar

```bash
python build.py
python test_ui.py
python test_historical.py
```

`test_ui.py` utiliza un Chromium disponible en la máquina de pruebas (`/usr/bin/chromium`) y comprueba el diseño en escritorio y móvil. Para servir el HTML en un hosting estático, publica `index.html`. No guarda tokens ni requiere API keys.

## Siguiente etapa para producción real

1. Motor de reglas por competición/temporada: puntos, desempates exactos, playoffs, descensos, cuotas, copa y fases suizas.
2. Backend de optimización MILP/CP-SAT con límites, colas, caché, escenarios condicionados y certificados de factibilidad.
3. Proveedor de datos bajo licencia para fixtures, resultados en vivo, IDs estables de equipos y permiso de redistribución.
4. Entrenamiento y backtesting con datos históricos autorizados, calibración temporal de 1X2 y evaluación de probabilidades, con controles de calidad.
5. Autenticación, escenarios guardados, monitoreo, CDN, alertas de latencia, automatización de actualizaciones y seguridad; solo entonces «en producción».

**Fuentes históricas abiertas:** https://github.com/openfootball/football.json (temporada 2024-25) y la documentación de MatchLab de octubre de 2026. Verificar siempre la licencia y la cobertura específica antes de comercializar.