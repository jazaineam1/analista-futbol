# MatchLab 4.0 · reconstrucción auditada (5 de noviembre de 2026)

**Producto web de escenarios 1X2 con selección libre de clubes, simulación probabilística conjunta, cotas matemáticas de posiciones y una API de optimización.**

## Qué cambió respecto de MatchLab 3.x

- Confirmados como terminados Cúcuta 1–0 Santa Fe y Millonarios 0–0 América (4 de noviembre de 2026); **ya no aparecen como partidos por jugar ni provisionales**.
- Los 20 clubes están con 17 PJ y quedan 20 partidos únicos pendientes, dos por equipo. Universo exacto `3^20 = 3.486.784.401` resultados 1X2.
- La simulación libre no hereda nunca el calendario del editor. El condicionamiento es una casilla **apagada por defecto**.
- Cada repetición sin semilla indicada obtiene semilla aleatoria distinta; se informa la semilla real para reproducción.
- Se modelan marcadores con Poisson condicional al desenlace; fijar una victoria 1X2 sin goles precisos no necesariamente fija al 100% un puesto en los desempates.
- Cada equipo tiene cotas (mejor y peor posición) obtenidas con **optimización MILP de puntos compartidos**. Son límites con empates favorables/adversos, no desempates oficiales exhaustivos.
- Permite importar cualquier temporada compatible mediante JSON. Incluye una liga sintética de cuatro equipos para comprobaciones.
- Las probabilidades iniciales vienen de un modelo Poisson suavizado, basado en GF/GC actuales, localía aproximada y dependencia simplificada. **No son un modelo ML prospectivamente calibrado**. Las probabilidades son editables encuentro por encuentro.

## Abrir sin instalación

Abre `MatchLab_4_2026-11-05.html` en un navegador. No usa recursos JavaScript externos y no necesita API ni credenciales. El archivo se puede enviar a un teléfono o servir como estático.

En la pantalla se puede:

1. Seleccionar cualquiera de los 20 clubes.
2. Revisar tabla, puntos máximos, explicación de eliminación o clasificación, cotas de puestos y fuentes.
3. Ejecutar 2.000, 10.000, 30.000 o 50.000 temporadas conjuntas de los 20 clubes, libres o condicionadas según la casilla.
4. Fijar un marcador exacto o desenlace 1/X/2 para cualquiera de los partidos.
5. Cargar rutas por identificador base 3, desde 0 hasta `3^20-1`, sin almacenar las 3.486.784.401 rutas.
6. Ver la distribución **exacta de puntos de un equipo** calculada por programación dinámica, condicionada por sus partidos si se solicita.
7. Importar ligas en formato JSON y exportar un escenario.

## Servicio de optimización (opcional)

Requiere Python 3.12+ (probado localmente en Python 3.13). Instalar:

```bash
python -m pip install -r requirements.txt
uvicorn server:app --host 127.0.0.1 --port 8765
```

Acceder a `http://127.0.0.1:8765/`. El servidor entrega el estado centralizado mediante `/api/state`. La pantalla añade el botón **Certificar escenarios fijados con servidor**, que llama a:

`POST /api/bounds`

```json
{"team_id":"DIM","fixed":{"18-FOR-DIM":1}}
```

Responde con extremos matemáticos de posición para la combinación impuesta y dos calendarios completos compatibles. `GET /api/health` informa de la salud del estado.

### Actualización administrativa de marcadores

Para habilitar resultados administrados, primero crea un archivo persistente distinto del original y un secreto de 24+ caracteres:

```bash
cp data/colombia-2026-ii.json /ruta/persistente/temporada.json
export MATCHLAB_DATA_PATH=/ruta/persistente/temporada.json
export MATCHLAB_ADMIN_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
uvicorn server:app --host 127.0.0.1 --port 8765
```

`POST /api/final` exige cabecera `X-Matchlab-Admin` y cuerpo `{fixture_id, home_goals, away_goals, source_url, verified_at}`. No acepta el mismo fixture finalizado dos veces y persiste la instantánea con reemplazo atómico. **Después de cualquier actualización, se invalidan todos los certificados previos**; se recalculan mediante el optimizador al consultar. No está conectado a resultados en vivo.

## Reglas matemáticas

El solver trata cada partido como una sola variable categórica: victoria local, empate, victoria visitante, repartiéndose 3/0 o 1/1 puntos de manera compatible. Calcula:

- Cota inferior de puesto = `1 + mínimo número de clubes estrictamente por delante en puntos`, permitiendo al equipo ganar todos los desempates por puntos.
- Cota superior = `1 + máximo número de clubes con puntos mayores o iguales`, permitiendo al equipo perderlos.

Estas cotas son **matemáticamente válidas respecto del puntaje**, pero no sustituyen los desempates de la Dimayor por diferencia de goles, goles a favor, goles visitantes, etc. Los testigos 1X2 incluidos son compatibles con todas las restricciones de enfrentamientos y se presentan como rutas por puntos. El simulador visual ordena por puntos, DG y GF como previsualización, no por el algoritmo oficial completo.

El contador de escenarios `3^N` y el de rutas compatibles con selecciones son exactos. El **conteo de posiciones para todo ese universo no ha sido realizado**; las distribuciones de posiciones corresponden a muestras Monte Carlo. La distribución marginal de puntos sí es exacta bajo independencia de los desenlaces 1X2.

## Pruebas ejecutadas

```bash
python tests/test_backend.py
python tests/test_browser.py
```

Se verificaron 7 grupos de pruebas de API/MILP, además de escenarios de interfaz en 1400x900 y 390x844, ausencia de desbordamiento horizontal, navegación base 3, pronóstico libre multirrango de Medellín, escenario condicionado y marcador exacto completamente determinado. La liga sintética de seis partidos se contrastó exhaustivamente frente a sus 729 combinaciones para los límites de posición del solver.

## Fuentes de la instantánea

- [Dimayor](https://www.dimayor.com.co/): resultados definitivos del 4 de noviembre.
- [El Espectador](https://www.elespectador.com/deportes/futbol-colombiano/asi-quedo-la-tabla-de-posiciones-de-la-liga-betplay-tras-la-fecha-17/): tabla de posiciones completa después de la jornada 17.
- [AS/Caracol, calendario jornada 18](https://caracol.com.co/resultados/futbol/colombia_ii/2026/jornada/regular_a_18/).
- [AS, jornada 19](https://colombia.as.com/resultados/futbol/colombia_ii/2026/jornada/regular_a_19/). La fecha y hora de jornada 19 siguen marcadas **por confirmar** por posibles reprogramaciones.
- [Reglamento Dimayor 2026](https://dimayor.com.co/wp-content/uploads/2026/01/REGLAMENTO-LIGA-2026-V12.pdf), artículos 19 y 20 para fase todos contra todos.

## Qué falta para producción comercial

- Fuente de datos licenciada con IDs estables, historial y reconciliación automática, caché, alertas de discrepancia y actualización en vivo.
- Archivo histórico detallado de goles locales/visitantes para reproducir totalmente los desempates; verificación adicional para límite del octavo puesto.
- Modelo predictivo entrenado con datos históricos suficientes, validación temporal, calibración por temporada y monitorización de precisión log-loss/Brier.
- Contador ponderado exacto de posiciones, con control de complejidad y retorno explícito a intervalos probabilísticos cuando no sea computacionalmente viable.
- Multiliga real con reglamentos por país/competición, UEFA, Conmebol, Arabia, formatos de grupos y eliminatorias.
- Plataforma pública TLS con servidor y base de datos replicada, autenticación, backups, tests end-to-end en CI, logs, límites de solicitudes distribuidos, permisos comerciales y gestión de costes.

**Estado:** MVP técnico verificable. La versión estática abre sin claves. El servicio API es una implementación de desarrollo/instancia única, todavía no una infraestructura comercial escalable ni una fuente de partidos en tiempo real.