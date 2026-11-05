# MatchLab 3.1 — Liga BetPlay II 2026 — actualización 4 de noviembre

## Alcance
- HTML autónomo para explorar los **21 partidos sin resultado definitivo** del corte; permite cambiar marcadores, seleccionar cualquiera de los 20 clubes, explorar `3^21 = 10.460.353.203` combinaciones 1X2 mediante índices, editar probabilidades por partido, simular la tabla y realizar muestreo Monte Carlo.
- La Liga BetPlay tiene 19 partidos de fase I por equipo. La instantánea registra 169 partidos terminados y 21 por resolver: todos los clubes satisfacen `PJ + pendientes = 19`.
- **Junior (16 pts, PJ17): matemáticamente eliminado del top 8**, porque su techo es 22 puntos y hay al menos 8 equipos que ya superaron 22.

## Corte y fuentes
Datos de puntos, PJ, W/E/L, GF y GC al término del 3 de noviembre, cruzando AS Colombia (tabla que muestra resultados al 04/11 19:46 COT) y la publicación de Claro Sports del 03/11; sustitución del partido Cúcuta 1–0 Santa Fe como final conforme al marcador publicado en la portada de Dimayor el 04/11. El resultado **en vivo** de Millonarios–América **no** fue considerado final y permanece entre las 21 variables futuras/pendientes. La tabla externa de AS aún mostraba un empate provisional en Cúcuta–Santa Fe, así que se documentó la discrepancia y se priorizó Dimayor.

- Resultado final oficial de Cúcuta–Santa Fe (04/11): https://dimayor.com.co/
- Tabla histórica de referencia y estadísticas por club: https://colombia.as.com/resultados/futbol/colombia_ii/clasificacion/
- Tabla jornada 17 y resultados recientes: https://www.clarosports.com/colombia/liga-bet-play/clasificacion-liga-betplay-dimayor-2026-ii/
- Fecha 18, actualizada: https://caracol.com.co/resultados/futbol/colombia_ii/2026/jornada/regular_a_18/
- Fecha 19 provisional: https://colombia.as.com/resultados/futbol/colombia_ii/2026/jornada/regular_a_19/
- Reglamento Dimayor: https://dimayor.com.co/wp-content/uploads/2026/01/REGLAMENTO-LIGA-2026-V12.pdf

La versión no tiene una licencia de proveedor para un feed de partidos ni cuotas, por lo que no debe afirmarse que se actualiza en tiempo real. Los horarios de fecha 19 pueden cambiar. Tras finalizar Millonarios–América, es necesario importar una instantánea posterior; la app permite importar JSON sin actualizar el código.

## Tres clases de resultados que no deben confundirse
1. **Exacto:** número total de calendarios `3^N`, IDs ternarios, conteo de calendarios compatibles con restricciones y distribución por puntos de un club (condicionada al modelo 1X2).
2. **Certificación suficiente:** eliminación por puntos, cuando ya hay ocho o más equipos cuyo puntaje actual supera el máximo del objetivo.
3. **Experimental:** Poisson 1X2, simulaciones Monte Carlo y posiciones calculadas con marcadores generados. No hay modelo calibrado en partidos prospectivos de noviembre de 2026.

Las rutas denominadas «Mejor camino» y «Peor camino» son muestras de búsqueda cuando no existe certificado óptimo, no mínimos o máximos globales demostrados. No se enumeran 10 mil millones de escenarios ni se han contado exhaustivamente todos los puestos finales. Desempates oficiales adicionales siguen pendientes de implementar.

## Validaciones
Ejecutar `python qa_matchlab_nov4.py` si tienes `playwright` y Chromium instalado. La suite prueba versión escritorio y móvil, 20 clubes, 21 partidos, universo ternario, búsqueda del último identificador, reducción exacta a `3^20` al fijar un juego, reestablecimiento, eliminación matemática para Junior y ausencia de errores JS. La prueba de datos comprueba conteos y consistencia goles a favor/contra.

## Para producción
1. Configurar proveedor de datos autorizado, versiones de temporada, claves servidor y reconciliación automática con el estado oficial.
2. Ejecutar instantáneas inmutables; no mezclar marcadores parciales y finalizados.
3. Implementar y validar motor completo de desempates Dimayor y optimización CP-SAT con pruebas de maximalidad.
4. Montar backend HTTPS con caché de resultados, cuotas/licencias y logs de auditoría; nunca poner claves privadas en el HTML.
5. Usar históricos con derechos de reutilización para calibrar ML y comparar log-loss, Brier y cobertura antes de anunciar probabilidades confiables.