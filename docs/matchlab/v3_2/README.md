# MatchLab 3.2 — América–Millonarios, actualización controlada (04/11/2026)

## Estado de la fuente

Al consultar a las 22:32 (hora Bogotá) el 04/11/2026, la página de Dimayor y marcadores secundarios mostraban **Millonarios 0–0 América de Cali**, pero **no acreditaban todavía `FINALIZADO`**. El 0–0 es una **hipótesis explícita y reversible**, NO un resultado final verificado. La pantalla presenta una advertencia y permite regresar al corte comprobado anterior a ese encuentro.

Fuentes usadas:
- https://dimayor.com.co/ (marcador 0–0 y estado «En Vivo»)
- https://www.365scores.com/football/league/liga-aguila-620 (marcador 0–0 en vivo)

## Cómo usar

Abre `MatchLab_3_2_America_Provisional_2026-11-04.html` directamente en un navegador moderno, sin instalar dependencias. La vista inicial aplica **provisionalmente** el empate 0–0 y muestra 20 partidos por resolver. Los botones **Simular empate 0–0** y **Volver a corte verificado** alternan entre ambas instantáneas, sin confundirlas con un marcador final oficial.

### Datos condicionales

- **Escenario provisional:** América 35 puntos, 17 PJ; Millonarios 27 puntos, 17 PJ; 20 partidos restantes; `3^20 = 3.486.784.401` desenlaces.
- **Corte verificado anterior:** América 34 puntos, 16 PJ; Millonarios 26 puntos, 16 PJ; 21 partidos restantes; `3^21 = 10.460.353.203` desenlaces.
- En ambos cortes todos los equipos terminan con 19 partidos al completar su calendario y las sumas de goles, juegos y puntos concuerdan.

No se ha alterado el resultado oficial en la versión anterior, ni se ha afirmado que el partido terminó 0–0. Al publicarse el acta/final oficial, reemplazar la instantánea con el marcador confirmado y retirar esta vista condicional.

## Pruebas de navegador

`python qa_matchlab_america.py` requiere Playwright y Chromium. Valida escritorio (1360 px) y móvil (390 px): selección de 20 clubes, paso reversible de 21 a 20 encuentros, contadores `3^21` y `3^20`, puntos de América y Millonarios, navegación al último ID, filtros de combinaciones, ausencia de excepciones JS y ausencia de desbordamiento horizontal.

## Límite predictivo

Los recuentos de combinaciones y de puntos son exactos para el calendario 1X2 representado. El modelo de probabilidades sigue siendo experimental: no incorpora cuotas actuales ni garantiza posiciones exactas cuando hay desempates reglamentarios complejos. Esta versión es un explorador local, no un sistema con actualización automática de resultados en vivo.