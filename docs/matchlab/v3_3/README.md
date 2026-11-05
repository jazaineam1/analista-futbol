# MatchLab 3.3 — corrección de pronósticos y universo 3^N

Corte de datos de origen: **4 de noviembre de 2026**. La vista Colombia sigue conservando el estado verificado anterior al partido Millonarios–América y una vista **0–0 provisional**, que **NO** constituye un marcador final oficial confirmado. Esta actualización **no incorpora un proveedor de resultados en vivo**. El modelo Poisson es experimental y no está calibrado prospectivamente.

## Problema reproducido y resuelto

En la versión 3.2, los botones «mejor camino», «peor camino» y «pronóstico modal» completaban todos los partidos con marcadores y luego el motor de Monte Carlo simulaba *solo* los partidos no fijados. Con un calendario totalmente fijado, la distribución era un punto: Medellín aparecía 100 % en un puesto. Esa era la probabilidad condicional degenerada de una ruta cerrada, **no** una predicción del torneo. Además, la antigua semilla fija reproducía siempre la misma muestra.

En 3.3 se separan estos conceptos:

- **Tabla del escenario**: resultado determinista de los partidos elegidos; no es un pronóstico.
- **Pronóstico libre** (por defecto): sortea **todos** los encuentros pendientes según probabilidades 1X2, independientemente de las selecciones de la tabla. Si el usuario selecciona un equipo distinto, ve la misma muestra global de temporadas.
- **Pronóstico condicionado** (optativo): respeta expresamente los partidos fijados, incluso cuando otros encuentros se simulan. Si están todos fijados, el panel avisa «ESCENARIO CERRADO: 100 % NO es una predicción».
- **Semilla aleatoria o reproducible**: semilla vacía genera un nuevo muestreo en cada ejecución; semilla introducida conserva resultados idénticos con los mismos datos y configuración.
- **Simulación conjunta**: todas las clasificaciones de una misma temporada se obtienen con un marcador único por encuentro, y las probabilidades de los *top N* suman N en la muestra.
- **Marcadores desde la perspectiva del equipo**: por ejemplo, Medellín visitante con marcador local 1–0 se representa como «Derrota 0–1», no como una victoria aparente.
- **Probabilidades editables**: el usuario puede introducir 1X2 por partido; el motor vuelve a simular al cambiar los valores. En ausencia de cuotas licenciadas, usa Poisson simplificado.

## 3^38: universos y límites

El número de rutas 1X2 es `3^N`, **N = cantidad real de partidos pendientes únicos** de la competición importada. El HTML colombiano usa N=20 para la **vista provisional** o N=21 para la **vista verificada anterior al clásico**. En ambos casos los clubes comparten los mismos partidos y no se suman dos veces.

Para una liga con 38 encuentros sin resolver: `3^38 = 1.350.851.717.672.992.089` rutas. El archivo `MatchLab_demo_38_partidos_sinteticos.json` es una **prueba de capacidad**, con clubes y encuentros **inventados** y etiquetados como sintéticos; al importarlo, el explorador BigInt indexa cualquiera de las rutas de 0 a `3^38−1` y muestra el conteo exacto.

**No se han enumerado ni evaluado probabilísticamente una por una las 3^38 rutas**, lo que sería computacionalmente inviable para un navegador. Se realiza:

1. Representación exacta e indexación por enteros BigInt de todo el universo.
2. Conteo exacto de los resultados compatibles con las selecciones (`3^(N−k)`).
3. Distribución exacta por **puntos finales** de un equipo, usando programación dinámica (supuesto de independencia de desenlaces).
4. Para **posición** y **clasificación**, estimación por Monte Carlo conjunto de 2.000 a 50.000 temporadas con intervalo muestral; NO es un conteo exacto de todas las posiciones.

Las posiciones dependen de partidos entre rivales y de desempates de goles. La tabla sigue aplicando puntos, diferencia de gol y goles a favor; criterios adicionales oficiales requieren otra implementación y datos correspondientes. Las predicciones Poisson son experimentales, sin calibración prospectiva. Ni una frecuencia estimada de 0 % demuestra imposibilidad ni una de 100 % demuestra clasificación matemática.

## Uso

Abrir `MatchLab_3_3_Simulacion_Corregida.html` en un navegador moderno. Seleccionar liga/equipo, elegir «Mejor camino» u otro escenario para explorar la **tabla**, y debajo seleccionar «Pronóstico libre» para ver el pronóstico, o «Condicionado» para fijar explícitamente los marcadores. Elegir 5.000, 10.000, 25.000 o 50.000 temporadas, opcionalmente una semilla. El cuadro inferior muestra las posibilidades para **todos los clubes de la misma muestra de temporadas**.

Para probar 38 partidos, en la opción de importar liga cargar `MatchLab_demo_38_partidos_sinteticos.json`. Esta importación no afirma describir la Liga BetPlay.

## Pruebas realizadas

- Reproducción del error de Medellín: con el peor camino cargado, la modalidad libre distribuye probabilidad en varios puestos; la condicionada a 20 marcadores cerrados advierte que es determinista.
- 5.000 temporadas: cada corrida asigna un solo resultado a cada encuentro. Para 20 clubes y 8 clasificados, la suma de las frecuencias de top 8 fue exactamente `8 * 5.000 = 40.000` clasificaciones.
- Semilla fija `20261104`: dos ejecuciones de 10.000 temporadas generaron exactamente las mismas frecuencias por posición; con semilla `20261105` cambiaron.
- Cambiar de club conserva la misma muestra y semilla.
- Modificar una probabilidad 1X2 desencadena un nuevo cálculo.
- Importación de **38 encuentros sintéticos**: el contador mostró `1.350.851.717.672.992.089` rutas y el índice máximo se aceptó.
- Prueba en Chromium escritorio y móvil de 390 px, sin errores JavaScript ni desplazamiento horizontal en la prueba móvil.

Para ejecutar las pruebas automatizadas (requiere `pip install playwright` y Chromium): `python tests/test_matchlab.py` desde la carpeta descomprimida.

## Siguientes pasos para producción

Proveedor de datos con licencia y validación de estado `FINAL`, reconciliación de encuentros, credenciales en backend, reglas de desempate por competición, modelos ML entrenados/calibrados fuera de muestra, pruebas de calibración, control de cuota/latencia, motor de optimización para mejores/peores posiciones verificables y hosting HTTPS con monitoreo. No afirmar producción en vivo por abrir este HTML local.