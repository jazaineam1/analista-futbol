# MatchLab 3.0 — Auditoría de escenarios y modelo probabilístico

**Fecha de corte analizada:** 20 de octubre de 2026, Liga BetPlay II. **50 partidos restantes en el archivo**. Son datos congelados, no un feed en vivo ni una afirmación de que el calendario continúe igual en una fecha futura.

## Hallazgos de la auditoría

1. La versión multiliga 2.0 muestra `3^N` como texto y puede simular temporadas, pero **no tenía un índice global de las combinaciones ni conteo analítico de los puntos de un equipo**. Incorporado en MatchLab 3.0.
2. La versión anterior usaba únicamente marcadores canónicos 1–0, 0–0 y 0–1 también en Monte Carlo. **Esto sesga las diferencias de gol y puede cambiar posiciones empatadas en puntos**. Corregido parcialmente: el nuevo Monte Carlo elige resultados 1X2 con las probabilidades configuradas y después genera un marcador de una distribución Poisson condicionada al resultado.
3. El ordenamiento todavía simplifica desempates (puntos, DG y GF). Por tanto **no se deben presentar las posiciones empatadas como certificaciones oficiales** sin cargar goles visitantes y todas las reglas pertinentes a la liga/temporada.
4. El modelo probabilístico predeterminado es **Poisson simplificado calculado de goles observados con regularización rudimentaria**; no es un modelo ML validado prospectivamente para 2026 ni una cotización de apuestas. El usuario puede ajustar las probabilidades de cada partido e indicar su fuente.
5. **No hay sincronización automática real, credenciales oficiales de una API de fútbol ni precio de mercado actualizado**. Para producción es indispensable proveedor licenciado, servicio de ingestión, reconciliación de resultados y monitoreo.
6. La versión estática admite ligas importadas, pero **las copas con fases de eliminación no comparten el mismo motor de clasificación**.

## Espacio exacto de escenarios

Si quedan `N` encuentros y cada uno se codifica con `x_i∈{0,1,2}` (local/empate/visitante), hay exactamente `3^N` vectores. Para 50 partidos:

`3^50 = 717897987691852588770249`.

Codificación ternaria: `ID(x)=Σ_{i=0}^{N-1} x_i*3^(N-1-i)`. El ID es un entero exacto (`BigInt`) entre cero y `3^N-1`. Permite recuperar **cualquier** secuencia de resultados sin almacenarla; se pueden paginar bloques pequeños. Este acceso por ID **no significa** que hayamos enumerado/procesado todos los escenarios.

Si el usuario fija `k` resultados independientes como condiciones, hay `3^(N-k)` calendarios compatibles. Bajo independencia condicional de partidos, la masa de probabilidad de la condición es `Π p_i(x_i)` para los partidos fijados. **No se debe interpretar como probabilidad de clasificación**.

**Importante:** `3^50` solo enumera las secuencias **1/X/2**, no todos los marcadores posibles. Para DG, GF, GF visitante, disciplinarios y desempates deben integrarse resultados con goles y reglas explícitas; el marcador canónico asociado a cada ID se usa para visualización, no para certificar posiciones oficiales.

## Modelo de puntos exactos de cada equipo

Para cualquier equipo, creamos un polinomio de probabilidad de sus puntos finales. Cada partido futuro del equipo aporta:

- Si juega de local: `p_H z^3 + p_D z^1 + p_A`.
- Si juega de visitante: `p_A z^3 + p_D z^1 + p_H`.

Multiplicamos estos factores mediante **programación dinámica**, no expandiendo rutas una a una. El coeficiente de `z^j` es la probabilidad condicional de sumar `j` puntos futuros **bajo el modelo independiente**. Paralelamente mantenemos coeficientes enteros que cuentan cuántos resultados 1X2 generan cada puntuación. Los partidos de otros clubes multiplican el conteo por `3` y suman masa probabilística total 1. Los partidos fijados sustituyen su factor por el resultado impuesto. **Este conteo por puntos es exhaustivo y matemáticamente exacto**, pero no lo es para la posición final, porque esta depende de todos los clubes simultáneamente.

## Modelo probabilístico: diseño definitivo

**A. Probabilidades por partido.** Versión de producción propuesta:

1. Datos: 3–5 temporadas de marcadores + localía, descanso, fuerza reciente (Elo/Glicko), xG cuando exista, bajas confirmadas, alineaciones y calendario. Cada predictor debe tener fecha de disponibilidad anterior al partido (sin fuga de información).
2. Baselines: frecuencias regularizadas, Elo 1X2 y Poisson ataque-defensa con correcciones tipo Dixon-Coles y decaimiento temporal. Modelo bivariante para correlación de goles cuando se justifique.
3. Candidatos ML: regresión logística multinomial, CatBoost/LightGBM, modelos jerárquicos Bayes y clasificadores calibrados. Entrenar con validación cronológica tipo walk-forward y separar por liga o aplicar efectos jerárquicos.
4. Mercado de apuestas: si hay permisos de uso, ingestión de cuotas 1X2 por casa y hora, eliminación del margen `q_i=(1/cuota_i)/Σ(1/cuota_j)` como aproximación básica, y mezcla calibrada con la predicción modelada; no mezclar cuotas de fechas o casas heterogéneas como si fueran una fotografía comparable.
5. Calibración: isotonic/temperature/Dirichlet según datos suficientes; comparar log-loss, Brier, calibración por deciles, confiabilidad y disponibilidad de mercado. Elegir pesos exclusivamente en conjuntos de validación pasados.
6. Producción: versionar el modelo, timestamp y datos usados, y bloquear predicciones cuando falten o se contradigan fuentes.

**B. Distribución de resultados y posiciones.** Cada encuentro tiene `(p_H,p_D,p_A)` y para simular diferencia de gol se generan marcadores `P(g_h,g_a | H/D/A)`. La probabilidad de un **camino completo** bajo la independencia simplificada es el producto de sus 50 probabilidades. Una simulación de Monte Carlo es **solo una estimación de la distribución**, con incertidumbre muestral y de modelo. Para eventos muy raros usar importance sampling, muestreo estratificado, intervalo de confianza y validación de convergencia.

**C. Optimizador matemático exacto.** Modelar un resultado único por partido y su reparto consistente de puntos. CP-SAT/MILP encuentra rutas factibles, mejor/peor posición y certificados de imposibilidad; las reglas de desempate deben ser específicas de competición y temporada. Un solver que halla UNA solución no cuenta cuántas existen.

**D. Conteo exacto de clasificación:** es un problema mucho más complejo (conteo ponderado de soluciones sobre resultados acoplados). Investigar un grafo de decisiones ordenado y reducido (BDD/ADD), eliminación de variables / transfer matrix y poda de límites. Calcular exactamente **solo si** el motor puede demostrar que agotó estados dentro de los recursos disponibles; de lo contrario devolver límites certificados si los hay y estimaciones Monte Carlo expresamente etiquetadas. No prometer exhaustividad para 50 partidos a costo constante.

## Interfaz propuesta

- Selector liga, temporada, equipo, reglas reglamentarias.
- Universo: total 3^N, ID ternario global, ID relativo a restricciones, saltar al ID y paginación virtual.
- Tabla interactiva unida a cada escenario, con diferencial explícito entre resultado 1X2 y marcador supuesto.
- Distribución exacta de **puntos** del equipo, con conteos enteros y probabilidades condicionadas.
- Posiciones: mínimo/máximo probado, probabilidad estimada por puesto, intervalos Monte Carlo, estado `PROBADO / ESTIMADO / DESCONOCIDO`.
- Editor de probabilidades por partido con fuente/fecha y revisión de `Σp_i=1`.
- Modo «¿Qué debe pasar para clasificar?» con resultados fijados, cálculo MILP y explicación de restricciones.
- Todos los escenarios accesibles por ID, nunca como un CSV de 10^24 filas.

## Plan de entrega a producción (8–12 semanas de desarrollo, sujeto a datos/licencia)

| Fase | Alcance | Criterio de aceptación |
|---|---|---|
| 1. Auditoría y reglas | Fuente oficial, fixtures por temporada, tabla, desempates, IDs y auditoría de 50 partidos | Diferencias contra fuente externa investigadas, cero partidos duplicados |
| 2. Probabilidades | Elo/Poisson + mercado permitido + validación temporal, calibración por liga | Log-loss y Brier superan baseline en holdout cronológico |
| 3. Solver | CP-SAT/MILP para cualquier equipo, objetivo y resultados fijados | Mejor/peor posición demostrable y ruta reproducible, timeouts etiquetados |
| 4. Escenario completo | ID ternario, DP puntos, Monte Carlo condicional, importancia para eventos raros | Concordancia exacta con brute-force para ligas sintéticas pequeñas |
| 5. DataOps | Ingesta desde API licenciada, backfills, webhooks/poll, reconciliación, caching | Retraso medido y alarma de fuente obsoleta, cero sobrescrituras silenciosas |
| 6. Producto y seguridad | React/Next, FastAPI, Postgres, Redis/colas, cuentas, rate-limit, backups, observabilidad | CI verde, pruebas móviles, carga, seguridad y despliegue HTTPS verificados |
| 7. Beta | Colombia + una liga europea con reglas completas | Publicación con transparencia de datos, predicciones y errores; monitoreo 2 semanas |

No instalar el backend de optimización en GitHub Pages: alojarlo detrás de un API segura con rate limiting. El HTML de esta entrega funciona offline como prototipo y conserva los 50 encuentros del corte. El universo ternario y las distribuciones exactas de puntos se calculan enteramente en el navegador; la probabilidad de posición sigue siendo simulada.

## Pruebas realizadas

- Colombia: `3^50` verificado con `BigInt`; acceder al último identificador, cargar ruta, devolver condiciones, fijar 2 partidos (`3^48`), modificar/restaurar probabilidades y cambiar de equipo.
- Microliga de 4 equipos y 6 partidos: comprobadas 729/729 rutas frente a enumeración completa, identidad de ID inversa, masa probabilística por puntos y conteos de distribución; fijados dos partidos, comprobadas 81/81 rutas condicionales.
- Monte Carlo: 500 muestras y marcadores condicionados al resultado 1X2; móvil con 390 px, sin desbordamiento horizontal observado.

**Limitaciones:** sin cuotas actuales verificadas por API, sin entrenamiento ML validado 2026 en esta versión, sin cálculos exhaustivos de posiciones para el campeonato de 50 partidos y sin servicio HTTPS publicado.