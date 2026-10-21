# MatchLab · Entrega de producción controlada (21/oct/2026)

## Alcance y estado

**20 clubes · 50 partidos restantes · corte 20/oct/2026 (hora Colombia).** Cuatro encuentros del 20/oct/2026 incorporados con marcador completo y evidencia en `verified_results_2026_10_20.json`:

- Deportivo Pereira 2–0 Boyacá Chicó
- Internacional de Bogotá 2–1 Fortaleza
- Cúcuta Deportivo 1–0 Atlético Nacional
- Millonarios 1–1 Once Caldas

**Fuente primaria:** Dimayor, marcador final; contraste AS Colombia / Sofascore para Millonarios–Once Caldas.

**NO ES UN FEED EN VIVO.** El corte es verificable, pero no se actualiza por sí solo. Los resultados posteriores al corte deben llegar de un proveedor autorizado o una operación manual. No se ofrece una cuota de mercado inventada. El histórico 2023–2025 está disponible en el módulo estadístico; el modelo no tiene calibración prospectiva demostrada para 2026.

## Funciona de dos formas

1. **Demo estática**: abre `MatchLab_FINAL.html` o `index.html`. Permite escoger cualquier club, fijar resultados 1/X/2, recorrer 50 encuentros, revisar probabilidades experimentales, y mostrar mejores/peores caminos precomputados para los 20 clubes.
2. **Servicio con optimizador**: con Python 3.12, ejecuta:

   ```bash
   python -m pip install -r requirements.txt
   export MATCHLAB_ADMIN_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')"
   export MATCHLAB_RESULTS_PATH="$(pwd)/resultados_confirmados.json"
   python matchlab_server_final.py
   ```

   Navega a `http://127.0.0.1:8765`.

### Docker

```bash
docker build -t matchlab:1.1 .
docker volume create matchlab-data
docker run -d --name matchlab -p 127.0.0.1:8765:8765 \
  -v matchlab-data:/app/data \
  -e MATCHLAB_ADMIN_TOKEN="$(python -c 'import secrets;print(secrets.token_urlsafe(32))')" \
  matchlab:1.1
```

Para dominio público, coloca el servicio detrás de HTTPS con Caddy, Nginx o el proxy administrado del host. Configura un volumen persistente y límites de tráfico en el proxy. El servidor HTTP incorporado es de instancia única y no es una arquitectura de escala horizontal.

### Seguridad y endpoints

- `GET /` y `GET /api/state`: UI y instantánea.
- `GET /api/health`: salud + integridad de fixtures.
- `POST /api/optimize` y `POST /api/options`: programación lineal entera. Límite por IP y concurrencia configurable.
- `POST /api/results`: **solo administrador** usando `X-Matchlab-Admin`; secreto de al menos 24 caracteres, comparación constante. Resultados verificados, persistencia atómica, no sobreescritura silenciosa.
- API misma procedencia (sin CORS abierto), límite de tamaño, cabeceras de seguridad. Si hay proxy, reforzar WAF / TLS, observabilidad y rate limiting distribuido.

### Publicación estática

Un servidor como GitHub Pages puede alojar `index.html` sin claves, pero **no ejecuta el solver Python ni ingiere finales**. Es un modo demostrativo con corte fijo. No publiques secretos o tokens en el HTML. Para el producto integral, despliega además el backend y un proveedor de datos licenciado.

## Certificados y ML

`certificados_20_equipos.json` contiene 60 soluciones MILP (mejor con victorias 1–0, mejor con 3–0 propios, peor con victorias 1–0). **No son una enumeración de todos los resultados exactos posibles**. Los criterios de desempate son simplificados; los desempates restantes de Dimayor requieren datos históricos completos y reglamento temporada.

La probabilidad de clasificación se aproxima con Monte Carlo de 3.000, 10.000 o 30.000 temporadas y depende del modelo elegido. No interpretar porcentajes minúsculos como imposibilidad. El entrenamiento histórico usa 1.051 marcadores de Colombia 2023–2025 con validación temporal; `MatchLab_ML_historico_validacion.json` aporta las métricas y limitaciones.

## Probar antes de publicar

```bash
python test_production.py
python qa_production.py
```

`qa_production.py` valida escritorio y pantalla móvil con Chromium por inspección HTML local (el navegador de pruebas restringe acceso HTTP localhost). `test_production.py` valida datos, restricciones, ruta de Junior, seguridad HTTP y los 60 certificados. Para una prueba de humo manual del servidor, usar `curl` contra los endpoints. Antes de afirmar disponibilidad pública, verificar el URL externo y la salud de la API desde fuera del servidor.

## Faltantes para producción comercial completa

- Proveedor oficial/licenciado de fixtures, resultados y cuotas con credenciales guardadas en servidor.
- Configuración y revisión de identificadores de los 20 clubes, conciliación automatizada y control de anomalías.
- Reentrenamiento periódico, backtesting/monitorización, calibración 2026 y alertas de deriva.
- Motor completo de desempates oficiales, incluyendo goles visitantes si aplican en el reglamento.
- Despliegue público HTTPS + logs centralizados, copias de seguridad, API rate limiting distribuida, monitor de disponibilidad y política de licencia/uso de datos.
- Multiliga: esta entrega corresponde **únicamente a Liga BetPlay II 2026**; no inferir que cubre Champions/otras ligas.

## Fuentes

- Dimayor: https://dimayor.com.co/
- AS Colombia (Millonarios–Once Caldas): https://colombia.as.com/futbol/once-caldas-amarga-a-millonarios-empate-en-el-campin-f202610-n/
- OpenFootball CC0: https://github.com/openfootball/south-america/tree/master/colombia
- Proveedor posible: https://www.api-football.com/ (credenciales, cobertura y derechos por verificar)