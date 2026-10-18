"""Descargas con caché local (datos abiertos de StatsBomb y SkillCorner).

La caché vive en ~/.cache/analista (o en la carpeta de la variable ANALISTA_CACHE).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

USER_AGENT = "analista-futbol (https://github.com/jazaineam1/analista-futbol)"


def carpeta_cache() -> Path:
    base = os.environ.get("ANALISTA_CACHE")
    ruta = Path(base) if base else Path.home() / ".cache" / "analista"
    ruta.mkdir(parents=True, exist_ok=True)
    return ruta


def descargar(url: str, destino: Path, progreso: bool = False, timeout: int = 120) -> Path:
    """Descarga `url` a `destino` si no existe ya (escritura atómica)."""
    destino = Path(destino)
    if destino.exists() and destino.stat().st_size > 0:
        return destino
    destino.parent.mkdir(parents=True, exist_ok=True)
    tmp = destino.with_suffix(destino.suffix + ".parcial")
    with requests.get(url, stream=True, timeout=timeout, headers={"User-Agent": USER_AGENT}) as r:
        if r.status_code == 404:
            raise FileNotFoundError(f"No existe: {url}")
        r.raise_for_status()
        total = int(r.headers.get("Content-Length") or 0)
        hecho = 0
        with open(tmp, "wb") as f:
            for trozo in r.iter_content(1 << 20):
                f.write(trozo)
                hecho += len(trozo)
                if progreso and total:
                    print(f"\r  descargando {destino.name}: {100 * hecho / total:5.1f}%", end="", file=sys.stderr, flush=True)
        if progreso and total:
            print(file=sys.stderr)
    tmp.replace(destino)
    return destino


def json_remoto(url: str, destino: Path) -> object:
    return json.loads(descargar(url, destino).read_text(encoding="utf-8"))
