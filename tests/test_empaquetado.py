"""Todo archivo que no es Python dentro del paquete tiene que viajar en la instalación (package-data)."""
from __future__ import annotations

import ast
import fnmatch
import re
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]


def test_archivos_del_paquete_incluidos_en_la_instalacion():
    texto = (RAIZ / "pyproject.toml").read_text(encoding="utf-8")   # sin tomllib: también corre en 3.10
    bloque = texto.split("[tool.setuptools.package-data]", 1)[1]
    patrones = ast.literal_eval(re.search(r"^analista\s*=\s*(\[.*?\])", bloque, re.M | re.S).group(1))
    paquete = RAIZ / "analista"
    faltan = []
    for f in paquete.rglob("*"):
        if not f.is_file() or f.suffix in (".py", ".pyc") or "__pycache__" in f.parts:
            continue
        rel = f.relative_to(paquete).as_posix()
        if not any(fnmatch.fnmatch(rel, p) for p in patrones):
            faltan.append(rel)
    assert not faltan, f"archivos que no se instalarían: {faltan}"
