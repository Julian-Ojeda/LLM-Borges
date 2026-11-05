#!/usr/bin/env python3
"""
Prueba a mano del transformador para completar frases: se escribe el comienzo de una frase y el modelo la
continua, token por token. Corre en CPU (no compite con un entrenamiento en curso).

Si el comienzo tiene menos de N tokens, se completa a la izquierda con puntos, como si viniera despues
del final de una oracion anterior. Las palabras que no estan en el vocabulario se toman como <UNK>.

Estrategias para elegir el token siguiente (Transformadores c, pag. 46-50):
    ansiosa   siempre el mas probable (texto previsible, suele repetirse)
    top_k     al azar entre los k mas probables
    top_p     al azar entre los mas probables hasta juntar una masa p de probabilidad (por defecto)

Uso:
    python scripts/generar_texto.py "volvió a su casa y"
    python scripts/generar_texto.py "la mujer abrió la puerta" "era una noche oscura" --palabras 40
    python scripts/generar_texto.py "volvió a su casa y" --estrategia top_k --k 20
    python scripts/generar_texto.py "volvió a su casa y" --estrategia ansiosa
    python scripts/generar_texto.py "había una vez" --modelo respaldos/transformador/modelo_completar_epoca_10.npz

Por defecto usa la época guardada con menor perdida de validacion (segun respaldos/transformador/metricas.csv).
"""

import argparse
import re
import sys
from pathlib import Path
import numpy as np
import torch

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))

from codigo.modelo_completar_torch import cargar_transformador, generar, unir_tokens

DIRECTORIO = ruta_raiz / "respaldos" / "transformador"


def mejor_respaldo(directorio: Path = DIRECTORIO) -> Path:
    """La época con menor perdida de validacion que tenga su .npz guardado."""
    tabla = np.loadtxt(directorio / "metricas.csv", delimiter=",", skiprows=1, ndmin=2)
    for epoca in tabla[np.argsort(tabla[:, 2]), 0].astype(int):      # columna 2 = perdida_validacion
        ruta = directorio / f"modelo_completar_epoca_{epoca}.npz"
        if epoca > 0 and ruta.exists():
            return ruta
    raise FileNotFoundError(f"No hay respaldos en '{directorio}'")


def main() -> None:
    parser = argparse.ArgumentParser(description="Completa frases con el transformador.")
    parser.add_argument("comienzo", nargs="*", help='Uno o más comienzos entre comillas, por ejemplo "volvió a su casa y".')
    parser.add_argument("--modelo", default=None, help="Respaldo .npz (por defecto, la época con mejor validación).")
    parser.add_argument("--palabras", type=int, default=30, help="Cantidad de tokens a generar.")
    parser.add_argument("--estrategia", choices=["ansiosa", "top_k", "top_p"], default="top_p")
    parser.add_argument("--k", type=int, default=40, help="Para top_k: cuántas candidatas.")
    parser.add_argument("--p", type=float, default=0.9, help="Para top_p: masa de probabilidad a juntar.")
    parser.add_argument("--temperatura", type=float, default=0.8, help="< 1 más conservador, > 1 más arriesgado.")
    parser.add_argument("--muestras", type=int, default=3, help="Cuántas continuaciones distintas generar.")
    parser.add_argument("--semilla", type=int, default=None, help="Fija el sorteo (para repetir un resultado).")
    args = parser.parse_args()

    ruta = Path(args.modelo) if args.modelo else mejor_respaldo()
    modelo, vocabulario, _ = cargar_transformador(ruta)
    mapeo = {str(p): i for i, p in enumerate(vocabulario)}
    print(f"Modelo: {ruta}  (N = {modelo.N}, h = {modelo.h}, L = {modelo.L}) | estrategia {args.estrategia}")

    # Cada argumento entre comillas es un comienzo distinto
    comienzos = args.comienzo or ["volvió a su casa y", "la mujer abrió la puerta", "era una noche oscura"]
    generador = torch.Generator()
    if args.semilla is not None:
        generador.manual_seed(args.semilla)
    muestras = 1 if args.estrategia == "ansiosa" else args.muestras    # la ansiosa siempre da lo mismo

    for comienzo in comienzos:
        tokens = re.findall(r"\w+|[^\w\s]", comienzo.lower())         # mismo patron que codigo/tokenizador.py
        desconocidas = [t for t in tokens if t not in mapeo]
        if desconocidas:
            print(f"  (fuera del vocabulario, se usan como <UNK>: {', '.join(desconocidas)})")
        ids = [mapeo.get(t, 0) for t in tokens]
        ids = [mapeo["."]] * max(0, modelo.N - len(ids)) + ids          # relleno a la izquierda con puntos
        print(f"\n=== {comienzo} ===")
        for _ in range(muestras):
            nuevos = generar(modelo, ids, args.palabras, args.estrategia, args.temperatura, args.k, args.p,
                             prohibidos=[0], generador=generador)
            print(f"  {comienzo} → {unir_tokens([str(vocabulario[i]) for i in nuevos])}")


if __name__ == "__main__":
    main()
