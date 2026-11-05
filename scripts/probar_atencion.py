#!/usr/bin/env python3
"""
Prueba a mano del CBOW con atencion: se escribe una frase con un hueco (___) y se compara lo que predicen
el CBOW de partida (promedio del contexto) y el modelo con atencion, mas el peso que la atencion le dio a
cada palabra del contexto. Corre en CPU (no necesita GPU ni compite con un entrenamiento en curso).

El modelo usa exactamente m = 4 palabras a cada lado del hueco (las mas cercanas): la frase tiene que tener
al menos 4 tokens antes y 4 despues. Los signos de puntuacion cuentan como tokens.

Uso:
    python scripts/probar_atencion.py "la casa de mi ___ estaba cerca del rio"
    python scripts/probar_atencion.py "frase con ___" --modelo respaldos/atencion/modelo_cbow_w4_atencion_epoca_10.npz --k 5
    python scripts/probar_atencion.py                    # frases de ejemplo
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

from codigo.modelo_atencion_torch import CBOWConAtencion, PARAMETROS_ATENCION, cargar_respaldo

EJEMPLOS = [
    "el hombre abrió la ___ de la casa y",
    "y ella le dijo : ___ , no sé qué hacer",
    "volvió a su casa al día ___ , cuando salió el sol",
]


def ultimo_respaldo_atencion(directorio: Path = ruta_raiz / "respaldos" / "atencion") -> Path:
    """El .npz de atencion de la época mas alta."""
    candidatos = list(directorio.glob("modelo_cbow_w*_atencion_epoca_*.npz"))
    if not candidatos:
        raise FileNotFoundError(f"No hay respaldos de atencion en '{directorio}'")
    return max(candidatos, key=lambda ruta: int(re.search(r"epoca_(\d+)", ruta.name).group(1)))


def cargar_modelo_atencion(ruta: Path) -> tuple[CBOWConAtencion, dict]:
    """Reconstruye el modelo con atencion desde su .npz (en CPU)."""
    datos = np.load(ruta, allow_pickle=True)
    H = int(datos["cantidad_cabezas"])
    C = datos["atencion_P"].shape[0]
    modelo = CBOWConAtencion(datos["W"], datos["W_prima"], C, H)
    with torch.no_grad():
        for nombre in PARAMETROS_ATENCION:                       # se pisan los parametros recien inicializados
            getattr(modelo, nombre).copy_(torch.tensor(datos[f"atencion_{nombre}"]))
    modelo.eval()
    configuracion = eval(str(datos["configuracion_json"]), {"__builtins__": {}})
    return modelo, configuracion


def armar_contexto(frase: str, mapeo: dict, m: int) -> tuple[list[str], list[int]]:
    """
    Tokeniza la frase como el corpus y devuelve las m palabras a cada lado del hueco y sus indices.
    """
    tokens = re.findall(r"\w+|[^\w\s]", frase.lower())          # mismo patron que codigo/tokenizador.py
    huecos = [i for i, t in enumerate(tokens) if set(t) == {"_"}]
    if len(huecos) != 1:
        raise ValueError("La frase tiene que tener exactamente un hueco (___).")
    j = huecos[0]
    if j < m or len(tokens) - j - 1 < m:
        raise ValueError(f"Hacen falta al menos {m} tokens antes y {m} despues del hueco "
                         f"(hay {j} antes y {len(tokens) - j - 1} despues).")
    palabras = tokens[j - m:j] + tokens[j + 1:j + 1 + m]          # orden p-m..p-1, p+1..p+m (como en el entrenamiento)
    return palabras, [mapeo.get(p, 0) for p in palabras]          # 0 = <UNK>


def top_k(u: torch.Tensor, vocabulario, k: int) -> list[tuple[str, float]]:
    """Las k palabras mas probables y su probabilidad."""
    probabilidades = torch.softmax(u, dim=0)
    valores, indices = probabilidades.topk(k)
    return [(str(vocabulario[i]), float(v)) for v, i in zip(valores, indices)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Prueba a mano: CBOW vs CBOW con atención.")
    parser.add_argument("frases", nargs="*", help='Frases con un hueco, por ejemplo "la ___ de la casa ..."')
    parser.add_argument("--modelo", default=None, help="Respaldo .npz de atención (por defecto, el de la época más alta).")
    parser.add_argument("--k", type=int, default=8, help="Cantidad de palabras a mostrar.")
    args = parser.parse_args()

    ruta = Path(args.modelo) if args.modelo else ultimo_respaldo_atencion()
    modelo, configuracion = cargar_modelo_atencion(ruta)
    print(f"Modelo con atención: {ruta}")
    base = cargar_respaldo(ruta_raiz / configuracion["respaldo_de_partida"])   # el CBOW del que partio
    W_base = torch.tensor(base["W"], dtype=torch.float32)
    Wp_base = torch.tensor(base["W_prima"], dtype=torch.float32)
    vocabulario = base["vocabulario_palabras"]
    mapeo = {str(p): i for i, p in enumerate(vocabulario)}
    m = configuracion["tamanio_ventana"]

    for frase in args.frases or EJEMPLOS:
        print(f"\n=== {frase} ===")
        try:
            palabras, indices = armar_contexto(frase, mapeo, m)
        except ValueError as error:
            print(f"  {error}")
            continue
        desconocidas = [p for p, i in zip(palabras, indices) if i == 0]
        if desconocidas:
            print(f"  (fuera del vocabulario, se usan como <UNK>: {', '.join(desconocidas)})")

        contexto = torch.tensor([indices])                        # (1, C): un lote de una sola muestra
        with torch.no_grad():
            u_base = (W_base[contexto[0]].mean(dim=0) @ Wp_base)  # CBOW: promedio del contexto
            u_atencion, a = modelo(contexto)                      # a: (1, H, C+1)

        pesos = a[0].mean(dim=0)                                  # promedio de las cabezas: (C+1,)
        posiciones = [f"{d:+d}" for d in list(range(-m, 0)) + list(range(1, m + 1))]
        print("  Atención sobre el contexto:")
        for posicion, palabra, peso in zip(posiciones, palabras, pesos[:-1]):
            print(f"    {posicion:>3} {palabra:<15} {peso:6.1%} {'█' * round(float(peso) * 50)}")
        print(f"    nula{'':<15}{pesos[-1]:6.1%}")

        print(f"  {'#':>2}  {'CBOW (promedio)':<28}  {'CBOW + atención':<28}")
        for n, ((p1, v1), (p2, v2)) in enumerate(zip(top_k(u_base, vocabulario, args.k),
                                                     top_k(u_atencion[0], vocabulario, args.k)), start=1):
            print(f"  {n:>2}  {p1:<18} {v1:8.1%}  {p2:<18} {v2:8.1%}")


if __name__ == "__main__":
    main()
