#!/usr/bin/env python3
"""
Pruebas de los embeddings CBOW: combinar palabras con sumas y restas de vectores
y buscar las palabras mas cercanas al resultado por producto punto y por distancia coseno.

Los vectores son las filas de W (la representacion contextual v_p de cada palabra).
Solo usa NumPy: no necesita GPU.

Uso:
    python pruebas.py                                   # ejemplos con el ultimo respaldo
    python pruebas.py "madre - mujer + hombre"          # una combinacion
    python pruebas.py "casa" --k 15                     # vecinos de una palabra
    python pruebas.py "rey - hombre + mujer" --modelo respaldos/modelo_cbow_w4_epoca_70.npz
"""

import argparse
import re
from pathlib import Path
import numpy as np

RUTA_RAIZ = Path(__file__).resolve().parent


def ultimo_respaldo(directorio: Path = RUTA_RAIZ / "respaldos") -> Path:
    """
    Busca el respaldo .npz de la época mas alta.

    :param directorio: Carpeta con los archivos modelo_cbow_w*_epoca_*.npz.
    :return: Ruta al respaldo mas reciente.
    """
    candidatos = list(directorio.glob("modelo_cbow_w*_epoca_*.npz"))
    if not candidatos:
        raise FileNotFoundError(f"No hay respaldos .npz en '{directorio}'")
    return max(candidatos, key=lambda ruta: int(re.search(r"epoca_(\d+)", ruta.name).group(1)))


def cargar_embeddings(ruta_modelo: str | Path) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """
    Carga W y el vocabulario de un respaldo .npz.

    :param ruta_modelo: Ruta al archivo .npz.
    :return: Tupla (W de forma (|V|, N), vocabulario, mapeo palabra -> indice).
    """
    datos = np.load(ruta_modelo, allow_pickle=True)
    W = datos["W"].astype(np.float32)
    vocabulario = np.array([str(palabra) for palabra in datos["vocabulario_palabras"]], dtype=object)
    mapeo = {palabra: indice for indice, palabra in enumerate(vocabulario)}

    epoca = int(datos["epoca_actual"]) if "epoca_actual" in datos else "?"
    print(f"Modelo: {Path(ruta_modelo).name} | época {epoca} | W {W.shape}")
    return W, vocabulario, mapeo


def interpretar_expresion(expresion: str) -> tuple[list[str], list[str]]:
    """
    Separa una expresion como "madre - mujer + hombre" en palabras que suman y que restan.

    :param expresion: Palabras unidas por '+' y '-'. La primera suma si no lleva signo.
    :return: Tupla (palabras positivas, palabras negativas).
    """
    positivas, negativas = [], []
    for signo, palabra in re.findall(r"([+-]?)\s*([^\s+-]+)", expresion.lower()):
        (negativas if signo == "-" else positivas).append(palabra)
    return positivas, negativas


def combinar_palabras(
    positivas: list[str],
    negativas: list[str],
    W: np.ndarray,
    mapeo: dict[str, int],
    normalizar: bool = False,
) -> np.ndarray:
    """
    Combina palabras sumando y restando sus vectores: v = sum(v_positivas) - sum(v_negativas).

    Con normalizar=True cada vector se lleva a norma 1 antes de sumar, para que las palabras
    frecuentes (que suelen tener vectores mas largos) no dominen la combinacion.

    :param positivas: Palabras que suman.
    :param negativas: Palabras que restan.
    :param W: Matriz de embeddings (|V| x N).
    :param mapeo: Diccionario palabra -> indice.
    :param normalizar: Si se normaliza cada vector antes de combinar.
    :return: Vector combinado de forma (N,).
    """
    ausentes = [palabra for palabra in positivas + negativas if palabra not in mapeo]
    if ausentes:
        raise KeyError(f"No están en el vocabulario: {', '.join(ausentes)}")
    if not positivas and not negativas:
        raise ValueError("La combinación no tiene palabras.")

    def vector(palabra: str) -> np.ndarray:
        v = W[mapeo[palabra]]
        return v / (np.linalg.norm(v) + 1e-12) if normalizar else v

    combinado = np.zeros(W.shape[1], dtype=np.float32)
    for palabra in positivas:
        combinado += vector(palabra)
    for palabra in negativas:
        combinado -= vector(palabra)
    return combinado


def palabras_mas_cercanas(
    vector: np.ndarray,
    W: np.ndarray,
    vocabulario: np.ndarray,
    metrica: str = "coseno",
    k: int = 10,
    excluir: list[str] | None = None,
    normas: np.ndarray | None = None,
) -> list[tuple[str, float]]:
    """
    Busca las k palabras mas cercanas a un vector.

    - producto_punto: puntaje = v_p . v   (mayor es mas cercano; favorece vectores largos)
    - coseno:         distancia = 1 - (v_p . v) / (|v_p| |v|)   (menor es mas cercano; entre 0 y 2)

    :param vector: Vector de consulta, forma (N,).
    :param W: Matriz de embeddings (|V| x N).
    :param vocabulario: Arreglo con las |V| palabras.
    :param metrica: 'producto_punto' o 'coseno'.
    :param k: Cantidad de palabras a devolver.
    :param excluir: Palabras a omitir del resultado (por ejemplo, las de la combinacion).
    :param normas: Normas precalculadas de las filas de W (opcional, para no recalcularlas).
    :return: Lista de tuplas (palabra, producto punto o distancia coseno).
    """
    producto = W @ vector
    if metrica == "producto_punto":
        puntajes = producto
        orden = np.argsort(-puntajes)
    elif metrica == "coseno":
        if normas is None:
            normas = np.linalg.norm(W, axis=1)
        puntajes = 1.0 - producto / (normas * np.linalg.norm(vector) + 1e-12)
        orden = np.argsort(puntajes)
    else:
        raise ValueError(f"Métrica no soportada: '{metrica}' (usar 'producto_punto' o 'coseno')")

    excluidas = set(excluir or []) | {"<UNK>"}
    resultado = []
    for indice in orden:
        if vocabulario[indice] in excluidas:
            continue
        resultado.append((vocabulario[indice], float(puntajes[indice])))
        if len(resultado) == k:
            break
    return resultado


def probar(expresion: str, W: np.ndarray, vocabulario: np.ndarray, mapeo: dict[str, int],
           normas: np.ndarray, k: int = 10) -> None:
    """
    Combina las palabras de la expresion y muestra las mas cercanas con ambas metricas.
    """
    positivas, negativas = interpretar_expresion(expresion)
    print(f"\n=== {expresion} ===")
    try:
        crudo = combinar_palabras(positivas, negativas, W, mapeo)
        normalizado = combinar_palabras(positivas, negativas, W, mapeo, normalizar=True)
    except (KeyError, ValueError) as error:
        print(f"  {error.args[0]}")
        return

    excluir = positivas + negativas
    por_punto = palabras_mas_cercanas(crudo, W, vocabulario, "producto_punto", k, excluir, normas)
    por_coseno = palabras_mas_cercanas(normalizado, W, vocabulario, "coseno", k, excluir, normas)

    print(f"  {'#':>2}  {'producto punto':<30}  {'distancia coseno':<30}")
    for posicion, ((p1, s1), (p2, s2)) in enumerate(zip(por_punto, por_coseno), start=1):
        print(f"  {posicion:>2}  {p1:<18} {s1:>10.3f}  {p2:<18} {s2:>10.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Combina palabras y busca las más cercanas (producto punto y coseno).")
    parser.add_argument("expresiones", nargs="*", help='Expresiones como "madre - mujer + hombre" o una sola palabra.')
    parser.add_argument("--modelo", type=str, default=None, help="Respaldo .npz (por defecto, el de la época más alta).")
    parser.add_argument("--k", type=int, default=10, help="Cantidad de palabras a mostrar (por defecto 10).")
    args = parser.parse_args()

    ruta_modelo = Path(args.modelo) if args.modelo else ultimo_respaldo()
    W, vocabulario, mapeo = cargar_embeddings(ruta_modelo)
    normas = np.linalg.norm(W, axis=1)

    expresiones = args.expresiones or [
        "mujer",
        "casa",
        "madre - mujer + hombre",
        "hermana - mujer + hombre",
        "muchacha - mujer + hombre",
        "ciudad - calle + habitación",
    ]
    for expresion in expresiones:
        probar(expresion, W, vocabulario, mapeo, normas, args.k)


if __name__ == "__main__":
    main()
