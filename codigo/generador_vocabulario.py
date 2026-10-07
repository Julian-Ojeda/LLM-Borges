"""
Modulo de construccion de vocabulario.
"""

from collections import Counter
import numpy as np


def construir_vocabulario(
    tokens_corpus: list[str],
    frecuencia_minima: int = 1,
    token_desconocido: str = "<UNK>",
) -> np.ndarray:
    """
    Construye el arreglo vocabulario_palabras de tamaño |V|, ordenado de mas a menos frecuente.
    La posicion de cada palabra es el indice activo de su representacion one-hot.
    Las palabras que aparecen menos de `frecuencia_minima` veces se representan con el token desconocido.

    :param tokens_corpus: Secuencia de tokens del corpus.
    :param frecuencia_minima: Cantidad minima de apariciones para incluir una palabra.
    :param token_desconocido: Token reservado para palabras fuera de vocabulario (indice 0).
    :return: Arreglo de NumPy unidimensional con las |V| cadenas del vocabulario.
    """
    frecuencias = Counter(tokens_corpus)
    frecuencias.pop(token_desconocido, None)
    palabras = [palabra for palabra, cantidad in frecuencias.most_common() if cantidad >= frecuencia_minima]
    return np.array([token_desconocido] + palabras, dtype=object)
