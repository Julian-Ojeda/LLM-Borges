"""
Modulo de construccion de vocabulario.

Rol en el proyecto: a partir de los tokens arma la lista de las |V| palabras distintas.
La posicion de cada palabra en esa lista es su indice: es la fila de W que la representa
(su embedding) y la columna de W' que la predice. Por eso el vocabulario se guarda en cada .npz:
sin el, los numeros de W no se pueden asociar a palabras.

Ejemplo: tokens ['de', 'la', 'de', 'casa'] con frecuencia_minima=1
    -> ['<UNK>', 'de', 'la', 'casa']     (indice 0 reservado para palabras desconocidas)
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
    frecuencias = Counter(tokens_corpus)                 # {palabra: cantidad de apariciones}
    frecuencias.pop(token_desconocido, None)             # por si el corpus ya contenia el texto "<UNK>"
    # most_common() devuelve las palabras de mas a menos frecuente; se descartan las poco frecuentes
    palabras = [palabra for palabra, cantidad in frecuencias.most_common() if cantidad >= frecuencia_minima]
    # <UNK> va primero (indice 0); dtype=object guarda cadenas de cualquier largo
    return np.array([token_desconocido] + palabras, dtype=object)
