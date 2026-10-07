"""
Red neuronal CBOW en CuPy / NumPy, con la notacion y las dimensiones de las clases
("Implementacion de metodos de calculo para representacion de palabras"):

    W  (|V| x N): pesos entrada -> capa oculta. La fila i es la representacion contextual v_i.
    W' (N x |V|): pesos capa oculta -> salida. La columna j es el vector de salida v'_j.
    h  (N x 1), u, y, e, t (|V| x 1), EH (N x 1).

El entrenamiento es por descenso por el gradiente muestra a muestra (sin lotes).
Como las entradas son one-hot, W^t x equivale a tomar una fila de W: por eso se trabaja
con los indices de las palabras en lugar de armar vectores one-hot.
"""

import sys
import ctypes
import ast
from pathlib import Path
import numpy as np

# Cargar automaticamente bibliotecas dinamicas CUDA instaladas en el .venv
_ruta_venv = Path(__file__).resolve().parent.parent / ".venv"
_carpeta_nvidia = (
    _ruta_venv
    / "lib"
    / f"python{sys.version_info.major}.{sys.version_info.minor}"
    / "site-packages"
    / "nvidia"
)

if _carpeta_nvidia.exists():
    for _lib_dir in _carpeta_nvidia.glob("*/lib"):
        for _so_file in _lib_dir.glob("*.so*"):
            try:
                ctypes.CDLL(str(_so_file))
            except Exception:
                pass

try:
    import cupy as cp
    import cupyx

    _ = cp.zeros((1,), dtype=cp.float32)
    USAR_CUPY = True
    xp = cp
except Exception:
    USAR_CUPY = False
    xp = np


def inicializar_pesos(tamanio_vocabulario: int, dimension_embedding: int) -> tuple:
    """
    Inicializa W (|V| x N) con distribucion uniforme en [-0.1, 0.1] y W' (N x |V|) en ceros.

    :param tamanio_vocabulario: Cardinalidad del vocabulario |V|.
    :param dimension_embedding: Cantidad de unidades de la capa oculta N.
    :return: Tupla (W, W_prima).
    """
    W = xp.random.uniform(-0.1, 0.1, (tamanio_vocabulario, dimension_embedding)).astype(xp.float32)
    W_prima = xp.zeros((dimension_embedding, tamanio_vocabulario), dtype=xp.float32)
    return W, W_prima


def _sumar_en_filas(matriz, indices, valores) -> None:
    """
    matriz[indices] += valores, acumulando cuando un indice se repite
    (por ejemplo, la misma palabra dos veces en un contexto).
    """
    if USAR_CUPY:
        cupyx.scatter_add(matriz, indices, valores)
    else:
        np.add.at(matriz, indices, valores)


def calcular_capa_oculta(contexto, W):
    """
    h = (1/C) W^t (x_1 + ... + x_C) = (1/C) (v_1 + ... + v_C)

    :param contexto: Indices de las C palabras del contexto, forma (C,).
    :param W: Matriz de pesos de entrada (|V| x N).
    :return: h, vector columna de forma (N, 1).
    """
    C = contexto.shape[0]
    return (W[contexto].sum(axis=0) / C).reshape(-1, 1)


def paso_softmax_completo(contexto, objetivo, W, W_prima, eta: float):
    """
    Un paso de descenso por el gradiente para una muestra (contexto, palabra central O),
    con softmax sobre todo el vocabulario:

        u = W'^t h,   y = softmax(u),   E = -u_O + log sum_j' exp(u_j') = -log y_O
        e = y - t,    EH = W' e
        w'_ij  <- w'_ij  - eta e_j h_i          (toda la matriz W')
        W_Ic,. <- W_Ic,. - eta (1/C) EH^t       (para cada palabra Ic del contexto)

    W y W_prima se modifican en el lugar.

    :param contexto: Indices de las C palabras del contexto, forma (C,).
    :param objetivo: Indice O de la palabra central.
    :param eta: Tasa de aprendizaje.
    :return: Tupla (E, acierto) como escalares en el dispositivo; acierto indica si argmax(y) = O.
    """
    C = contexto.shape[0]

    h = calcular_capa_oculta(contexto, W)            # (N, 1)
    u = W_prima.T @ h                                # (|V|, 1): u_j = v'_j^t h
    y = xp.exp(u - u.max())                          # se resta el maximo por estabilidad numerica
    y /= y.sum()                                     # (|V|, 1): y_j = P(p_j | contexto)

    E = -xp.log(xp.maximum(y[objetivo, 0], 1e-12))
    acierto = xp.argmax(y) == objetivo

    e = y                                            # e = y - t (reutiliza la memoria de y)
    e[objetivo, 0] -= 1.0
    EH = W_prima @ e                                 # (N, 1), con W' antes de actualizarla

    W_prima -= eta * (h @ e.T)                       # (N, |V|)
    _sumar_en_filas(W, contexto, (-eta / C) * EH.T)  # (1, N) sumado a cada fila del contexto
    return E, acierto


def distribucion_negativos(frecuencias):
    """
    Distribucion unigrama elevada a 3/4 (Mikolov et al.), U'(p) = U(p)^{3/4} / Z.
    Se devuelve acumulada para poder muestrear con searchsorted.

    :param frecuencias: Cantidad de apariciones de cada palabra del vocabulario, forma (|V|,).
    :return: Distribucion acumulada, forma (|V|,).
    """
    p = np.asarray(frecuencias, dtype=np.float64) ** 0.75
    return xp.asarray(np.cumsum(p / p.sum()))


def muestrear_negativos(objetivo, cantidad: int, acumulada):
    """
    Arma P_sel para cada muestra: `cantidad` palabras tomadas de U'(p), sin incluir a la palabra objetivo p_O.

    :param objetivo: Indice de la palabra objetivo de cada muestra, forma (T,).
    :param cantidad: |P_sel|, cantidad de ejemplos negativos por muestra.
    :param acumulada: Distribucion acumulada devuelta por distribucion_negativos.
    :return: Indices de los ejemplos negativos, forma (T, cantidad).
    """
    def sortear(forma):
        indices = xp.searchsorted(acumulada, xp.random.random(forma))
        return xp.minimum(indices, len(acumulada) - 1)

    negativos = sortear((len(objetivo), cantidad))
    repetidos = negativos == objetivo[:, None]
    while bool(repetidos.any()):
        negativos[repetidos] = sortear(int(repetidos.sum()))
        repetidos = negativos == objetivo[:, None]
    return negativos


def paso_muestreo_negativo(contexto, objetivo, negativos, W, W_prima, eta: float):
    """
    Un paso de descenso por el gradiente para una muestra, con muestreo negativo:

        E = -log sigma(v'_O^t h) - sum_{j en P_sel} log sigma(-v'_j^t h)
        dE/du_j = sigma(u_j) - t_j              (t_j = 1 si j = O, 0 si j en P_sel)
        v'_j   <- v'_j   - eta (sigma(u_j) - t_j) h     solo para j en {O} U P_sel
        EH = sum_{j en {O} U P_sel} (sigma(u_j) - t_j) v'_j
        W_Ic,. <- W_Ic,. - eta (1/C) EH^t

    W y W_prima se modifican en el lugar.

    :param contexto: Indices de las C palabras del contexto, forma (C,).
    :param objetivo: Indice O de la palabra central.
    :param negativos: Indices de P_sel, forma (K,).
    :param eta: Tasa de aprendizaje.
    :return: Tupla (E, acierto) como escalares en el dispositivo; acierto indica si argmax(W'^t h) = O.
    """
    C = contexto.shape[0]

    h = calcular_capa_oculta(contexto, W)                          # (N, 1)
    palabras = xp.concatenate([objetivo.reshape(1), negativos])    # (1+K,): p_O y P_sel
    v_salida = W_prima[:, palabras]                                # (N, 1+K): columnas v'_j
    u = v_salida.T @ h                                             # (1+K, 1): u_j = v'_j^t h

    t = xp.zeros_like(u)
    t[0, 0] = 1.0
    # -log sigma(x) = log(1 + exp(-x))
    E = xp.logaddexp(0, -u[0, 0]) + xp.logaddexp(0, u[1:, 0]).sum()
    acierto = xp.argmax(W_prima.T @ h) == objetivo                 # solo para la metrica: prediccion sobre todo |V|

    g = 1.0 / (1.0 + xp.exp(-u)) - t                               # (1+K, 1): sigma(u_j) - t_j
    EH = v_salida @ g                                              # (N, 1)

    _sumar_en_filas(W_prima.T, palabras, -eta * (g @ h.T))         # (1+K, N): columnas de W' en {O} U P_sel
    _sumar_en_filas(W, contexto, (-eta / C) * EH.T)
    return E, acierto


def guardar_modelo(modelo: dict, ruta_archivo: str | Path) -> None:
    """
    Almacena el estado completo del modelo en un archivo binario comprimido .npz.

    :param modelo: Diccionario con W, W_prima, vocabulario_palabras, historial_perdida, historial_accuracy, etc.
    :param ruta_archivo: Ruta del archivo de destino (.npz).
    :return: None.
    """
    destino = Path(ruta_archivo)
    destino.parent.mkdir(parents=True, exist_ok=True)

    W_np = cp.asnumpy(modelo["W"]) if USAR_CUPY else np.asarray(modelo["W"])
    W_prima_np = cp.asnumpy(modelo["W_prima"]) if USAR_CUPY else np.asarray(modelo["W_prima"])
    vocabulario_palabras = modelo["vocabulario_palabras"]

    datos_guardar = {
        "W": W_np,
        "W_prima": W_prima_np,
        "vocabulario_palabras": np.asarray(vocabulario_palabras, dtype=object),
        "epoca_actual": int(modelo.get("epoca_actual", 0)),
        "historial_perdida": np.asarray(modelo.get("historial_perdida", []), dtype=np.float32),
        "historial_accuracy": np.asarray(modelo.get("historial_accuracy", []), dtype=np.float32),
        "tamanio_vocabulario": int(len(vocabulario_palabras)),
        "dimension_embedding": int(W_np.shape[1]),
    }

    if "configuracion" in modelo:
        datos_guardar["configuracion_json"] = str(modelo["configuracion"])

    np.savez_compressed(destino, **datos_guardar)
    print(f"Modelo guardado exitosamente en: '{destino}'")


def cargar_modelo(ruta_archivo: str | Path) -> dict:
    """
    Carga y restaura el estado del modelo desde un archivo comprimido .npz.

    :param ruta_archivo: Ruta al archivo .npz de respaldo.
    :return: Diccionario con W, W_prima, vocabulario_palabras, epoca_actual, historial_perdida e historial_accuracy.
    """
    origen = Path(ruta_archivo)
    if not origen.exists():
        raise FileNotFoundError(f"No se encontro el archivo de modelo en: '{origen}'")

    datos = np.load(origen, allow_pickle=True)
    W = xp.asarray(datos["W"], dtype=xp.float32)
    W_prima = xp.asarray(datos["W_prima"], dtype=xp.float32)
    vocabulario_palabras = datos["vocabulario_palabras"]

    modelo = {
        "W": W,
        "W_prima": W_prima,
        "vocabulario_palabras": vocabulario_palabras,
        "epoca_actual": int(datos.get("epoca_actual", 0)),
        "historial_perdida": [float(v) for v in datos.get("historial_perdida", [])],
        "historial_accuracy": [float(v) for v in datos.get("historial_accuracy", [])],
        "tamanio_vocabulario": len(vocabulario_palabras),
        "dimension_embedding": W.shape[1],
    }

    if "configuracion_json" in datos:
        modelo["configuracion"] = ast.literal_eval(str(datos["configuracion_json"]))

    print(f"Modelo cargado exitosamente desde: '{origen}' (Epoca cargada: {modelo['epoca_actual']})")
    return modelo
