"""
CBOW con una capa de auto-atencion (CuPy / NumPy, gradiente calculado a mano, sin autograd).

En CBOW la capa oculta es el promedio del contexto, h = (1/C) sum_c v_Ic: una atencion con pesos fijos 1/C.
Aca los pesos los decide una capa de atencion de H cabezas (d = N / H por cabeza). Para una muestra:

    X   = W[I_1..I_C] + P                      (C x N)  embeddings del contexto + posicionales
    q   = x_centro Wq                           (N)      consulta de la posicion central (vector aprendido)
    K   = X Wk,  V = X Wv                       (C x N)  se parten en H cabezas de d columnas
    s_hc = q_h . K_hc / sqrt(d),  s_h,nulo = b_nulo_h      puntuaciones (+ una "clave nula" por cabeza)
    a_h = softmax(s_h)                          (C + 1)  pesos de atencion
    z_h = sum_c a_hc V_hc + a_h,nulo v_nulo_h   (d)      salida de cada cabeza
    h   = [z_1 .. z_H] Wo                       (N)      reemplaza al promedio de CBOW
    u   = W'^t h,  y = softmax(u),  E = -log y_O         igual que en CBOW

La clave nula le da a cada cabeza un lugar donde "no mirar nada" sin tener que apoyarse en los signos.

Inicializacion: Wv = Wo = I, x_centro = 0 (=> q = 0 => atencion uniforme), P = 0, v_nulo = 0 y b_nulo = -10
(=> peso nulo ~ 0). Con eso h = promedio del contexto: el modelo arranca siendo exactamente el CBOW cargado.

Todas las funciones trabajan sobre lotes de B muestras y sirven tanto con CuPy como con NumPy
(scripts/verificar_gradientes_atencion.py compara el gradiente contra diferencias finitas en float64).
"""

from pathlib import Path
import numpy as np

from .modelo_cbow_cupy import USAR_CUPY

if USAR_CUPY:
    import cupy
    import cupyx

PARAMETROS_ATENCION = ("x_centro", "P", "Wq", "Wk", "Wv", "Wo", "b_nulo", "v_nulo")


def modulo_de(arreglo):
    """Devuelve cupy o numpy segun donde viva el arreglo."""
    return cupy.get_array_module(arreglo) if USAR_CUPY else np


def sumar_en_filas(matriz, indices, valores) -> None:
    """matriz[indices] += valores, acumulando los indices repetidos."""
    if USAR_CUPY and modulo_de(matriz) is cupy:
        cupyx.scatter_add(matriz, indices, valores)
    else:
        np.add.at(matriz, indices, valores)


def inicializar_atencion(N: int, C: int, H: int, semilla: int, modulo=np, dtype=np.float32) -> dict:
    """
    Parametros de la capa de atencion, inicializados para que el modelo arranque igual al CBOW.

    :param N: Dimension del embedding (debe ser multiplo de H).
    :param C: Cantidad de palabras de contexto.
    :param H: Cantidad de cabezas.
    """
    if N % H != 0:
        raise ValueError(f"La dimension del embedding ({N}) debe ser multiplo de la cantidad de cabezas ({H}).")
    d = N // H
    rng = np.random.default_rng(semilla)
    parametros = {
        "x_centro": np.zeros(N),
        "P": np.zeros((C, N)),
        "Wq": rng.normal(0.0, 1.0 / np.sqrt(N), (N, N)),
        "Wk": rng.normal(0.0, 1.0 / np.sqrt(N), (N, N)),
        "Wv": np.eye(N),
        "Wo": np.eye(N),
        "b_nulo": np.full(H, -10.0),
        "v_nulo": np.zeros((H, d)),
    }
    return {k: modulo.asarray(v, dtype=dtype) for k, v in parametros.items()}


def adelante(W, W_prima, p: dict, contexto, H: int) -> tuple:
    """
    Pasada hacia adelante para un lote.

    :param contexto: Indices del contexto (B x C).
    :return: (u de forma B x |V|, cache con los intermedios para la pasada hacia atras).
    """
    m = modulo_de(W)
    B, C = contexto.shape
    N = W.shape[1]
    d = N // H

    X = W[contexto] + p["P"]                                        # (B, C, N)
    q = (p["x_centro"] @ p["Wq"]).reshape(H, d)                     # (H, d)
    K = (X @ p["Wk"]).reshape(B, C, H, d)
    V = (X @ p["Wv"]).reshape(B, C, H, d)
    V_tot = m.concatenate([V, m.broadcast_to(p["v_nulo"][None, None], (B, 1, H, d))], axis=1)  # (B, C+1, H, d)

    s = m.einsum("hd,bchd->bhc", q, K) / np.sqrt(d)                 # (B, H, C)
    s = m.concatenate([s, m.broadcast_to(p["b_nulo"][None, :, None], (B, H, 1))], axis=2)      # (B, H, C+1)
    a = m.exp(s - s.max(axis=2, keepdims=True))
    a /= a.sum(axis=2, keepdims=True)

    Z = m.einsum("bhc,bchd->bhd", a, V_tot).reshape(B, N)           # cabezas concatenadas
    h = Z @ p["Wo"]                                                 # (B, N)
    u = h @ W_prima                                                 # (B, |V|)
    cache = {"contexto": contexto, "X": X, "q": q, "K": K, "V_tot": V_tot, "a": a, "Z": Z, "h": h}
    return u, cache


def perdida_salida(u, objetivo) -> tuple:
    """
    Softmax de la salida, perdida E = -log y_O y error e = y - t (se calcula sobre u, que se descarta).

    :return: (E por muestra (B), e (B x |V|), prediccion argmax (B)).
    """
    m = modulo_de(u)
    filas = m.arange(len(objetivo))
    prediccion = u.argmax(axis=1)
    u_max = u.max(axis=1, keepdims=True)
    u_O = u[filas, objetivo]
    u -= u_max
    m.exp(u, out=u)
    S = u.sum(axis=1, keepdims=True)
    u /= S
    E = m.log(S[:, 0]) + u_max[:, 0] - u_O
    u[filas, objetivo] -= 1.0
    return E, u, prediccion


def atras(e, W, W_prima, p: dict, cache: dict, H: int, gradiente_W: bool = True) -> dict:
    """
    Pasada hacia atras: gradiente del promedio de E en el lote respecto de todos los parametros.

    :param e: y - t (B x |V|).
    :param gradiente_W: Si es False no arma el gradiente de W (embeddings congelados).
    :return: Diccionario nombre -> gradiente (incluye "W_prima" y, si corresponde, "W").
    """
    m = modulo_de(e)
    X, q, K, V_tot, a, Z, h = (cache[k] for k in ("X", "q", "K", "V_tot", "a", "Z", "h"))
    B, C, N = X.shape
    d = N // H
    e = e / B

    g = {"W_prima": h.T @ e}                                        # (N, |V|)
    dh = e @ W_prima.T                                              # (B, N)    = EH de CBOW
    g["Wo"] = Z.T @ dh
    dZ = (dh @ p["Wo"].T).reshape(B, H, d)

    da = m.einsum("bhd,bchd->bhc", dZ, V_tot)                       # (B, H, C+1)
    dV_tot = m.einsum("bhc,bhd->bchd", a, dZ)                       # (B, C+1, H, d)
    ds = a * (da - (a * da).sum(axis=2, keepdims=True))             # derivada de la softmax
    g["b_nulo"] = ds[:, :, C].sum(axis=0)
    g["v_nulo"] = dV_tot[:, C].sum(axis=0)

    ds = ds[:, :, :C] / np.sqrt(d)
    dq = m.einsum("bhc,bchd->hd", ds, K).reshape(N)
    dK = m.einsum("bhc,hd->bchd", ds, q).reshape(B, C, N)
    dV = dV_tot[:, :C].reshape(B, C, N)

    g["Wq"] = m.outer(p["x_centro"], dq)
    g["x_centro"] = p["Wq"] @ dq
    X2 = X.reshape(B * C, N)
    g["Wk"] = X2.T @ dK.reshape(B * C, N)
    g["Wv"] = X2.T @ dV.reshape(B * C, N)
    dX = dK @ p["Wk"].T + dV @ p["Wv"].T                            # (B, C, N)
    g["P"] = dX.sum(axis=0)
    if gradiente_W:
        g["W"] = m.zeros_like(W)
        sumar_en_filas(g["W"], cache["contexto"], dX)
    return g


class Adam:
    """
    Adam como en microgpt.py (beta1 = 0.85, beta2 = 0.99), con una tasa por parametro.
    Los parametros con tasa 0 quedan congelados (no se les reserva memoria ni se actualizan).
    """

    def __init__(self, parametros: dict, tasas: dict, beta1: float = 0.85, beta2: float = 0.99, eps: float = 1e-8):
        self.tasas = {k: t for k, t in tasas.items() if t > 0}
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m1 = {k: modulo_de(parametros[k]).zeros_like(parametros[k]) for k in self.tasas}
        self.m2 = {k: modulo_de(parametros[k]).zeros_like(parametros[k]) for k in self.tasas}
        self.t = 0

    def paso(self, parametros: dict, gradientes: dict, factor: float = 1.0) -> None:
        self.t += 1
        c1 = 1.0 - self.beta1 ** self.t
        c2 = 1.0 - self.beta2 ** self.t
        for k, tasa in self.tasas.items():
            gr = gradientes[k]
            m1, m2 = self.m1[k], self.m2[k]
            m1 *= self.beta1
            m1 += (1.0 - self.beta1) * gr
            m2 *= self.beta2
            m2 += (1.0 - self.beta2) * gr * gr
            parametros[k] -= (tasa * factor / c1) * m1 / (modulo_de(m2).sqrt(m2 / c2) + self.eps)


def a_numpy(arreglo):
    return cupy.asnumpy(arreglo) if USAR_CUPY and modulo_de(arreglo) is cupy else np.asarray(arreglo)


def guardar_modelo_atencion(modelo: dict, ruta_archivo: str | Path) -> None:
    """
    Guarda W, W' y el vocabulario con las mismas claves que guardar_modelo (asi cargar_modelo y
    scripts/pruebas.py pueden leer los embeddings), mas los parametros de la atencion con prefijo "atencion_".
    """
    destino = Path(ruta_archivo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    W = a_numpy(modelo["W"])
    datos = {
        "W": W,
        "W_prima": a_numpy(modelo["W_prima"]),
        "vocabulario_palabras": np.asarray(modelo["vocabulario_palabras"], dtype=object),
        "epoca_actual": int(modelo["epoca_actual"]),
        "historial_perdida": np.asarray(modelo["historial_perdida"], dtype=np.float32),
        "historial_accuracy": np.asarray(modelo["historial_accuracy"], dtype=np.float32),
        "tamanio_vocabulario": int(W.shape[0]),
        "dimension_embedding": int(W.shape[1]),
        "cantidad_cabezas": int(modelo["cantidad_cabezas"]),
        "configuracion_json": str(modelo["configuracion"]),
    }
    for k in PARAMETROS_ATENCION:
        datos[f"atencion_{k}"] = a_numpy(modelo["atencion"][k])
    if "atencion_media" in modelo:
        datos["atencion_media"] = a_numpy(modelo["atencion_media"])
    np.savez_compressed(destino, **datos)
    print(f"Modelo guardado exitosamente en: '{destino}'", flush=True)
