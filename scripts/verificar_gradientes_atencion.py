#!/usr/bin/env python3
"""
Verifica el modelo de atencion (codigo/modelo_atencion_cupy.py) en NumPy float64, con dimensiones chicas:

1. Con la inicializacion por defecto, h es exactamente el promedio del contexto (el CBOW).
2. El gradiente calculado a mano coincide con diferencias finitas centradas para todos los parametros
   (W, W', x_centro, P, Wq, Wk, Wv, Wo, b_nulo, v_nulo), con parametros al azar y varias cabezas.
   Incluye una palabra repetida en un contexto, para probar la acumulacion en las filas de W.

Uso:
    python scripts/verificar_gradientes_atencion.py
"""

import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from codigo.modelo_atencion_cupy import inicializar_atencion, adelante, perdida_salida, atras, PARAMETROS_ATENCION


def perdida(W, W_prima, p, contexto, objetivo, H):
    u, _ = adelante(W, W_prima, p, contexto, H)
    E, _, _ = perdida_salida(u, objetivo)
    return E.mean()


def main() -> None:
    rng = np.random.default_rng(1)
    Vtam, N, C, B, H = 13, 8, 4, 5, 2
    W = rng.normal(0, 1, (Vtam, N))
    W_prima = rng.normal(0, 1, (N, Vtam))
    contexto = rng.integers(0, Vtam, (B, C))
    contexto[0, 1] = contexto[0, 3]
    objetivo = rng.integers(0, Vtam, B)

    # 1. Arranque identico al CBOW
    p0 = inicializar_atencion(N, C, H, semilla=0, dtype=np.float64)
    _, cache = adelante(W, W_prima, p0, contexto, H)
    diferencia = np.abs(cache["h"] - W[contexto].mean(axis=1)).max()
    # La clave nula arranca con peso e^-10 / C ~ 1e-5 (no exactamente 0), de ahi la tolerancia
    print(f"Inicializacion: max |h - promedio del contexto| = {diferencia:.2e}")
    assert diferencia < 1e-4

    # 2. Gradiente contra diferencias finitas, con parametros al azar
    p = {k: rng.normal(0, 0.5, v.shape) for k, v in p0.items()}
    u, cache = adelante(W, W_prima, p, contexto, H)
    _, e, _ = perdida_salida(u, objetivo)
    g = atras(e, W, W_prima, p, cache, H, gradiente_W=True)

    todos = {"W": W, "W_prima": W_prima, **p}
    paso, peor = 1e-6, 0.0
    for nombre in ("W", "W_prima") + PARAMETROS_ATENCION:
        arreglo = todos[nombre]
        numerico = np.zeros_like(arreglo)
        for i in np.ndindex(arreglo.shape):
            original = arreglo[i]
            arreglo[i] = original + paso
            mas = perdida(W, W_prima, p, contexto, objetivo, H)
            arreglo[i] = original - paso
            menos = perdida(W, W_prima, p, contexto, objetivo, H)
            arreglo[i] = original
            numerico[i] = (mas - menos) / (2 * paso)
        error = np.abs(g[nombre] - numerico).max() / max(np.abs(numerico).max(), 1e-12)
        peor = max(peor, error)
        print(f"{nombre:9s} error relativo maximo {error:.2e}")

    assert peor < 1e-6, "El gradiente a mano no coincide con las diferencias finitas"
    print("OK: el gradiente a mano coincide con las diferencias finitas.")


if __name__ == "__main__":
    main()
