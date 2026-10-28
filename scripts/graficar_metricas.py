#!/usr/bin/env python3
"""
Grafica la perdida y el accuracy por época a partir del CSV de metricas (se puede usar durante el entrenamiento)
o de un modelo guardado (.npz).

Uso:
    python scripts/graficar_metricas.py respaldos/metricas_w4.csv
    python scripts/graficar_metricas.py respaldos/modelo_cbow_w4_epoca_50.npz
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def leer_metricas(ruta: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    :param ruta: Ruta a un CSV de metricas o a un modelo .npz.
    :return: Tupla (epocas, perdida, accuracy).
    """
    if ruta.suffix == ".npz":
        datos = np.load(ruta, allow_pickle=True)
        perdida = datos["historial_perdida"]
        accuracy = datos["historial_accuracy"] if "historial_accuracy" in datos else np.full(len(perdida), np.nan)
        return np.arange(1, len(perdida) + 1), perdida, accuracy

    tabla = np.loadtxt(ruta, delimiter=",", skiprows=1, ndmin=2)
    return tabla[:, 0], tabla[:, 1], tabla[:, 2]


def main() -> None:
    parser = argparse.ArgumentParser(description="Grafica perdida y accuracy por época del entrenamiento CBOW.")
    parser.add_argument("ruta", type=str, help="CSV de metricas o modelo .npz.")
    parser.add_argument("--salida", type=str, default=None, help="Imagen de salida (por defecto, junto al archivo).")
    args = parser.parse_args()

    ruta = Path(args.ruta)
    epocas, perdida, accuracy = leer_metricas(ruta)

    fig, (eje_perdida, eje_accuracy) = plt.subplots(1, 2, figsize=(13, 4.5))
    eje_perdida.plot(epocas, perdida, color="#2b5c8f", linewidth=2)
    eje_perdida.set_title("Pérdida promedio por época")
    eje_perdida.set_xlabel("Época")
    eje_perdida.set_ylabel("Pérdida (entropía cruzada)")
    eje_perdida.grid(alpha=0.3)

    eje_accuracy.plot(epocas, accuracy, color="#d95f02", linewidth=2)
    eje_accuracy.set_title("Accuracy por época (palabra central más probable)")
    eje_accuracy.set_xlabel("Época")
    eje_accuracy.set_ylabel("Accuracy")
    eje_accuracy.set_ylim(0, 1)
    eje_accuracy.grid(alpha=0.3)

    fig.tight_layout()
    salida = Path(args.salida) if args.salida else ruta.with_suffix(".png")
    fig.savefig(salida, dpi=120)
    print(f"Gráfico guardado en: '{salida}'")


if __name__ == "__main__":
    main()
