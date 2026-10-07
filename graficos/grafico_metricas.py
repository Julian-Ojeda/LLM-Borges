"""
Grafica la perdida y el accuracy por época del entrenamiento CBOW (ventana 4).

Uso, desde la carpeta del proyecto:
    python graficos/grafico_metricas.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
ARCHIVO_METRICAS = CARPETA_PROYECTO / "respaldos" / "metricas_w4.csv"
ARCHIVO_SALIDA = CARPETA_PROYECTO / "graficos" / "metricas_w4_epoca70.png"
EPOCA_MAXIMA = 70

# 1. Leer el CSV (columnas: epoca, perdida, accuracy, segundos) hasta la época elegida
datos = np.loadtxt(ARCHIVO_METRICAS, delimiter=",", skiprows=1)
datos = datos[datos[:, 0] <= EPOCA_MAXIMA]
epocas = datos[:, 0]
perdida = datos[:, 1]
accuracy = datos[:, 2]

# 2. Dos gráficos separados, porque la pérdida y el accuracy tienen escalas distintas
figura, (eje_perdida, eje_accuracy) = plt.subplots(1, 2, figsize=(11, 4.3))

graficos = [
    (eje_perdida, perdida, "#2a78d6", "Pérdida promedio por época", "Entropía cruzada"),
    (eje_accuracy, accuracy, "#eb6834", "Accuracy por época", "Palabra central acertada"),
]

for eje, valores, color, titulo, etiqueta_y in graficos:
    eje.plot(epocas, valores, color=color, linewidth=2)
    eje.plot(epocas[-1], valores[-1], "o", color=color)  # marca el último valor

    # Se escribe solo el último valor, no uno por época
    texto = f"{valores[-1]:.3f}".replace(".", ",")
    eje.annotate(texto, (epocas[-1], valores[-1]), xytext=(-6, 10),
                 textcoords="offset points", ha="right", fontweight="bold")

    eje.set_title(titulo, loc="left", fontweight="bold")
    eje.set_xlabel("Época")
    eje.set_ylabel(etiqueta_y)
    eje.set_ylim(bottom=0)
    eje.grid(axis="y", color="#e1e0d9")
    eje.spines[["top", "right"]].set_visible(False)

# 3. Título general con los hiperparámetros y guardar la imagen
figura.suptitle("CBOW · |V| = 53.409 · N = 80 · ventana 4 (C = 8) · η = 0,05 · softmax completo, sin lotes",
                color="#52514e")
figura.tight_layout()
figura.savefig(ARCHIVO_SALIDA, dpi=150)
print(f"Gráfico guardado en: {ARCHIVO_SALIDA}")
