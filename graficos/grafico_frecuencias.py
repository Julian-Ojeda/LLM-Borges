"""
Histograma de la distribución de frecuencia de las palabras del corpus.

Uso, desde la carpeta del proyecto:
    python graficos/grafico_frecuencias.py
"""

import sys
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CARPETA_PROYECTO = Path(__file__).resolve().parent.parent
sys.path.append(str(CARPETA_PROYECTO))
from codigo.tokenizador import tokenizar_corpus

ARCHIVO_CORPUS = CARPETA_PROYECTO / "datos" / "corpus_ApAvAu.txt"
ARCHIVO_SALIDA = CARPETA_PROYECTO / "graficos" / "distribucion_frecuencias.png"

# 1. Contar cuántas veces aparece cada palabra (con la misma tokenización del entrenamiento)
tokens = tokenizar_corpus(ARCHIVO_CORPUS, incluir_puntuacion_y_numeros=True)
frecuencias = np.array(list(Counter(tokens).values()))

# 2. Armar intervalos que se duplican: 1, 2-3, 4-7, 8-15, ...
#    Cada barra cubre el doble que la anterior, así la altura es la cantidad real de palabras.
#    Se resta 0,5 para que cada frecuencia entera caiga dentro de un solo intervalo.
potencias = 2 ** np.arange(0, int(np.log2(frecuencias.max())) + 2)
bordes = potencias - 0.5
cantidades, _ = np.histogram(frecuencias, bins=bordes)

# 3. Dibujar las barras con los dos ejes en escala logarítmica
#    (si no, la barra de las palabras que aparecen una sola vez tapa a todas las demás)
figura, eje = plt.subplots(figsize=(10, 4.8))
eje.bar(bordes[:-1], cantidades, width=np.diff(bordes), align="edge", color="#2a78d6", edgecolor="white")
eje.set_xscale("log", base=2)
eje.set_yscale("log")
eje.minorticks_off()
eje.set_xticks(potencias[::2])
eje.xaxis.set_major_formatter("{x:.0f}")
eje.yaxis.set_major_formatter("{x:.0f}")

# Se escribe solo la primera barra: las palabras que aparecen una sola vez
eje.annotate(str(cantidades[0]), (1, cantidades[0]), xytext=(4, 5), textcoords="offset points", fontweight="bold")

eje.set_title("Distribución de la frecuencia de las palabras", loc="left", fontweight="bold")
eje.set_xlabel("Frecuencia: apariciones en el corpus (cada barra va de n a 2n − 1)")
eje.set_ylabel("Cantidad de palabras")
eje.grid(axis="y", color="#e1e0d9")
eje.set_axisbelow(True)
eje.spines[["top", "right"]].set_visible(False)

# 4. Guardar la imagen
figura.tight_layout()
figura.savefig(ARCHIVO_SALIDA, dpi=150)
print(f"{len(frecuencias)} palabras distintas, {len(tokens)} tokens")
print(f"Gráfico guardado en: {ARCHIVO_SALIDA}")
