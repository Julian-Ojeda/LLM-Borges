#!/usr/bin/env python3
"""
Evalua el modelo de "palabra siguiente" con contadores de ranking: para cada prediccion de validacion
se calcula en que puesto quedo la palabra correcta (1 = la mas probable) y se cuenta si entro entre
las primeras 5, 10 y 20. Esos son los contadores de "le pego / no le pego".

Usa las mismas muestras de validacion que el entrenamiento (el 5% final del corpus), que el modelo nunca vio.

Uso:
    python scripts/evaluar_ranking.py                          # todas las épocas guardadas de respaldos/transformador
    python scripts/evaluar_ranking.py --modelo respaldos/transformador/modelo_completar_epoca_5.npz
    python scripts/evaluar_ranking.py --detalle                # ademas, un CSV con cada prediccion de la mejor época

El CSV de --detalle (ranking_predicciones.csv) tiene una fila por prediccion:
    contexto, palabra_real, puesto, entre_5, entre_10, entre_20, prediccion_1 .. prediccion_5
"""

import argparse
import csv
import re
import sys
from pathlib import Path
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))

from codigo.tokenizador import tokenizar_corpus
from codigo.modelo_completar_torch import CORTES_RANKING, puesto_en_ranking, cargar_transformador
from generar_texto import DIRECTORIO


def muestras_validacion(configuracion: dict, vocabulario) -> tuple[np.ndarray, np.ndarray]:
    """Arma las mismas muestras de validacion que scripts/entrenar_completar_torch.py."""
    k = configuracion["tamanio_contexto"]
    tokens = tokenizar_corpus(ruta_raiz / configuracion["ruta_corpus"], configuracion["incluir_puntuacion_y_numeros"])
    mapeo = {str(p): i for i, p in enumerate(vocabulario)}
    ids = np.array([mapeo.get(t, 0) for t in tokens], dtype=np.int64)
    ventanas = sliding_window_view(ids, k + 1)
    corte = int(len(ventanas) * (1 - configuracion["fraccion_validacion"]))
    validacion = ventanas[corte + k:]                    # misma separacion de k ventanas que en el entrenamiento
    return validacion[:, :k], validacion[:, k]


@torch.no_grad()
def rankear(modelo, contexto: np.ndarray, objetivo: np.ndarray, dispositivo) -> tuple[np.ndarray, np.ndarray]:
    """Puesto de la palabra correcta y las 5 primeras predicciones, para cada muestra."""
    modelo.to(dispositivo)
    puestos, top5 = [], []
    for desde in range(0, len(objetivo), 1000):
        ctx = torch.tensor(contexto[desde:desde + 1000], device=dispositivo)
        o = torch.tensor(objetivo[desde:desde + 1000], device=dispositivo)
        u, _ = modelo(ctx)
        puestos.append(puesto_en_ranking(u, o).cpu().numpy())
        top5.append(u.topk(5, dim=1).indices.cpu().numpy())
    return np.concatenate(puestos), np.concatenate(top5)


def main() -> None:
    parser = argparse.ArgumentParser(description="Contadores de ranking (top-5, top-10, top-20) del modelo de palabra siguiente.")
    parser.add_argument("--modelo", default=None, help="Un respaldo .npz (por defecto, todas las épocas guardadas).")
    parser.add_argument("--detalle", action="store_true", help="Guarda un CSV con cada predicción de la mejor época.")
    args = parser.parse_args()

    if args.modelo:
        rutas = [Path(args.modelo)]
    else:
        rutas = sorted(DIRECTORIO.glob("modelo_completar_epoca_*.npz"),
                       key=lambda r: int(re.search(r"epoca_(\d+)", r.name).group(1)))
    if not rutas:
        raise FileNotFoundError(f"No hay respaldos en '{DIRECTORIO}'")
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    contexto = objetivo = None
    resultados = []
    print(f"{'época':>5} | {'entre 5':>18} | {'entre 10':>18} | {'entre 20':>18} | {'exacta':>8}")
    for ruta in rutas:
        modelo, vocabulario, configuracion = cargar_transformador(ruta)
        if contexto is None:
            contexto, objetivo = muestras_validacion(configuracion, vocabulario)
        puestos, top5 = rankear(modelo, contexto, objetivo, dispositivo)
        n = len(puestos)
        contadores = {corte: int((puestos <= corte).sum()) for corte in CORTES_RANKING}   # aciertos por corte
        epoca = int(re.search(r"epoca_(\d+)", ruta.name).group(1))
        resultados.append((contadores[CORTES_RANKING[0]], ruta, puestos, top5, vocabulario))
        celdas = " | ".join(f"{contadores[c]:>7,}/{n:,} {contadores[c] / n:5.1%}" for c in CORTES_RANKING)
        print(f"{epoca:>5} | {celdas} | {(puestos == 1).mean():8.1%}")

    if args.detalle:
        _, ruta, puestos, top5, vocabulario = max(resultados, key=lambda r: r[0])   # la de mas aciertos entre 5
        destino = DIRECTORIO / "ranking_predicciones.csv"
        with open(destino, "w", encoding="utf-8", newline="") as archivo:
            escritor = csv.writer(archivo)
            escritor.writerow(["contexto", "palabra_real", "puesto"] + [f"entre_{c}" for c in CORTES_RANKING]
                              + [f"prediccion_{i}" for i in range(1, 6)])
            for ctx, o, puesto, mejores in zip(contexto, objetivo, puestos, top5):
                escritor.writerow([" ".join(str(vocabulario[i]) for i in ctx), str(vocabulario[o]), int(puesto)]
                                  + [int(puesto <= c) for c in CORTES_RANKING]
                                  + [str(vocabulario[i]) for i in mejores])
        print(f"\nDetalle de cada predicción ({ruta.name}) guardado en: {destino}")


if __name__ == "__main__":
    main()
