#!/usr/bin/env python3
"""
Entrena el PMC como modelo de lenguaje (consignas 1 y 2 del TP2): codigo/modelo_pmc_torch.py.

Entrada: los k tokens anteriores, representados con W del CBOW del TP1 (d = 80, época 254) y concatenados.
W se lee del respaldo y queda fija: no hay capa Embedding ni se entrena W.
Salida: "vocabulario" (softmax + entropia cruzada) o "vector" (tanh/logistica + error cuadratico).

Datos: una ventana de k tokens se corre de a 1 por el corpus; el objetivo es el token siguiente.
El 5% final del corpus no se usa para entrenar (validacion): ahi se miden la perdida y los contadores de
ranking (la palabra correcta quedo entre las primeras 5, 10 y 20).

Archivos de salida (en el directorio de la corrida, nunca se pisa uno existente):
    metricas.csv           por época: perdida de entrenamiento y de validacion y contadores de ranking
    modelo_pmc_mejor.npz   pesos de la época con menor perdida de validacion
    modelo_pmc_final.npz   pesos de la última época
Cada .npz guarda solo los pesos del PMC y la configuracion (W se vuelve a leer del respaldo del CBOW).

Uso:
    python scripts/entrenar_pmc_torch.py                                    # config/pmc.yaml
    python scripts/entrenar_pmc_torch.py --modo vector --k 8 --ocultas 256,256 --directorio respaldos/pmc/vector_k8
    python scripts/entrenar_pmc_torch.py --modo vector --salida logistica --directorio respaldos/pmc/vector_k8_logistica
    python scripts/entrenar_pmc_torch.py --lote 1 --optimizador sgd --tasa 0.01 --epocas 2 --directorio respaldos/pmc/vocabulario_k8_lote1_sgd
    python scripts/entrenar_pmc_torch.py --epocas 1 --max-muestras 100000 --directorio <carpeta>   # prueba corta
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))
os.chdir(ruta_raiz)

from codigo.configuracion import cargar_configuracion
from codigo.tokenizador import tokenizar_corpus
from codigo.entrenador_cbow import imprimir_resumen
from codigo.modelo_pmc_torch import CORTES_RANKING, cargar_respaldo, puesto_en_ranking


def evitar_suspension() -> None:
    """En Windows, pide que el equipo no se suspenda por inactividad mientras corre este proceso."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


@torch.no_grad()
def evaluar(modelo, contexto, objetivo, indices) -> dict:
    """Perdida de validacion y contadores de ranking (puesto de la palabra correcta <= 1, 5, 10, 20)."""
    suma = 0.0
    aciertos = {1: 0, **{c: 0 for c in CORTES_RANKING}}
    for bloque in torch.tensor(indices, device=contexto.device).split(1000):
        c, o = contexto[bloque], objetivo[bloque]
        suma += modelo.perdida(c, o).item() * len(bloque)
        puesto = puesto_en_ranking(modelo(c), o)
        for corte in aciertos:
            aciertos[corte] += (puesto <= corte).sum().item()
    n = len(indices)
    return {"perdida": suma / n, **{f"acc{c}": aciertos[c] / n for c in aciertos}}


def guardar(modelo, configuracion: dict, epoca: int, destino: Path) -> None:
    """Pesos del PMC (sin W) + configuracion como texto. Se lee sin pickle (ver scripts/probar_pmc.py)."""
    datos = {f"pmc.{nombre}": t.cpu().numpy() for nombre, t in modelo.pmc.state_dict().items()}
    datos["epoca"] = np.array(epoca)
    datos["configuracion_json"] = np.array(repr(configuracion))
    np.savez(destino, **datos)


def entrenar(configuracion: dict, max_muestras: int | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Este script necesita una GPU NVIDIA con PyTorch CUDA.")
    dispositivo = torch.device("cuda")
    from codigo.modelo_pmc_torch import ModeloLenguajePMC

    ruta_checkpoint = Path(configuracion["ruta_checkpoint"])
    directorio = Path(configuracion["directorio_respaldos"])
    k = configuracion["tamanio_contexto"]
    B = configuracion["tamanio_lote"]
    cantidad_epocas = configuracion["cantidad_epocas"]
    semilla = configuracion["semilla_aleatoria"]
    torch.manual_seed(semilla)

    ruta_metricas = directorio / "metricas.csv"
    if ruta_metricas.exists():
        raise FileExistsError(f"'{directorio}' ya tiene metricas de otro entrenamiento. Use otra carpeta o borre esa.")
    directorio.mkdir(parents=True, exist_ok=True)
    evitar_suspension()

    # 1. W del CBOW (solo lectura) y muestras "k anteriores -> siguiente"
    respaldo = cargar_respaldo(ruta_checkpoint)
    vocabulario = respaldo["vocabulario_palabras"]
    incluir_puntuacion = respaldo["configuracion"].get("incluir_puntuacion_y_numeros", True)
    tokens = tokenizar_corpus(configuracion["ruta_corpus"], incluir_puntuacion)
    mapeo = {str(p): i for i, p in enumerate(vocabulario)}
    ids = np.array([mapeo.get(t, 0) for t in tokens], dtype=np.int64)
    ventanas = sliding_window_view(ids, k + 1)
    contexto = torch.tensor(ventanas[:, :k], device=dispositivo)
    objetivo = torch.tensor(ventanas[:, k], device=dispositivo)
    T = len(ventanas)
    corte = int(T * (1 - configuracion["fraccion_validacion"]))
    indices_entrenamiento = np.arange(corte)
    indices_validacion = np.arange(corte + k, T)       # k ventanas de separacion
    if max_muestras is not None:
        indices_entrenamiento = indices_entrenamiento[:max_muestras]

    # 2. Modelo y optimizador
    modelo = ModeloLenguajePMC(respaldo["W"], k, configuracion["modo"], configuracion["capas_ocultas"],
                               configuracion["normalizacion"], configuracion["activacion"],
                               configuracion["activacion_salida"]).to(dispositivo)
    tasa = configuracion["tasa_aprendizaje"]
    if configuracion["optimizador"] == "adam":
        optimizador = torch.optim.Adam(modelo.pmc.parameters(), lr=tasa, betas=(0.85, 0.99), eps=1e-8)
    else:
        optimizador = torch.optim.SGD(modelo.pmc.parameters(), lr=tasa)
    total = len(indices_entrenamiento)
    pasos_totales = math.ceil(total / B) * cantidad_epocas
    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, lambda paso: 1.0 - paso / pasos_totales)

    d = respaldo["W"].shape[1]
    salida = (f"{len(vocabulario):,} unidades + softmax (entropía cruzada)" if configuracion["modo"] == "vocabulario"
              else f"{d} unidades + {configuracion['activacion_salida']} (error cuadrático, W escalada min-max)")
    imprimir_resumen("PMC COMO MODELO DE LENGUAJE (PyTorch)", {
        "Respaldo del CBOW": f"{ruta_checkpoint} (W fija, d = {d})",
        "Directorio de salida": str(directorio),
        "Entrada": f"k = {k} tokens concatenados -> {k * d} componentes (sin capa Embedding)",
        "Capas ocultas": f"{configuracion['capas_ocultas']} {configuracion['activacion']}"
                         f"{' + normalización por capa' if configuracion['normalizacion'] else ''}",
        "Salida": salida,
        "Parámetros entrenables": f"{sum(p.numel() for p in modelo.pmc.parameters()):,}",
        "Muestras": f"{total:,} de entrenamiento, {len(indices_validacion):,} de validación (final del corpus)",
        "Lote / optimizador / tasa": f"{B} / {configuracion['optimizador']} / {tasa:g} (decae linealmente a 0)",
        "Épocas": cantidad_epocas,
    })

    ruta_metricas.write_text("epoca,perdida_entrenamiento,perdida_validacion,acc1_validacion,acc5_validacion,"
                             "acc10_validacion,acc20_validacion,segundos\n", encoding="utf-8")
    configuracion_guardada = {**configuracion, "incluir_puntuacion_y_numeros": incluir_puntuacion}
    mejor = float("inf")

    def registrar(epoca: int, perdida_entrenamiento: float, segundos: float) -> float:
        modelo.eval()
        r = evaluar(modelo, contexto, objetivo, indices_validacion)
        modelo.train()
        with open(ruta_metricas, "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca},{perdida_entrenamiento:.6f},{r['perdida']:.6f},{r['acc1']:.6f},{r['acc5']:.6f},"
                          f"{r['acc10']:.6f},{r['acc20']:.6f},{segundos:.2f}\n")
        print(f"Época {epoca}: entrenamiento {perdida_entrenamiento:.4f} | validación {r['perdida']:.4f} | "
              f"top-5 {r['acc5']:.4f} | top-10 {r['acc10']:.4f} | top-20 {r['acc20']:.4f} (exacta {r['acc1']:.4f})",
              flush=True)
        return r["perdida"]

    registrar(0, float("nan"), 0.0)

    # 3. Entrenamiento por lotes
    for epoca in range(1, cantidad_epocas + 1):
        inicio = time.time()
        orden = torch.tensor(np.random.default_rng(semilla + epoca).permutation(indices_entrenamiento), device=dispositivo)
        suma = torch.zeros((), dtype=torch.float64, device=dispositivo)
        for lote in tqdm(orden.split(B), desc=f"Epoca {epoca}/{cantidad_epocas}", unit="lote", mininterval=30.0):
            perdida = modelo.perdida(contexto[lote], objetivo[lote])
            optimizador.zero_grad(set_to_none=True)
            perdida.backward()
            optimizador.step()
            planificador.step()
            suma += perdida.detach() * len(lote)
        perdida_validacion = registrar(epoca, suma.item() / total, time.time() - inicio)
        if perdida_validacion < mejor:
            mejor = perdida_validacion
            guardar(modelo, configuracion_guardada, epoca, directorio / "modelo_pmc_mejor.npz")
            print(f"  (mejor validación hasta ahora: guardada la época {epoca})", flush=True)
    guardar(modelo, configuracion_guardada, cantidad_epocas, directorio / "modelo_pmc_final.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el PMC como modelo de lenguaje.")
    parser.add_argument("--config", default="config/pmc.yaml")
    parser.add_argument("--modo", choices=["vocabulario", "vector"], default=None)
    parser.add_argument("--k", type=int, default=None, help="Tokens de contexto.")
    parser.add_argument("--ocultas", default=None, help="Unidades de las capas ocultas separadas por coma, ej. 512 o 256,256.")
    parser.add_argument("--lote", type=int, default=None)
    parser.add_argument("--tasa", type=float, default=None)
    parser.add_argument("--optimizador", choices=["adam", "sgd"], default=None)
    parser.add_argument("--salida", choices=["tanh", "logistica"], default=None, help="Activacion de salida (modo vector).")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--directorio", default=None)
    parser.add_argument("--max-muestras", type=int, default=None, help="Limita las muestras por época (prueba corta).")
    args = parser.parse_args()

    configuracion = cargar_configuracion(args.config)
    for clave, valor in [("modo", args.modo), ("tamanio_contexto", args.k), ("tamanio_lote", args.lote),
                         ("tasa_aprendizaje", args.tasa), ("optimizador", args.optimizador),
                         ("activacion_salida", args.salida), ("cantidad_epocas", args.epocas),
                         ("directorio_respaldos", args.directorio)]:
        if valor is not None:
            configuracion[clave] = valor
    if args.ocultas is not None:
        configuracion["capas_ocultas"] = [int(x) for x in args.ocultas.split(",")]
    entrenar(configuracion, args.max_muestras)
    print("ENTRENAMIENTO FINALIZADO", flush=True)


if __name__ == "__main__":
    main()
