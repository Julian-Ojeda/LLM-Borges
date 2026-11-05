#!/usr/bin/env python3
"""
Entrena el transformador para completar frases (codigo/modelo_completar_torch.py), siguiendo el material
de la catedra: bloques de transformacion con auto-atencion causal, vectores posicionales sinusoidales y
cabezal con la matriz E (los embeddings del CBOW). PyTorch en GPU, por lotes y con Adam.

Datos (Transformadores c, pag. 55-56): una ventana de N tokens se corre de a 1 token por el corpus; el modelo
predice el token siguiente y se usa la entropia cruzada como perdida.
El 5% final del corpus queda afuera del entrenamiento (validacion): ahi se miden los contadores de ranking.
Los signos de puntuacion se predicen como cualquier palabra (hacen falta para generar texto).

No modifica nada existente: el respaldo del CBOW solo se lee y todo se escribe en directorio_respaldos.

Archivos de salida (en directorio_respaldos):
    metricas.csv     por época: perdida de entrenamiento y de validacion, perplejidad y los contadores de
                     ranking (la palabra correcta quedo entre las primeras 1, 5, 10 y 20)
    modelo_completar_epoca_{k}.npz    E (con la clave W, como los respaldos del CBOW) y los pesos de los bloques

Uso:
    python scripts/entrenar_completar_torch.py                       # config/completar.yaml
    python scripts/entrenar_completar_torch.py --epocas 1 --max-muestras 100000 --directorio <carpeta>   # prueba
"""

import argparse
import math
import os
import re
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))
os.chdir(ruta_raiz)     # las rutas relativas (config/, datos/, respaldos/) se resuelven desde la raiz del proyecto

from codigo.configuracion import cargar_configuracion
from codigo.tokenizador import tokenizar_corpus
from codigo.entrenador_cbow import imprimir_resumen
from codigo.modelo_atencion_torch import cargar_respaldo
from codigo.modelo_completar_torch import SiguientePalabra, CORTES_RANKING, puesto_en_ranking, generar, unir_tokens

ESPACIO_MINIMO_BYTES = 500 * 1024**2        # si queda menos disco libre que esto (500 MB), no se guarda el .npz
COMIENZO_EJEMPLO = "volvió a su casa y"     # al final de cada época se genera una continuacion de esto


def evitar_suspension() -> None:
    """En Windows, pide que el equipo no se suspenda por inactividad mientras corre este proceso."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def tokenizar_texto(texto: str) -> list[str]:
    """Tokeniza un texto suelto igual que codigo/tokenizador.py (con puntuacion)."""
    return re.findall(r"\w+|[^\w\s]", texto.lower())


@torch.no_grad()
def evaluar(modelo, contexto, objetivo, indices) -> dict:
    """
    Perdida, perplejidad y los contadores de acierto por ranking sobre las muestras de validacion:
    para cada prediccion se mira en que puesto quedo la palabra correcta y se cuenta si entro
    entre las primeras 5, 10 y 20 (ademas del acierto exacto, puesto 1).
    """
    suma_E = 0.0
    aciertos = {1: 0, **{c: 0 for c in CORTES_RANKING}}             # contadores: {corte: cantidad de aciertos}
    for bloque in torch.tensor(indices, device=contexto.device).split(1000):
        u, _ = modelo(contexto[bloque])
        o = objetivo[bloque]
        suma_E += F.cross_entropy(u, o, reduction="sum").item()
        puesto = puesto_en_ranking(u, o)
        for corte in aciertos:
            aciertos[corte] += (puesto <= corte).sum().item()        # 1 si le pego dentro de ese corte, 0 si no
    n = len(indices)
    perdida = suma_E / n
    # Perplejidad = exp(perdida): "entre cuantas palabras duda" el modelo en promedio
    return {"perdida": perdida, "perplejidad": math.exp(min(perdida, 700)),
            **{f"acc{corte}": aciertos[corte] / n for corte in aciertos}}


def guardar(modelo, vocabulario, epoca, historial_perdida, configuracion, destino) -> None:
    """
    Guarda E con la clave "W" (como los respaldos del CBOW: scripts/pruebas.py la puede leer), el vocabulario,
    las dimensiones del modelo y todos los pesos de los bloques con el prefijo "transformador.".
    """
    datos = {
        "W": modelo.E.detach().cpu().numpy(),
        "vocabulario_palabras": np.asarray(vocabulario, dtype=object),
        "epoca_actual": epoca,
        "historial_perdida": np.asarray(historial_perdida, dtype=np.float32),
        "tamanio_contexto": modelo.N,
        "cantidad_cabezas": modelo.h,
        "cantidad_bloques": modelo.L,
        "configuracion_json": str(configuracion),
    }
    for nombre, tensor in modelo.state_dict().items():
        if nombre not in ("E", "vp"):                               # E ya esta como W; vp se recalcula
            datos[f"transformador.{nombre}"] = tensor.cpu().numpy()
    np.savez_compressed(destino, **datos)
    print(f"Modelo guardado exitosamente en: '{destino}'", flush=True)


def entrenar(configuracion: dict, max_muestras: int | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Este script necesita una GPU NVIDIA con PyTorch CUDA.")
    dispositivo = torch.device("cuda")

    # Parametros (ver config/completar.yaml)
    ruta_checkpoint = Path(configuracion["ruta_checkpoint"])
    directorio = Path(configuracion["directorio_respaldos"])
    N = configuracion["tamanio_contexto"]
    h = configuracion["cantidad_cabezas"]
    L = configuracion["cantidad_bloques"]
    B = configuracion["tamanio_lote"]
    cantidad_epocas = configuracion["cantidad_epocas"]
    semilla = configuracion["semilla_aleatoria"]
    torch.manual_seed(semilla)          # inicializacion reproducible de los pesos de los bloques

    # Protecciones para no pisar nada existente
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"No existe el respaldo de partida: '{ruta_checkpoint}'")
    if directorio.resolve() == ruta_checkpoint.resolve().parent:
        raise ValueError(f"directorio_respaldos ('{directorio}') debe ser distinto de la carpeta del respaldo de partida.")
    ruta_metricas = directorio / "metricas.csv"
    if ruta_metricas.exists():
        raise FileExistsError(f"'{directorio}' ya tiene metricas de otro entrenamiento. Use otra carpeta o borre esa.")
    directorio.mkdir(parents=True, exist_ok=True)

    evitar_suspension()

    # 1. Embeddings del CBOW (matriz E) y muestras "N anteriores -> siguiente"
    respaldo = cargar_respaldo(ruta_checkpoint)
    vocabulario = respaldo["vocabulario_palabras"]
    incluir_puntuacion = respaldo["configuracion"].get("incluir_puntuacion_y_numeros", True)
    tokens = tokenizar_corpus(configuracion["ruta_corpus"], incluir_puntuacion)
    mapeo = {str(palabra): indice for indice, palabra in enumerate(vocabulario)}
    ids = np.array([mapeo.get(token, 0) for token in tokens], dtype=np.int64)

    ventanas = sliding_window_view(ids, N + 1)          # (T, N+1): una ventana por posicion del corpus
    contexto = torch.tensor(ventanas[:, :N], device=dispositivo)     # (T, N) los N tokens de entrada
    objetivo = torch.tensor(ventanas[:, N], device=dispositivo)      # (T,) el token siguiente
    T = len(ventanas)

    # Division entrenamiento / validacion por posicion en el corpus (no al azar): el final es validacion.
    # Se dejan N ventanas de separacion para que ningun objetivo de entrenamiento quede dentro de una
    # ventana de validacion.
    corte = int(T * (1 - configuracion["fraccion_validacion"]))
    indices_entrenamiento = np.arange(corte)
    indices_validacion = np.arange(corte + N, T)
    if max_muestras is not None:
        indices_entrenamiento = indices_entrenamiento[:max_muestras]

    # 2. Modelo y optimizador (Adam con una tasa para E y otra para los bloques)
    modelo = SiguientePalabra(respaldo["W"], N, h, L).to(dispositivo)
    parametros_bloques = [p for nombre, p in modelo.named_parameters() if nombre != "E"]
    grupos = [
        {"params": [modelo.E], "lr": configuracion["tasa_embeddings"]},
        {"params": parametros_bloques, "lr": configuracion["tasa_bloques"]},
    ]
    for grupo in grupos:
        for parametro in grupo["params"]:
            parametro.requires_grad_(grupo["lr"] > 0)     # tasa 0 = congelado
    optimizador = torch.optim.Adam([g for g in grupos if g["lr"] > 0], betas=(0.85, 0.99), eps=1e-8)

    total = len(indices_entrenamiento)
    pasos_totales = math.ceil(total / B) * cantidad_epocas
    # La tasa baja linealmente hasta 0 a lo largo de todo el entrenamiento (como en microgpt.py)
    planificador = torch.optim.lr_scheduler.LambdaLR(optimizador, lambda paso: 1.0 - paso / pasos_totales)

    # Para el ejemplo de cada época: comienzo fijo (rellenado con puntos a la izquierda) y sorteo reproducible
    ids_ejemplo = [mapeo.get(t, 0) for t in tokenizar_texto(COMIENZO_EJEMPLO)]
    ids_ejemplo = [mapeo["."]] * max(0, N - len(ids_ejemplo)) + ids_ejemplo

    imprimir_resumen("TRANSFORMADOR PARA COMPLETAR FRASES (PyTorch)", {
        "Respaldo de partida": f"{ruta_checkpoint} (solo lectura; de ahí sale E)",
        "Directorio de salida": str(directorio),
        "Entrada": f"ventana de N = {N} tokens + vectores posicionales sinusoidales",
        "Bloques": f"L = {L}: auto-atención causal de h = {h} cabezas (dk = {modelo.d // h}) + PMC "
                   f"{modelo.d} -> {4 * modelo.d} -> {modelo.d} (GELU), residuales y normalización por capa",
        "Cabezal": "h_N^L · E^t (la misma E de la entrada)",
        "Parámetros": f"{sum(p.numel() for p in parametros_bloques):,} en los bloques + {modelo.E.numel():,} en E",
        "Muestras": f"{total:,} de entrenamiento, {len(indices_validacion):,} de validación (final del corpus)",
        "Tamaño de lote": f"{B} ({math.ceil(total / B):,} pasos por época)",
        "Tasas (Adam, decaimiento lineal)": f"bloques {configuracion['tasa_bloques']:g}, E {configuracion['tasa_embeddings']:g}",
        "Épocas": cantidad_epocas,
    })

    ruta_metricas.write_text("epoca,perdida_entrenamiento,perdida_validacion,perplejidad_validacion,"
                             "acc1_validacion,acc5_validacion,acc10_validacion,acc20_validacion,segundos\n",
                             encoding="utf-8")

    def registrar(epoca: int, perdida_entrenamiento: float, segundos: float) -> None:
        modelo.eval()
        r = evaluar(modelo, contexto, objetivo, indices_validacion)
        generador = torch.Generator(device=dispositivo).manual_seed(semilla)
        nuevos = generar(modelo, ids_ejemplo, 25, generador=generador, prohibidos=[0])
        modelo.train()
        with open(ruta_metricas, "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca},{perdida_entrenamiento:.6f},{r['perdida']:.6f},{r['perplejidad']:.2f},"
                          f"{r['acc1']:.6f},{r['acc5']:.6f},{r['acc10']:.6f},{r['acc20']:.6f},{segundos:.2f}\n")
        print(f"Época {epoca}: validación pérdida {r['perdida']:.4f} | perplejidad {r['perplejidad']:.1f} | "
              f"top-5 {r['acc5']:.4f} | top-10 {r['acc10']:.4f} | top-20 {r['acc20']:.4f} "
              f"(exacta {r['acc1']:.4f})", flush=True)
        print(f"  ejemplo: {COMIENZO_EJEMPLO} → {unir_tokens([str(vocabulario[i]) for i in nuevos])}", flush=True)

    registrar(0, float("nan"), 0.0)     # época 0: antes de entrenar

    # 3. Entrenamiento por lotes
    historial_perdida = []
    for epoca in range(1, cantidad_epocas + 1):
        inicio = time.time()
        orden = torch.tensor(np.random.default_rng(semilla + epoca).permutation(indices_entrenamiento), device=dispositivo)
        suma_E = torch.zeros((), dtype=torch.float64, device=dispositivo)
        barra = tqdm(orden.split(B), desc=f"Epoca {epoca}/{cantidad_epocas}", unit="lote", mininterval=30.0)
        for lote in barra:
            u, _ = modelo(contexto[lote])
            perdida = F.cross_entropy(u, objetivo[lote])     # promedio de L_EC = -log y_siguiente en el lote

            optimizador.zero_grad(set_to_none=True)
            perdida.backward()                                # autograd
            optimizador.step()
            planificador.step()
            suma_E += perdida.detach() * len(lote)

        historial_perdida.append(suma_E.item() / total)
        registrar(epoca, historial_perdida[-1], time.time() - inicio)

        if epoca % configuracion["frecuencia_respaldo"] == 0 or epoca == cantidad_epocas:
            destino = directorio / f"modelo_completar_epoca_{epoca}.npz"
            if destino.exists():
                print(f"AVISO: '{destino}' ya existe; no se sobrescribe.", flush=True)
            elif shutil.disk_usage(directorio).free < ESPACIO_MINIMO_BYTES:
                print(f"AVISO: poco espacio libre; se omite el respaldo de la época {epoca}.", flush=True)
            else:
                guardar(modelo, vocabulario, epoca, historial_perdida,
                        {**configuracion, "incluir_puntuacion_y_numeros": incluir_puntuacion,
                         "respaldo_de_partida": str(ruta_checkpoint)}, destino)


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena el transformador para completar frases.")
    parser.add_argument("--config", default="config/completar.yaml")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--directorio", default=None, help="Sobrescribe directorio_respaldos.")
    parser.add_argument("--max-muestras", type=int, default=None, help="Limita las muestras por época (prueba corta).")
    args = parser.parse_args()

    configuracion = cargar_configuracion(args.config)
    if args.epocas is not None:
        configuracion["cantidad_epocas"] = args.epocas
    if args.directorio is not None:
        configuracion["directorio_respaldos"] = args.directorio
    entrenar(configuracion, args.max_muestras)
    print("ENTRENAMIENTO FINALIZADO", flush=True)


if __name__ == "__main__":
    main()
