#!/usr/bin/env python3
"""
Ajuste fino del CBOW sin reentrenar desde cero: se carga un respaldo (.npz) y se sigue entrenando
solo con las muestras cuya palabra central NO es un signo de puntuacion. Los signos siguen
apareciendo en el contexto (el modelo aprendio a usarlos ahi), pero dejan de ser objetivo.

Reusa los kernels CUDA de entrenar_cuda.py (el paso por muestra es identico) y no modifica nada existente:
    - el respaldo de partida solo se lee;
    - todo se escribe en un directorio aparte (directorio_respaldos del YAML), que debe ser distinto
      del directorio del respaldo de partida y no debe tener metricas previas;
    - nunca se sobrescribe un .npz existente.
Para deshacer el ajuste alcanza con borrar ese directorio.

Ademas de las metricas habituales (metricas_w{m}.csv, mismo formato que entrenar_cuda.py), cada época
evalua una muestra fija de 20.000 ejemplos y escribe evaluacion_w{m}.csv con:
    acc_palabras             acierto (argmax sobre todo |V|) cuando la palabra central es una palabra
    acc_palabras_sin_signos  idem, sin dejar que la salida elija un signo
    acc_signos               acierto cuando la palabra central es un signo
    pred_signos              fraccion de predicciones que son un signo
La primera fila es el respaldo de partida, sin ajustar (linea base).

Uso:
    python scripts/ajustar_puntuacion_cuda.py                       # config/ajuste_puntuacion.yaml
    python scripts/ajustar_puntuacion_cuda.py --epocas 1 --max-muestras 20000 --directorio <carpeta>   # prueba corta
"""

import argparse
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

ruta_scripts = Path(__file__).resolve().parent
if str(ruta_scripts) not in sys.path:
    sys.path.append(str(ruta_scripts))

# entrenar_cuda agrega la raiz del proyecto al path y se ubica en ella
from entrenar_cuda import PasoCBOWCuda
from codigo.configuracion import cargar_configuracion
from codigo.tokenizador import tokenizar_corpus
from codigo.entrenador_cbow import preparar_muestras, imprimir_resumen, TOKEN_DESCONOCIDO
from codigo.modelo_cbow_cupy import guardar_modelo, cargar_modelo, USAR_CUPY, xp

MUESTRAS_EVALUACION = 20000                 # tamaño de la muestra fija con la que se evalua cada época
ESPACIO_MINIMO_BYTES = 500 * 1024**2        # si queda menos disco libre que esto (500 MB), no se guarda el .npz


def es_signo(token: str) -> bool:
    """Un token es signo de puntuacion si no tiene ningun caracter alfanumerico (y no es <UNK>)."""
    return token != TOKEN_DESCONOCIDO and not any(caracter.isalnum() for caracter in str(token))


def evitar_suspension() -> None:
    """En Windows, pide que el equipo no se suspenda por inactividad mientras corre este proceso."""
    if os.name == "nt":
        import ctypes
        # Llamada a la API de Windows: "el sistema es necesario" mientras viva el proceso.
        # No evita que se apague si se agota la bateria.
        ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001
        ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)


def evaluar(W, W_prima, contexto, objetivo, indices, mascara_signos, penalizacion_signos) -> dict:
    """Metricas sobre una muestra fija, sin actualizar pesos. h = promedio del contexto, como en el entrenamiento."""
    aciertos_pal = aciertos_pal_sin = aciertos_sig = predichos_sig = 0
    total_pal = total_sig = 0
    # Se procesa en 40 bloques de ~500 muestras: u de todas juntas (20.000 x 53.409) no entra en la GPU
    for bloque in np.array_split(indices, 40):
        bloque = xp.asarray(bloque)
        o = objetivo[bloque]                                   # palabras centrales del bloque
        u = W[contexto[bloque]].mean(axis=1) @ W_prima         # (B, |V|): h de cada muestra por W'
        pred = u.argmax(axis=1)                                # prediccion libre
        pred_sin = (u + penalizacion_signos).argmax(axis=1)    # prediccion sin permitir signos (-1e30 en sus columnas)
        sig = mascara_signos[o]                                # True si la palabra central es un signo
        pal = ~sig                                             # True si es una palabra
        aciertos_pal += int(((pred == o) & pal).sum())
        aciertos_pal_sin += int(((pred_sin == o) & pal).sum())
        aciertos_sig += int(((pred == o) & sig).sum())
        predichos_sig += int(mascara_signos[pred].sum())
        total_pal += int(pal.sum())
        total_sig += int(sig.sum())
    return {
        "acc_palabras": aciertos_pal / max(total_pal, 1),
        "acc_palabras_sin_signos": aciertos_pal_sin / max(total_pal, 1),
        "acc_signos": aciertos_sig / max(total_sig, 1),
        "pred_signos": predichos_sig / len(indices),
    }


def ajustar(configuracion: dict, max_muestras: int | None = None) -> None:
    if not USAR_CUPY:
        raise RuntimeError("Este script necesita una GPU NVIDIA con CuPy.")

    # Parametros (ver config/ajuste_puntuacion.yaml)
    ruta_checkpoint = Path(configuracion["ruta_checkpoint"])        # CBOW de partida (solo se lee)
    directorio = Path(configuracion["directorio_respaldos"])        # donde se escribe todo lo nuevo
    tamanio_ventana = configuracion["tamanio_ventana"]              # m (debe coincidir con el respaldo)
    eta = configuracion["tasa_aprendizaje"]
    cantidad_epocas = configuracion["cantidad_epocas"]
    frecuencia_respaldo = configuracion["frecuencia_respaldo"]
    semilla = configuracion["semilla_aleatoria"]

    # Protecciones para no pisar nada existente
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"No existe el respaldo de partida: '{ruta_checkpoint}'")
    if directorio.resolve() == ruta_checkpoint.resolve().parent:
        raise ValueError(f"directorio_respaldos ('{directorio}') debe ser distinto de la carpeta del respaldo de partida.")
    ruta_metricas = directorio / f"metricas_w{tamanio_ventana}.csv"
    ruta_evaluacion = directorio / f"evaluacion_w{tamanio_ventana}.csv"
    if ruta_metricas.exists() or ruta_evaluacion.exists():
        raise FileExistsError(f"'{directorio}' ya tiene metricas de otro ajuste. Use otra carpeta o borre esa.")
    directorio.mkdir(parents=True, exist_ok=True)

    evitar_suspension()

    modelo = cargar_modelo(ruta_checkpoint)
    W = xp.ascontiguousarray(modelo["W"], dtype=xp.float32)
    W_prima = xp.ascontiguousarray(modelo["W_prima"], dtype=xp.float32)
    vocabulario = modelo["vocabulario_palabras"]
    epoca_inicial = modelo["epoca_actual"]
    historial_perdida = list(modelo["historial_perdida"])
    historial_accuracy = list(modelo["historial_accuracy"])
    configuracion_guardada = modelo.get("configuracion", {})
    incluir_puntuacion = configuracion_guardada.get("incluir_puntuacion_y_numeros", True)
    if configuracion_guardada.get("tamanio_ventana", tamanio_ventana) != tamanio_ventana:
        raise ValueError("tamanio_ventana no coincide con el del respaldo de partida.")

    tokens = tokenizar_corpus(configuracion["ruta_corpus"], incluir_puntuacion)
    mapeo = {palabra: indice for indice, palabra in enumerate(vocabulario)}
    _, contexto_cpu, objetivo_cpu = preparar_muestras(tokens, mapeo, tamanio_ventana)
    contexto = xp.ascontiguousarray(xp.asarray(contexto_cpu, dtype=xp.int32))
    objetivo = xp.ascontiguousarray(xp.asarray(objetivo_cpu, dtype=xp.int32))

    signos_cpu = np.array([es_signo(p) for p in vocabulario])      # (|V|,) True en los 27 signos
    mascara_signos = xp.asarray(signos_cpu)
    # (1, |V|): 0 en las palabras y -1e30 en los signos; sumado a u hace que un signo nunca sea el maximo
    penalizacion_signos = xp.where(mascara_signos, xp.float32(-1e30), xp.float32(0))[None, :]
    # EL CAMBIO DEL AJUSTE: solo se entrena con las muestras cuya palabra central NO es un signo.
    # El kernel no cambia: simplemente se le pasan menos indices de muestra
    indices_entrenamiento = np.flatnonzero(~signos_cpu[objetivo_cpu])
    if max_muestras is not None:
        indices_entrenamiento = indices_entrenamiento[:max_muestras]
    # Muestra de evaluacion fija (semilla 0): la misma en todas las épocas y en todos los scripts,
    # asi los numeros son comparables entre el CBOW original, el ajuste y la atencion
    indices_evaluacion = np.random.default_rng(0).choice(len(objetivo_cpu), MUESTRAS_EVALUACION, replace=False)

    paso = PasoCBOWCuda(W.shape[0], W.shape[1], contexto.shape[1])

    imprimir_resumen("AJUSTE FINO SIN SIGNOS COMO OBJETIVO (CUDA)", {
        "Respaldo de partida": f"{ruta_checkpoint} (Época {epoca_inicial}, solo lectura)",
        "Directorio de salida": str(directorio),
        "Signos de puntuación en el vocabulario": f"{int(signos_cpu.sum())}: {' '.join(map(str, np.asarray(vocabulario)[signos_cpu]))}",
        "Muestras totales": f"{len(objetivo_cpu):,}",
        "Muestras por época (objetivo = palabra)": f"{len(indices_entrenamiento):,}",
        "Tasa de aprendizaje (eta)": eta,
        "Épocas a entrenar": cantidad_epocas,
        "Respaldo cada": f"{frecuencia_respaldo} épocas (y la última)",
    })

    ruta_metricas.write_text("epoca,perdida,accuracy,segundos\n", encoding="utf-8")
    ruta_evaluacion.write_text("epoca,acc_palabras,acc_palabras_sin_signos,acc_signos,pred_signos\n", encoding="utf-8")

    def registrar_evaluacion(epoca: int) -> None:
        m = evaluar(W, W_prima, contexto, objetivo, indices_evaluacion, mascara_signos, penalizacion_signos)
        with open(ruta_evaluacion, "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca},{m['acc_palabras']:.6f},{m['acc_palabras_sin_signos']:.6f},"
                          f"{m['acc_signos']:.6f},{m['pred_signos']:.6f}\n")
        print(f"Evaluación época {epoca}: " + " | ".join(f"{k} {v:.4f}" for k, v in m.items()), flush=True)

    registrar_evaluacion(epoca_inicial)   # linea base: el respaldo de partida, antes de entrenar

    epoca_final = epoca_inicial + cantidad_epocas
    total = len(indices_entrenamiento)
    for epoca in range(epoca_inicial + 1, epoca_final + 1):
        inicio = time.time()
        orden = np.random.default_rng(semilla + epoca).permutation(indices_entrenamiento)
        paso.reiniciar_metricas()
        barra = tqdm(orden, desc=f"Epoca {epoca}/{epoca_final}", unit="muestra", mininterval=30.0)
        for procesadas, indice in enumerate(barra, start=1):
            paso(W, W_prima, contexto, objetivo, int(indice), eta)
            if procesadas % 100000 == 0:
                perdida, aciertos = paso.leer_metricas()
                barra.set_postfix({"Perdida": f"{perdida / procesadas:.4f}", "Acc": f"{aciertos / procesadas:.3f}"})

        perdida, aciertos = paso.leer_metricas()
        duracion = time.time() - inicio
        historial_perdida.append(perdida / total)
        historial_accuracy.append(aciertos / total)
        with open(ruta_metricas, "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca},{historial_perdida[-1]:.6f},{historial_accuracy[-1]:.6f},{duracion:.2f}\n")
        print(f"Epoca {epoca} finalizada. Perdida: {historial_perdida[-1]:.4f} | "
              f"Accuracy (palabras): {historial_accuracy[-1]:.4f} | Tiempo: {duracion:.2f}s", flush=True)
        registrar_evaluacion(epoca)

        if (epoca - epoca_inicial) % frecuencia_respaldo == 0 or epoca == epoca_final:
            destino = directorio / f"modelo_cbow_w{tamanio_ventana}_ajuste_epoca_{epoca}.npz"
            libre = shutil.disk_usage(directorio).free
            if destino.exists():
                print(f"AVISO: '{destino}' ya existe; no se sobrescribe.", flush=True)
            elif libre < ESPACIO_MINIMO_BYTES:
                print(f"AVISO: quedan {libre / 1024**2:.0f} MB libres; se omite el respaldo de la época {epoca}.", flush=True)
            else:
                guardar_modelo({
                    "W": W, "W_prima": W_prima, "vocabulario_palabras": vocabulario, "epoca_actual": epoca,
                    "historial_perdida": historial_perdida, "historial_accuracy": historial_accuracy,
                    "configuracion": {**configuracion_guardada, **configuracion, "ajuste": "sin_signos_como_objetivo",
                                      "incluir_puntuacion_y_numeros": incluir_puntuacion},
                }, destino)


def main() -> None:
    parser = argparse.ArgumentParser(description="Ajuste fino del CBOW sin signos de puntuación como objetivo.")
    parser.add_argument("--config", default="config/ajuste_puntuacion.yaml")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--tasa", type=float, default=None)
    parser.add_argument("--directorio", default=None, help="Sobrescribe directorio_respaldos.")
    parser.add_argument("--max-muestras", type=int, default=None, help="Limita las muestras por época (prueba corta).")
    args = parser.parse_args()

    configuracion = cargar_configuracion(args.config)
    if args.epocas is not None:
        configuracion["cantidad_epocas"] = args.epocas
    if args.tasa is not None:
        configuracion["tasa_aprendizaje"] = args.tasa
    if args.directorio is not None:
        configuracion["directorio_respaldos"] = args.directorio
    ajustar(configuracion, args.max_muestras)
    print("AJUSTE FINALIZADO", flush=True)


if __name__ == "__main__":
    main()
