"""
Modulo de entrenamiento de la red neuronal CBOW, muestra a muestra (sin lotes).
Coordina la tokenizacion, el vocabulario, las muestras (contexto, palabra central) y los pasos de gradiente.
"""

import time
from pathlib import Path
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from tqdm import tqdm

from .tokenizador import tokenizar_corpus
from .generador_vocabulario import construir_vocabulario
from .modelo_cbow_cupy import (
    inicializar_pesos,
    paso_softmax_completo,
    paso_muestreo_negativo,
    distribucion_negativos,
    muestrear_negativos,
    guardar_modelo,
    cargar_modelo,
    USAR_CUPY,
    xp,
)

TOKEN_DESCONOCIDO = "<UNK>"


def preparar_muestras(
    tokens_corpus: list[str], mapeo_vocabulario: dict[str, int], tamanio_ventana: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convierte el corpus en indices y arma, para cada posicion j = m+1, ..., |D|-m,
    el contexto (p_{j-m}, ..., p_{j-1}, p_{j+1}, ..., p_{j+m}) y la palabra central p_j.

    :param tokens_corpus: Secuencia de tokens del corpus.
    :param mapeo_vocabulario: Diccionario palabra -> indice (las palabras ausentes van a <UNK>, indice 0).
    :param tamanio_ventana: Tamaño m de la ventana a izquierda y a derecha.
    :return: Tupla (ids del corpus, contexto de forma (T, C) con C = 2m, objetivo de forma (T,)).
    """
    ids = np.array([mapeo_vocabulario.get(token, 0) for token in tokens_corpus], dtype=np.int32)
    if len(ids) < 2 * tamanio_ventana + 1:
        raise ValueError("El corpus tiene menos tokens que una ventana de contexto completa.")

    ventanas = sliding_window_view(ids, 2 * tamanio_ventana + 1)
    objetivo = ventanas[:, tamanio_ventana].copy()
    contexto = np.delete(ventanas, tamanio_ventana, axis=1)
    return ids, contexto, objetivo


def imprimir_resumen(titulo: str, datos: dict) -> None:
    """
    Imprime en formato estructurado un diccionario de parametros.
    """
    print(f"\n================ {titulo} ================")
    for clave, valor in datos.items():
        print(f"{clave}: {valor}")
    print("=================================================================\n", flush=True)


def entrenar(configuracion_dict: dict) -> dict:
    """
    Ejecuta el pipeline completo de entrenamiento de la red CBOW.
    Permite iniciar un entrenamiento limpio o reanudar desde una copia de respaldo (.npz).
    Por cada época registra la perdida promedio y el accuracy (fraccion de muestras en que la
    palabra central fue la mas probable, medida sobre cada muestra antes de su actualizacion)
    en el modelo y en un archivo CSV para graficar.

    :param configuracion_dict: Diccionario que contiene todos los hiperparametros y rutas.
    :return: Diccionario 'modelo' con W, W_prima, vocabulario_palabras, historiales y metadatos.
    """
    tiempo_inicio_total = time.time()
    configuracion = dict(configuracion_dict)

    ruta_corpus = configuracion["ruta_corpus"]
    directorio_respaldos = Path(configuracion["directorio_respaldos"])
    tamanio_ventana = configuracion["tamanio_ventana"]
    tasa_aprendizaje = configuracion["tasa_aprendizaje"]
    cantidad_epocas = configuracion["cantidad_epocas"]
    cantidad_negativos = configuracion["cantidad_negativos"]
    hacer_respaldo = configuracion["hacer_respaldo"]
    frecuencia_respaldo = configuracion["frecuencia_respaldo"]
    ruta_checkpoint = configuracion["ruta_checkpoint"]
    semilla_aleatoria = configuracion["semilla_aleatoria"]

    xp.random.seed(semilla_aleatoria)

    # 1. Pesos y vocabulario: desde un checkpoint o inicializados desde el corpus
    if configuracion["reanudar_entrenamiento"]:
        if not Path(ruta_checkpoint).exists():
            raise FileNotFoundError(f"No existe el checkpoint para reanudar: '{ruta_checkpoint}'")
        modelo_cargado = cargar_modelo(ruta_checkpoint)
        W, W_prima = modelo_cargado["W"], modelo_cargado["W_prima"]
        vocabulario_palabras = modelo_cargado["vocabulario_palabras"]
        epoca_inicial = modelo_cargado["epoca_actual"]
        historial_perdida = modelo_cargado["historial_perdida"]
        historial_accuracy = modelo_cargado["historial_accuracy"]
        # El corpus se tokeniza igual que cuando se construyo el vocabulario guardado
        configuracion_guardada = modelo_cargado.get("configuracion", {})
        configuracion["incluir_puntuacion_y_numeros"] = configuracion_guardada.get(
            "incluir_puntuacion_y_numeros", configuracion["incluir_puntuacion_y_numeros"]
        )
        tokens_corpus = tokenizar_corpus(ruta_corpus, configuracion["incluir_puntuacion_y_numeros"])
        estado = f"Reanudando desde '{ruta_checkpoint}' (Época {epoca_inicial})"
    else:
        tokens_corpus = tokenizar_corpus(ruta_corpus, configuracion["incluir_puntuacion_y_numeros"])
        vocabulario_palabras = construir_vocabulario(
            tokens_corpus, configuracion["frecuencia_minima"], TOKEN_DESCONOCIDO
        )
        W, W_prima = inicializar_pesos(len(vocabulario_palabras), configuracion["dimension_embedding"])
        epoca_inicial = 0
        historial_perdida, historial_accuracy = [], []
        estado = "Entrenamiento limpio desde el corpus"

    tamanio_vocabulario, dimension_embedding = W.shape
    if tamanio_vocabulario != len(vocabulario_palabras) or W_prima.shape != (dimension_embedding, tamanio_vocabulario):
        raise ValueError(f"Dimensiones inconsistentes: W {W.shape}, W' {W_prima.shape}, |V| = {len(vocabulario_palabras)}")

    # 2. Muestras (contexto, palabra central) como indices, cargadas una sola vez en GPU/CPU
    mapeo_vocabulario = {palabra: indice for indice, palabra in enumerate(vocabulario_palabras)}
    ids_corpus, contexto_cpu, objetivo_cpu = preparar_muestras(tokens_corpus, mapeo_vocabulario, tamanio_ventana)
    contexto = xp.asarray(contexto_cpu)
    objetivo = xp.asarray(objetivo_cpu)
    total_muestras = len(objetivo_cpu)

    if cantidad_negativos > 0:
        acumulada = distribucion_negativos(np.bincount(ids_corpus, minlength=tamanio_vocabulario))
        modo = f"Muestreo negativo (|P_sel| = {cantidad_negativos})"
    else:
        modo = "Softmax completo sobre todo |V|"

    ruta_metricas = directorio_respaldos / f"metricas_w{tamanio_ventana}.csv"
    directorio_respaldos.mkdir(parents=True, exist_ok=True)
    if epoca_inicial == 0 or not ruta_metricas.exists():
        ruta_metricas.write_text("epoca,perdida,accuracy,segundos\n", encoding="utf-8")

    datos_resumen = {
        "Estado": estado,
        "Ruta del corpus": ruta_corpus,
        "Incluir puntuación y números": configuracion["incluir_puntuacion_y_numeros"],
        "Tokens del corpus": f"{len(tokens_corpus):,}",
        "Muestras por época": f"{total_muestras:,} (una actualización por muestra, sin lotes)",
        "Tamaño del vocabulario (|V|)": f"{tamanio_vocabulario:,}",
        "Dimensión del embedding (N)": dimension_embedding,
        "Dimensiones": f"W {tuple(W.shape)}, W' {tuple(W_prima.shape)}, h ({dimension_embedding}, 1), y ({tamanio_vocabulario}, 1)",
        "Ventana": f"{tamanio_ventana} palabras a cada lado (C = {2 * tamanio_ventana})",
        "Modo de salida": modo,
        "Tasa de aprendizaje (eta)": tasa_aprendizaje,
        "Épocas a entrenar": cantidad_epocas,
        "Semilla aleatoria": semilla_aleatoria,
        "Cómputo": "GPU (CuPy)" if USAR_CUPY else "CPU (NumPy)",
        "Métricas por época": str(ruta_metricas),
    }
    imprimir_resumen("CONFIGURACIÓN DEL ENTRENAMIENTO", datos_resumen)

    # 3. Bucle de entrenamiento: una actualización de W y W' por cada muestra
    epoca_final = epoca_inicial + cantidad_epocas
    for epoca in range(epoca_inicial + 1, epoca_final + 1):
        tiempo_inicio = time.time()
        orden = np.random.default_rng(semilla_aleatoria + epoca).permutation(total_muestras)
        if cantidad_negativos > 0:
            negativos_epoca = muestrear_negativos(objetivo, cantidad_negativos, acumulada)  # un P_sel por muestra

        # Acumuladores en el dispositivo: se sincronizan con la CPU solo para mostrar el progreso
        perdida_acumulada = xp.zeros((), dtype=xp.float64)
        aciertos = xp.zeros((), dtype=xp.int64)

        barra_progreso = tqdm(orden, desc=f"Epoca {epoca}/{epoca_final}", unit="muestra", mininterval=2.0)
        for procesadas, indice in enumerate(barra_progreso, start=1):
            indice = int(indice)
            if cantidad_negativos > 0:
                E, acierto = paso_muestreo_negativo(
                    contexto[indice], objetivo[indice], negativos_epoca[indice], W, W_prima, tasa_aprendizaje
                )
            else:
                E, acierto = paso_softmax_completo(contexto[indice], objetivo[indice], W, W_prima, tasa_aprendizaje)
            perdida_acumulada += E
            aciertos += acierto

            if procesadas % 5000 == 0:
                barra_progreso.set_postfix(
                    {"Perdida": f"{float(perdida_acumulada) / procesadas:.4f}", "Acc": f"{int(aciertos) / procesadas:.3f}"}
                )

        duracion_epoca = time.time() - tiempo_inicio
        historial_perdida.append(float(perdida_acumulada) / total_muestras)
        historial_accuracy.append(int(aciertos) / total_muestras)
        with open(ruta_metricas, "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca},{historial_perdida[-1]:.6f},{historial_accuracy[-1]:.6f},{duracion_epoca:.2f}\n")
        print(
            f"Epoca {epoca} finalizada. Perdida Promedio: {historial_perdida[-1]:.4f} | "
            f"Accuracy: {historial_accuracy[-1]:.4f} | Tiempo: {duracion_epoca:.2f}s",
            flush=True,
        )

        # Respaldo periodico y, siempre, el modelo de la ultima época
        if (hacer_respaldo and epoca % frecuencia_respaldo == 0) or epoca == epoca_final:
            modelo_temp = {
                "W": W,
                "W_prima": W_prima,
                "vocabulario_palabras": vocabulario_palabras,
                "epoca_actual": epoca,
                "historial_perdida": historial_perdida,
                "historial_accuracy": historial_accuracy,
                "configuracion": configuracion,
            }
            guardar_modelo(modelo_temp, directorio_respaldos / f"modelo_cbow_w{tamanio_ventana}_epoca_{epoca}.npz")

    duracion_total_segundos = time.time() - tiempo_inicio_total
    datos_resumen["Tiempo total de entrenamiento"] = f"{duracion_total_segundos:.2f} s"
    imprimir_resumen("RESUMEN FINAL DE CONFIGURACIÓN Y RESULTADOS", datos_resumen)

    return {
        "W": W,
        "W_prima": W_prima,
        "vocabulario_palabras": vocabulario_palabras,
        "epoca_actual": epoca_final,
        "historial_perdida": historial_perdida,
        "historial_accuracy": historial_accuracy,
        "configuracion": configuracion,
        "duracion_total_segundos": duracion_total_segundos,
    }
