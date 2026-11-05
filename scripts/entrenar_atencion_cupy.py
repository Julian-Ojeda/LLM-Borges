#!/usr/bin/env python3
"""
Entrena el CBOW con una capa de auto-atencion (codigo/modelo_atencion_cupy.py) en CuPy, por lotes y con Adam.

Parte de un CBOW ya entrenado (solo lectura): W y W' se cargan del respaldo y la atencion se inicializa para
que el modelo arranque siendo exactamente ese CBOW. Asi, cualquier mejora la aporta la atencion.
Para separar el efecto de la atencion del de "entrenar mas", con --congelar-atencion se corre el control:
mismo optimizador, mismos lotes y mismas épocas, pero la atencion queda fija en el promedio del contexto.

No modifica nada existente: todo se escribe en directorio_respaldos (distinto de la carpeta del respaldo
de partida y sin metricas previas) y nunca se sobrescribe un .npz. Para deshacer, se borra esa carpeta.

Archivos de salida (en directorio_respaldos):
    metricas_w{m}.csv     perdida y acierto de entrenamiento por época (mismo formato que entrenar_cuda.py)
    evaluacion_w{m}.csv   sobre 20.000 muestras fijas (las mismas que ajustar_puntuacion_cuda.py):
                          perdida_palabras, acc_palabras, acc_palabras_sin_signos, acc_signos, pred_signos,
                          masa_signos (peso de atencion puesto en signos del contexto), masa_nula (en la clave nula)
    atencion_w{m}.csv     peso medio de atencion por cabeza y posicion del contexto (-m..-1, +1..+m, nulo)
    modelo_cbow_w{m}_atencion_epoca_{k}.npz   W, W', vocabulario (compatible con cargar_modelo) y la atencion

Uso:
    python scripts/entrenar_atencion_cupy.py                                   # config/atencion.yaml
    python scripts/entrenar_atencion_cupy.py --congelar-atencion --directorio respaldos/atencion_control
    python scripts/entrenar_atencion_cupy.py --epocas 1 --max-muestras 50000 --directorio <carpeta>   # prueba
"""

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))
os.chdir(ruta_raiz)

from codigo.configuracion import cargar_configuracion
from codigo.tokenizador import tokenizar_corpus
from codigo.entrenador_cbow import preparar_muestras, imprimir_resumen, TOKEN_DESCONOCIDO
from codigo.modelo_cbow_cupy import cargar_modelo, USAR_CUPY, xp
from codigo.modelo_atencion_cupy import (
    PARAMETROS_ATENCION, inicializar_atencion, adelante, perdida_salida, atras, Adam, guardar_modelo_atencion,
)

MUESTRAS_EVALUACION = 20000
ESPACIO_MINIMO_BYTES = 500 * 1024**2


def es_signo(token: str) -> bool:
    """Un token es signo de puntuacion si no tiene ningun caracter alfanumerico (y no es <UNK>)."""
    return token != TOKEN_DESCONOCIDO and not any(caracter.isalnum() for caracter in str(token))


def evitar_suspension() -> None:
    """En Windows, pide que el equipo no se suspenda por inactividad mientras corre este proceso."""
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


def evaluar(W, W_prima, p, H, contexto, objetivo, indices, mascara_signos, penalizacion_signos) -> dict:
    """Metricas y estadisticas de atencion sobre una muestra fija, sin actualizar pesos."""
    C = contexto.shape[1]
    suma_E_pal = aciertos_pal = aciertos_pal_sin = aciertos_sig = predichos_sig = 0.0
    total_pal = total_sig = 0
    suma_a = xp.zeros((H, C + 1), dtype=xp.float64)
    masa_signos = 0.0
    for bloque in np.array_split(indices, 40):
        bloque = xp.asarray(bloque)
        ctx, o = contexto[bloque], objetivo[bloque]
        u, cache = adelante(W, W_prima, p, ctx, H)
        pred_sin = (u + penalizacion_signos).argmax(axis=1)
        E, _, pred = perdida_salida(u, o)
        sig = mascara_signos[o]
        pal = ~sig
        suma_E_pal += float((E * pal).sum())
        aciertos_pal += int(((pred == o) & pal).sum())
        aciertos_pal_sin += int(((pred_sin == o) & pal).sum())
        aciertos_sig += int(((pred == o) & sig).sum())
        predichos_sig += int(mascara_signos[pred].sum())
        total_pal += int(pal.sum())
        total_sig += int(sig.sum())
        a = cache["a"]
        suma_a += a.sum(axis=0)
        masa_signos += float((a[:, :, :C] * mascara_signos[ctx][:, None, :]).sum())
    n = len(indices)
    atencion_media = suma_a / n
    return {
        "perdida_palabras": suma_E_pal / max(total_pal, 1),
        "acc_palabras": aciertos_pal / max(total_pal, 1),
        "acc_palabras_sin_signos": aciertos_pal_sin / max(total_pal, 1),
        "acc_signos": aciertos_sig / max(total_sig, 1),
        "pred_signos": predichos_sig / n,
        "masa_signos": masa_signos / (n * H),
        "masa_nula": float(atencion_media[:, C].mean()),
        "atencion_media": atencion_media,
    }


def entrenar(configuracion: dict, max_muestras: int | None = None) -> None:
    if not USAR_CUPY:
        raise RuntimeError("Este script necesita una GPU NVIDIA con CuPy.")

    ruta_checkpoint = Path(configuracion["ruta_checkpoint"])
    directorio = Path(configuracion["directorio_respaldos"])
    m_ventana = configuracion["tamanio_ventana"]
    H = configuracion["cantidad_cabezas"]
    B = configuracion["tamanio_lote"]
    cantidad_epocas = configuracion["cantidad_epocas"]
    congelar_atencion = configuracion["congelar_atencion"]
    semilla = configuracion["semilla_aleatoria"]

    # Protecciones para no pisar nada existente
    if not ruta_checkpoint.exists():
        raise FileNotFoundError(f"No existe el respaldo de partida: '{ruta_checkpoint}'")
    if directorio.resolve() == ruta_checkpoint.resolve().parent:
        raise ValueError(f"directorio_respaldos ('{directorio}') debe ser distinto de la carpeta del respaldo de partida.")
    rutas = {n: directorio / f"{n}_w{m_ventana}.csv" for n in ("metricas", "evaluacion", "atencion")}
    if any(r.exists() for r in rutas.values()):
        raise FileExistsError(f"'{directorio}' ya tiene metricas de otro entrenamiento. Use otra carpeta o borre esa.")
    directorio.mkdir(parents=True, exist_ok=True)

    evitar_suspension()

    modelo = cargar_modelo(ruta_checkpoint)
    W = xp.ascontiguousarray(modelo["W"], dtype=xp.float32)
    W_prima = xp.ascontiguousarray(modelo["W_prima"], dtype=xp.float32)
    vocabulario = modelo["vocabulario_palabras"]
    configuracion_guardada = modelo.get("configuracion", {})
    incluir_puntuacion = configuracion_guardada.get("incluir_puntuacion_y_numeros", True)
    if configuracion_guardada.get("tamanio_ventana", m_ventana) != m_ventana:
        raise ValueError("tamanio_ventana no coincide con el del respaldo de partida.")

    tokens = tokenizar_corpus(configuracion["ruta_corpus"], incluir_puntuacion)
    mapeo = {palabra: indice for indice, palabra in enumerate(vocabulario)}
    _, contexto_cpu, objetivo_cpu = preparar_muestras(tokens, mapeo, m_ventana)
    contexto = xp.ascontiguousarray(xp.asarray(contexto_cpu, dtype=xp.int32))
    objetivo = xp.ascontiguousarray(xp.asarray(objetivo_cpu, dtype=xp.int32))
    C = contexto.shape[1]

    signos_cpu = np.array([es_signo(p) for p in vocabulario])
    mascara_signos = xp.asarray(signos_cpu)
    penalizacion_signos = xp.where(mascara_signos, xp.float32(-1e30), xp.float32(0))[None, :]
    if configuracion["excluir_signos_objetivo"]:
        indices_entrenamiento = np.flatnonzero(~signos_cpu[objetivo_cpu])
    else:
        indices_entrenamiento = np.arange(len(objetivo_cpu))
    if max_muestras is not None:
        indices_entrenamiento = indices_entrenamiento[:max_muestras]
    indices_evaluacion = np.random.default_rng(0).choice(len(objetivo_cpu), MUESTRAS_EVALUACION, replace=False)

    p = inicializar_atencion(W.shape[1], C, H, semilla, modulo=xp)
    parametros = {"W": W, "W_prima": W_prima, **p}
    tasas = {"W": configuracion["tasa_embeddings"], "W_prima": configuracion["tasa_salida"]}
    tasas.update({k: 0.0 if congelar_atencion else configuracion["tasa_atencion"] for k in PARAMETROS_ATENCION})
    adam = Adam(parametros, tasas)
    entrenar_W = tasas["W"] > 0

    total = len(indices_entrenamiento)
    pasos_por_epoca = math.ceil(total / B)
    pasos_totales = pasos_por_epoca * cantidad_epocas

    imprimir_resumen("CBOW + AUTO-ATENCIÓN (CuPy, gradiente a mano)", {
        "Respaldo de partida": f"{ruta_checkpoint} (solo lectura)",
        "Directorio de salida": str(directorio),
        "Atención": "CONGELADA (control: promedio del contexto)" if congelar_atencion
                    else f"{H} cabeza(s) de dimensión {W.shape[1] // H} + clave nula",
        "Contexto": f"{m_ventana} palabras a cada lado (C = {C}), con embeddings posicionales",
        "Muestras por época": f"{total:,}" + (" (objetivo = palabra)" if configuracion["excluir_signos_objetivo"] else ""),
        "Tamaño de lote": f"{B} ({pasos_por_epoca:,} pasos por época)",
        "Optimizador": "Adam (beta1 0.85, beta2 0.99), decaimiento lineal" if configuracion["decaimiento_lineal"]
                       else "Adam (beta1 0.85, beta2 0.99)",
        "Tasas": ", ".join(f"{k} {v:g}" for k, v in (("atención", 0.0 if congelar_atencion else configuracion["tasa_atencion"]),
                                                     ("W'", tasas["W_prima"]), ("W", tasas["W"]))),
        "Épocas": cantidad_epocas,
    })

    rutas["metricas"].write_text("epoca,perdida,accuracy,segundos\n", encoding="utf-8")
    rutas["evaluacion"].write_text("epoca,perdida_palabras,acc_palabras,acc_palabras_sin_signos,acc_signos,"
                                   "pred_signos,masa_signos,masa_nula\n", encoding="utf-8")
    posiciones = [f"p{d:+d}" for d in list(range(-m_ventana, 0)) + list(range(1, m_ventana + 1))]
    rutas["atencion"].write_text("epoca,cabeza," + ",".join(posiciones) + ",nulo\n", encoding="utf-8")

    def registrar_evaluacion(epoca: int) -> np.ndarray:
        r = evaluar(W, W_prima, p, H, contexto, objetivo, indices_evaluacion, mascara_signos, penalizacion_signos)
        claves = ("perdida_palabras", "acc_palabras", "acc_palabras_sin_signos", "acc_signos",
                  "pred_signos", "masa_signos", "masa_nula")
        with open(rutas["evaluacion"], "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca}," + ",".join(f"{r[k]:.6f}" for k in claves) + "\n")
        atencion_media = xp.asnumpy(r["atencion_media"])
        with open(rutas["atencion"], "a", encoding="utf-8") as archivo:
            for cabeza in range(H):
                archivo.write(f"{epoca},{cabeza}," + ",".join(f"{v:.6f}" for v in atencion_media[cabeza]) + "\n")
        print(f"Evaluación época {epoca}: " + " | ".join(f"{k} {r[k]:.4f}" for k in claves), flush=True)
        print("  atención media por posición: " + " ".join(
            f"{n} {v:.3f}" for n, v in zip(posiciones + ["nulo"], atencion_media.mean(axis=0))), flush=True)
        return atencion_media

    registrar_evaluacion(0)

    historial_perdida, historial_accuracy = [], []
    paso = 0
    for epoca in range(1, cantidad_epocas + 1):
        inicio = time.time()
        orden = xp.asarray(np.random.default_rng(semilla + epoca).permutation(indices_entrenamiento))
        suma_E = xp.zeros((), dtype=xp.float64)
        aciertos = xp.zeros((), dtype=xp.int64)
        barra = tqdm(range(0, total, B), desc=f"Epoca {epoca}/{cantidad_epocas}", unit="lote", mininterval=30.0)
        for procesados, desde in enumerate(barra, start=1):
            lote = orden[desde:desde + B]
            o = objetivo[lote]
            u, cache = adelante(W, W_prima, p, contexto[lote], H)
            E, e, pred = perdida_salida(u, o)
            suma_E += E.sum()
            aciertos += (pred == o).sum()
            gradientes = atras(e, W, W_prima, p, cache, H, gradiente_W=entrenar_W)
            factor = 1.0 - paso / pasos_totales if configuracion["decaimiento_lineal"] else 1.0
            adam.paso(parametros, gradientes, factor)
            paso += 1
            if procesados % 500 == 0:
                vistas = min(procesados * B, total)
                barra.set_postfix({"Perdida": f"{float(suma_E) / vistas:.4f}", "Acc": f"{int(aciertos) / vistas:.3f}"})

        duracion = time.time() - inicio
        historial_perdida.append(float(suma_E) / total)
        historial_accuracy.append(int(aciertos) / total)
        with open(rutas["metricas"], "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca},{historial_perdida[-1]:.6f},{historial_accuracy[-1]:.6f},{duracion:.2f}\n")
        print(f"Epoca {epoca} finalizada. Perdida: {historial_perdida[-1]:.4f} | "
              f"Accuracy: {historial_accuracy[-1]:.4f} | Tiempo: {duracion:.2f}s", flush=True)
        atencion_media = registrar_evaluacion(epoca)

        if epoca % configuracion["frecuencia_respaldo"] == 0 or epoca == cantidad_epocas:
            destino = directorio / f"modelo_cbow_w{m_ventana}_atencion_epoca_{epoca}.npz"
            libre = shutil.disk_usage(directorio).free
            if destino.exists():
                print(f"AVISO: '{destino}' ya existe; no se sobrescribe.", flush=True)
            elif libre < ESPACIO_MINIMO_BYTES:
                print(f"AVISO: quedan {libre / 1024**2:.0f} MB libres; se omite el respaldo de la época {epoca}.", flush=True)
            else:
                guardar_modelo_atencion({
                    "W": W, "W_prima": W_prima, "vocabulario_palabras": vocabulario, "epoca_actual": epoca,
                    "historial_perdida": historial_perdida, "historial_accuracy": historial_accuracy,
                    "cantidad_cabezas": H, "atencion": p, "atencion_media": atencion_media,
                    "configuracion": {**configuracion, "incluir_puntuacion_y_numeros": incluir_puntuacion,
                                      "respaldo_de_partida": str(ruta_checkpoint)},
                }, destino)


def main() -> None:
    parser = argparse.ArgumentParser(description="CBOW con auto-atención en CuPy (gradiente a mano).")
    parser.add_argument("--config", default="config/atencion.yaml")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--cabezas", type=int, default=None)
    parser.add_argument("--directorio", default=None, help="Sobrescribe directorio_respaldos.")
    parser.add_argument("--congelar-atencion", action="store_true", help="Control: la atención queda en el promedio.")
    parser.add_argument("--max-muestras", type=int, default=None, help="Limita las muestras por época (prueba corta).")
    args = parser.parse_args()

    configuracion = cargar_configuracion(args.config)
    if args.epocas is not None:
        configuracion["cantidad_epocas"] = args.epocas
    if args.cabezas is not None:
        configuracion["cantidad_cabezas"] = args.cabezas
    if args.directorio is not None:
        configuracion["directorio_respaldos"] = args.directorio
    if args.congelar_atencion:
        configuracion["congelar_atencion"] = True
    entrenar(configuracion, args.max_muestras)
    print("ENTRENAMIENTO FINALIZADO", flush=True)


if __name__ == "__main__":
    main()
