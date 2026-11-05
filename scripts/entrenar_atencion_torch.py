#!/usr/bin/env python3
"""
Entrena el CBOW con una capa de auto-atencion (codigo/modelo_atencion_torch.py) en PyTorch, por lotes y con Adam.

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
    python scripts/entrenar_atencion_torch.py                                  # config/atencion.yaml
    python scripts/entrenar_atencion_torch.py --congelar-atencion --directorio respaldos/atencion_control
    python scripts/entrenar_atencion_torch.py --epocas 1 --max-muestras 50000 --directorio <carpeta>   # prueba

Diferencias con el entrenamiento del CBOW (entrenar_cuda.py):
    - Por lotes de B muestras (tamanio_lote) en lugar de muestra a muestra: un paso de Adam por lote.
    - Adam en lugar de descenso por el gradiente simple: cada parametro se mueve segun un promedio de sus
      gradientes recientes, normalizado por su escala (igual que en microgpt.py).
    - El gradiente no esta escrito a mano: lo calcula perdida.backward().
"""

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))
os.chdir(ruta_raiz)     # las rutas relativas (config/, datos/, respaldos/) se resuelven desde la raiz del proyecto

from codigo.configuracion import cargar_configuracion
from codigo.tokenizador import tokenizar_corpus
from codigo.entrenador_cbow import preparar_muestras, imprimir_resumen, TOKEN_DESCONOCIDO
from codigo.modelo_atencion_torch import CBOWConAtencion, PARAMETROS_ATENCION, cargar_respaldo, guardar_respaldo

MUESTRAS_EVALUACION = 20000                 # tamaño de la muestra fija con la que se evalua cada época
ESPACIO_MINIMO_BYTES = 500 * 1024**2        # si queda menos disco libre que esto (500 MB), no se guarda el .npz


def es_signo(token: str) -> bool:
    """Un token es signo de puntuacion si no tiene ningun caracter alfanumerico (y no es <UNK>)."""
    return token != TOKEN_DESCONOCIDO and not any(caracter.isalnum() for caracter in str(token))


def evitar_suspension() -> None:
    """En Windows, pide que el equipo no se suspenda por inactividad mientras corre este proceso."""
    if os.name == "nt":
        import ctypes
        # ES_CONTINUOUS | ES_SYSTEM_REQUIRED. No evita que se apague si se agota la bateria
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)


@torch.no_grad()    # en la evaluacion no se calculan gradientes: menos memoria y mas rapido
def evaluar(modelo, contexto, objetivo, indices, es_signo_vocab) -> dict:
    """Metricas y estadisticas de atencion sobre una muestra fija, sin actualizar pesos."""
    C = contexto.shape[1]
    suma_E_pal = aciertos_pal = aciertos_pal_sin = aciertos_sig = predichos_sig = masa_signos = 0.0
    total_pal = total_sig = 0
    suma_a = 0.0
    # De a 500 muestras: u de las 20.000 juntas (20.000 x 53.409 floats) no entra en la GPU
    for bloque in torch.tensor(indices, device=contexto.device).split(500):
        ctx, o = contexto[bloque], objetivo[bloque]
        u, a = modelo(ctx)
        E = F.cross_entropy(u, o, reduction="none")                     # -log y_O de cada muestra
        pred = u.argmax(dim=1)                                          # palabra mas probable
        pred_sin = u.masked_fill(es_signo_vocab, float("-inf")).argmax(dim=1)   # sin dejar elegir un signo
        sig = es_signo_vocab[o]                                         # la palabra central es un signo?
        pal = ~sig
        suma_E_pal += E[pal].sum().item()                               # .item(): tensor de 1 valor -> numero de Python
        aciertos_pal += ((pred == o) & pal).sum().item()
        aciertos_pal_sin += ((pred_sin == o) & pal).sum().item()
        aciertos_sig += ((pred == o) & sig).sum().item()
        predichos_sig += es_signo_vocab[pred].sum().item()
        total_pal += pal.sum().item()
        total_sig += sig.sum().item()
        suma_a = suma_a + a.sum(dim=0)                                  # (H, C+1) suma de los pesos de atencion
        # Peso que la atencion pone en posiciones del contexto ocupadas por signos
        masa_signos += (a[:, :, :C] * es_signo_vocab[ctx][:, None, :]).sum().item()
    n, H = len(indices), modelo.H
    atencion_media = (suma_a / n).cpu().numpy()
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
    if not torch.cuda.is_available():
        raise RuntimeError("Este script necesita una GPU NVIDIA con PyTorch CUDA.")
    dispositivo = torch.device("cuda")

    # Parametros (ver config/atencion.yaml para el significado de cada uno)
    ruta_checkpoint = Path(configuracion["ruta_checkpoint"])
    directorio = Path(configuracion["directorio_respaldos"])
    m_ventana = configuracion["tamanio_ventana"]
    H = configuracion["cantidad_cabezas"]
    B = configuracion["tamanio_lote"]
    cantidad_epocas = configuracion["cantidad_epocas"]
    congelar_atencion = configuracion["congelar_atencion"]
    semilla = configuracion["semilla_aleatoria"]
    torch.manual_seed(semilla)

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

    # 1. CBOW de partida y muestras (contexto, palabra central)
    respaldo = cargar_respaldo(ruta_checkpoint)
    vocabulario = respaldo["vocabulario_palabras"]
    # Se tokeniza igual que cuando se armo el vocabulario del respaldo (con o sin puntuacion)
    incluir_puntuacion = respaldo["configuracion"].get("incluir_puntuacion_y_numeros", True)
    if respaldo["configuracion"].get("tamanio_ventana", m_ventana) != m_ventana:
        raise ValueError("tamanio_ventana no coincide con el del respaldo de partida.")

    tokens = tokenizar_corpus(configuracion["ruta_corpus"], incluir_puntuacion)
    mapeo = {palabra: indice for indice, palabra in enumerate(vocabulario)}
    _, contexto_np, objetivo_np = preparar_muestras(tokens, mapeo, m_ventana)
    # Todas las muestras se cargan una sola vez en la GPU (long = int64, el tipo que usa PyTorch para indices)
    contexto = torch.tensor(contexto_np, dtype=torch.long, device=dispositivo)    # (T, C)
    objetivo = torch.tensor(objetivo_np, dtype=torch.long, device=dispositivo)    # (T,)
    C = contexto.shape[1]

    es_signo_np = np.array([es_signo(p) for p in vocabulario])     # (|V|,) True en los signos
    es_signo_vocab = torch.tensor(es_signo_np, device=dispositivo)
    if configuracion["excluir_signos_objetivo"]:
        # Igual que en el ajuste: se entrena solo con las muestras cuya palabra central no es un signo
        indices_entrenamiento = np.flatnonzero(~es_signo_np[objetivo_np])
    else:
        indices_entrenamiento = np.arange(len(objetivo_np))
    if max_muestras is not None:
        indices_entrenamiento = indices_entrenamiento[:max_muestras]
    # Muestra de evaluacion fija (semilla 0): la misma que en ajustar_puntuacion_cuda.py, para comparar
    indices_evaluacion = np.random.default_rng(0).choice(len(objetivo_np), MUESTRAS_EVALUACION, replace=False)

    # 2. Modelo y optimizador: Adam con una tasa por grupo de parametros; tasa 0 = congelado
    modelo = CBOWConAtencion(respaldo["W"], respaldo["W_prima"], C, H, semilla).to(dispositivo)
    tasa_atencion = 0.0 if congelar_atencion else configuracion["tasa_atencion"]
    grupos = [
        {"params": [modelo.W], "lr": configuracion["tasa_embeddings"]},
        {"params": [modelo.W_prima], "lr": configuracion["tasa_salida"]},
        {"params": [getattr(modelo, n) for n in PARAMETROS_ATENCION], "lr": tasa_atencion},
    ]
    # requires_grad = False: el autograd no calcula el gradiente de ese parametro y queda fijo
    for grupo in grupos:
        for parametro in grupo["params"]:
            parametro.requires_grad_(grupo["lr"] > 0)
    # Mismos beta1, beta2 y eps que microgpt.py; al optimizador solo van los grupos que se entrenan
    optimizador = torch.optim.Adam([g for g in grupos if g["lr"] > 0], betas=(0.85, 0.99), eps=1e-8)

    total = len(indices_entrenamiento)
    pasos_por_epoca = math.ceil(total / B)
    pasos_totales = pasos_por_epoca * cantidad_epocas
    # La tasa baja linealmente hasta 0 a lo largo de todo el entrenamiento (como en microgpt.py).
    # LambdaLR multiplica la tasa de cada grupo por lo que devuelve la funcion en cada paso
    planificador = torch.optim.lr_scheduler.LambdaLR(
        optimizador, lambda paso: 1.0 - paso / pasos_totales if configuracion["decaimiento_lineal"] else 1.0
    )

    imprimir_resumen("CBOW + AUTO-ATENCIÓN (PyTorch)", {
        "Respaldo de partida": f"{ruta_checkpoint} (solo lectura)",
        "Directorio de salida": str(directorio),
        "Atención": "CONGELADA (control: promedio del contexto)" if congelar_atencion
                    else f"{H} cabeza(s) de dimensión {modelo.d} + clave nula",
        "Contexto": f"{m_ventana} palabras a cada lado (C = {C}), con embeddings posicionales",
        "Muestras por época": f"{total:,}" + (" (objetivo = palabra)" if configuracion["excluir_signos_objetivo"] else ""),
        "Tamaño de lote": f"{B} ({pasos_por_epoca:,} pasos por época)",
        "Optimizador": "Adam (beta1 0.85, beta2 0.99)" + (", decaimiento lineal" if configuracion["decaimiento_lineal"] else ""),
        "Tasas": f"atención {tasa_atencion:g}, W' {configuracion['tasa_salida']:g}, W {configuracion['tasa_embeddings']:g}",
        "Épocas": cantidad_epocas,
    })

    # Encabezados de los CSV
    rutas["metricas"].write_text("epoca,perdida,accuracy,segundos\n", encoding="utf-8")
    claves = ("perdida_palabras", "acc_palabras", "acc_palabras_sin_signos", "acc_signos",
              "pred_signos", "masa_signos", "masa_nula")
    rutas["evaluacion"].write_text("epoca," + ",".join(claves) + "\n", encoding="utf-8")
    posiciones = [f"p{d:+d}" for d in list(range(-m_ventana, 0)) + list(range(1, m_ventana + 1))]   # p-4 .. p+4
    rutas["atencion"].write_text("epoca,cabeza," + ",".join(posiciones) + ",nulo\n", encoding="utf-8")

    def registrar_evaluacion(epoca: int) -> np.ndarray:
        """Evalua, agrega una fila a evaluacion_w{m}.csv y H filas a atencion_w{m}.csv, e imprime el resumen."""
        modelo.eval()       # modo evaluacion (en este modelo no cambia nada, es la convencion de PyTorch)
        r = evaluar(modelo, contexto, objetivo, indices_evaluacion, es_signo_vocab)
        modelo.train()
        with open(rutas["evaluacion"], "a", encoding="utf-8") as archivo:
            archivo.write(f"{epoca}," + ",".join(f"{r[k]:.6f}" for k in claves) + "\n")
        with open(rutas["atencion"], "a", encoding="utf-8") as archivo:
            for cabeza in range(H):
                archivo.write(f"{epoca},{cabeza}," + ",".join(f"{v:.6f}" for v in r["atencion_media"][cabeza]) + "\n")
        print(f"Evaluación época {epoca}: " + " | ".join(f"{k} {r[k]:.4f}" for k in claves), flush=True)
        print("  atención media por posición: " + " ".join(
            f"{n} {v:.3f}" for n, v in zip(posiciones + ["nulo"], r["atencion_media"].mean(axis=0))), flush=True)
        return r["atencion_media"]

    registrar_evaluacion(0)     # época 0 = antes de entrenar: tiene que dar lo mismo que el CBOW de partida

    # 3. Entrenamiento por lotes
    historial_perdida, historial_accuracy = [], []
    for epoca in range(1, cantidad_epocas + 1):
        inicio = time.time()
        # Orden aleatorio de las muestras, distinto en cada época pero reproducible
        orden = torch.tensor(np.random.default_rng(semilla + epoca).permutation(indices_entrenamiento), device=dispositivo)
        # Acumuladores en la GPU (leerlos en cada paso obligaria a esperar a la GPU)
        suma_E = torch.zeros((), dtype=torch.float64, device=dispositivo)
        aciertos = torch.zeros((), dtype=torch.long, device=dispositivo)
        # orden.split(B): trozos de B indices (el ultimo puede ser mas chico); cada trozo es un lote
        barra = tqdm(orden.split(B), desc=f"Epoca {epoca}/{cantidad_epocas}", unit="lote", mininterval=30.0)
        for procesados, lote in enumerate(barra, start=1):
            o = objetivo[lote]                        # (B,) palabras centrales del lote
            u, _ = modelo(contexto[lote])             # (B, |V|) pasada hacia adelante
            perdida = F.cross_entropy(u, o)          # promedio de E = -log y_O en el lote

            optimizador.zero_grad(set_to_none=True)   # borra los gradientes del paso anterior (si no, se acumulan)
            perdida.backward()                        # autograd: gradiente de la perdida respecto de cada parametro
            optimizador.step()                        # Adam actualiza los parametros con esos gradientes
            planificador.step()                       # baja la tasa para el proximo paso

            # .detach(): para las metricas no hace falta (ni conviene) seguir el grafo del autograd
            suma_E += perdida.detach() * len(lote)
            aciertos += (u.detach().argmax(dim=1) == o).sum()
            if procesados % 500 == 0:
                vistas = min(procesados * B, total)
                barra.set_postfix({"Perdida": f"{suma_E.item() / vistas:.4f}", "Acc": f"{aciertos.item() / vistas:.3f}"})

        # Fin de la época: metricas de entrenamiento, evaluacion y respaldo
        duracion = time.time() - inicio
        historial_perdida.append(suma_E.item() / total)
        historial_accuracy.append(aciertos.item() / total)
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
                guardar_respaldo(modelo, {
                    "vocabulario_palabras": vocabulario, "epoca_actual": epoca,
                    "historial_perdida": historial_perdida, "historial_accuracy": historial_accuracy,
                    "atencion_media": atencion_media,
                    "configuracion": {**configuracion, "incluir_puntuacion_y_numeros": incluir_puntuacion,
                                      "respaldo_de_partida": str(ruta_checkpoint)},
                }, destino)


def main() -> None:
    parser = argparse.ArgumentParser(description="CBOW con auto-atención en PyTorch.")
    parser.add_argument("--config", default="config/atencion.yaml")
    parser.add_argument("--epocas", type=int, default=None)
    parser.add_argument("--cabezas", type=int, default=None)
    parser.add_argument("--directorio", default=None, help="Sobrescribe directorio_respaldos.")
    parser.add_argument("--congelar-atencion", action="store_true", help="Control: la atención queda en el promedio.")
    parser.add_argument("--max-muestras", type=int, default=None, help="Limita las muestras por época (prueba corta).")
    args = parser.parse_args()

    # YAML primero; los argumentos que se pasen por consola pisan esos valores
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
