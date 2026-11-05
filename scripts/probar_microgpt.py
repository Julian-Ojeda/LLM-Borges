#!/usr/bin/env python3
"""
Evalua el microgpt de palabras (scripts/microgpt_palabras.py) a partir de los pesos guardados.

Hace la MISMA cuenta que la funcion gpt() de microgpt (rmsnorm, atencion causal multi-cabeza, MLP con ReLU,
lm_head), pero con matrices de NumPy en lugar de Values escalares: no hace falta el gradiente, y asi es
cientos de veces mas rapido. (Se verifico que da la misma perdida que el entrenamiento.)

Muestra:
    - perdida y contadores de ranking (top-5, top-10, top-20) sobre las oraciones de validacion (ultimo 10%);
    - las tablas de la presentacion (top 5 con % y generacion continua de 5 tokens).
Las palabras de una frase que no estan en el vocabulario de la fraccion se descartan (y se avisa).

Uso:
    python scripts/probar_microgpt.py respaldos/microgpt/<carpeta>
"""

import argparse
import ast
import re
import sys
from pathlib import Path
import numpy as np

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))

from codigo.tokenizador import tokenizar_corpus
from probar_pmc import FRASES

CORTES = (1, 5, 10, 20)


def rmsnorm(x):
    return x / np.sqrt((x * x).mean(axis=-1, keepdims=True) + 1e-5)


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def gpt_secuencia(p: dict, ids: list[int], n_layer: int, n_head: int) -> np.ndarray:
    """Logits (T x vocab) para cada posicion de la secuencia: equivale a llamar a gpt() token por token."""
    T = len(ids)
    x = rmsnorm(p["wte"][ids] + p["wpe"][:T])                    # (T, d)
    d = x.shape[1]
    dk = d // n_head
    causal = np.triu(np.ones((T, T), dtype=bool), 1)             # cada posicion solo ve las anteriores
    for li in range(n_layer):
        residual = x
        x = rmsnorm(x)
        q, k, v = (x @ p[f"layer{li}.attn_w{c}"].T for c in "qkv")   # linear(x, w) = w @ x
        cabezas = []
        for h in range(n_head):
            s = slice(h * dk, (h + 1) * dk)
            puntajes = q[:, s] @ k[:, s].T / dk**0.5
            puntajes[causal] = -np.inf
            cabezas.append(softmax(puntajes) @ v[:, s])
        x = np.concatenate(cabezas, axis=1) @ p[f"layer{li}.attn_wo"].T + residual
        residual = x
        x = np.maximum(rmsnorm(x) @ p[f"layer{li}.mlp_fc1"].T, 0) @ p[f"layer{li}.mlp_fc2"].T + residual
    return x @ p["lm_head"].T


def docs_validacion(hp: dict, palabra_a_id: dict, BOS: int) -> list[list[int]]:
    """Rearma los tramos de validacion igual que scripts/microgpt_palabras.py."""
    tokens = tokenizar_corpus(ruta_raiz / "datos/corpus_ApAvAu.txt", True)
    tokens = tokens[: int(len(tokens) * hp["fraccion"])]
    oraciones, actual = [], [BOS]
    for w in tokens:
        actual.append(palabra_a_id[w])
        if w in (".", "?", "!"):
            oraciones.append(actual + [BOS])
            actual = [BOS]
    N = hp["block_size"]
    return [o[i: i + N + 1] for o in oraciones[int(len(oraciones) * 0.9):] for i in range(0, len(o) - 1, N)]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evalua el microgpt de palabras.")
    parser.add_argument("directorio")
    args = parser.parse_args()
    directorio = Path(args.directorio)

    datos = np.load(directorio / "microgpt_palabras.npz")      # sin pickle
    hp = ast.literal_eval(str(datos["hiperparametros"]))
    vocabulario = [str(w) for w in datos["vocabulario"]]
    BOS = len(vocabulario) - 1
    palabra_a_id = {w: i for i, w in enumerate(vocabulario[:-1])}
    p = {c: datos[c] for c in datos.files if c not in ("adam_m", "adam_v", "paso", "vocabulario", "hiperparametros")}
    L, H, N = hp["n_layer"], hp["n_head"], hp["block_size"]
    print(f"Pesos: {directorio} (paso {int(datos['paso'])} de {hp['num_steps']}) | {hp}")

    # 1. Validacion: perdida y contadores de ranking
    suma, n, aciertos = 0.0, 0, {c: 0 for c in CORTES}
    for doc in docs_validacion(hp, palabra_a_id, BOS):
        logits = gpt_secuencia(p, doc[:-1], L, H)
        objetivos = np.array(doc[1:])
        correcto = logits[np.arange(len(objetivos)), objetivos]
        suma += -(np.log(softmax(logits)[np.arange(len(objetivos)), objetivos])).sum()
        puestos = (logits >= correcto[:, None]).sum(axis=1)
        for c in CORTES:
            aciertos[c] += int((puestos <= c).sum())
        n += len(objetivos)
    resumen = (f"Validación ({n} predicciones): pérdida {suma / n:.4f} | top-5 {aciertos[5] / n:.2%} | "
               f"top-10 {aciertos[10] / n:.2%} | top-20 {aciertos[20] / n:.2%} | exacta {aciertos[1] / n:.2%}")
    print(resumen)

    # 2. Tablas como en la presentacion: el contexto empieza con BOS y entran a lo sumo N tokens
    top = ["| Frase | Tipo | Opción 1 | Opción 2 | Opción 3 | Opción 4 | Opción 5 |", "|---|---|---|---|---|---|---|"]
    generacion = ["| Frase | Tipo | Salida (ansiosa, 5 tokens) |", "|---|---|---|"]
    for frase, tipo in FRASES:
        tokens = re.findall(r"\w+|[^\w\s]", frase.lower())
        fuera = [t for t in tokens if t not in palabra_a_id]
        ids = [palabra_a_id[t] for t in tokens if t in palabra_a_id]
        nota = f" (sin: {', '.join(fuera)})" if fuera else ""
        probs = softmax(gpt_secuencia(p, ([BOS] + ids)[-N:], L, H)[-1])
        mejores = np.argsort(-probs)[:5]
        top.append(f"| {frase}{nota} | {tipo} | " + " | ".join(f"{vocabulario[i]} ({probs[i]:.2%})" for i in mejores) + " |")
        secuencia = [BOS] + ids
        for _ in range(5):
            secuencia.append(int(gpt_secuencia(p, secuencia[-N:], L, H)[-1].argmax()))
        generacion.append(f"| {frase} | {tipo} | {' '.join(vocabulario[i] for i in secuencia[-5:])} |")

    texto = (f"{resumen}\n\n## Top 5 predicciones\n\n" + "\n".join(top)
             + "\n\n## Generación continua\n\n" + "\n".join(generacion) + "\n")
    print("\n" + texto)
    (directorio / "predicciones.md").write_text(texto, encoding="utf-8")
    print(f"Guardado en: {directorio / 'predicciones.md'}")


if __name__ == "__main__":
    main()
