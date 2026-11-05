#!/usr/bin/env python3
"""
Prueba a mano del PMC (consignas 1 y 2): las mismas tablas que la presentacion del TP2.
    - Top 5 predicciones (con su %) para frases del corpus y fuera del corpus.
    - Generacion continua: 5 tokens eligiendo siempre el mas probable (ansiosa).
En la variante "vocabulario" el % es la probabilidad de la softmax; en la "vector" es el coseno entre el
vector predicho y el de cada palabra (igual que en el TP).

El contexto son los ultimos k tokens de la frase; si la frase es mas corta se rellena con puntos a la izquierda.
Corre en CPU. Tambien guarda las tablas en <carpeta>/predicciones.md.

Uso:
    python scripts/probar_pmc.py respaldos/pmc/vocabulario_k8
    python scripts/probar_pmc.py respaldos/pmc/vocabulario_k8 --frase "volvió a su casa y"
"""

import argparse
import ast
import re
import sys
from pathlib import Path
import numpy as np
import torch

ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))

from codigo.modelo_pmc_torch import ModeloLenguajePMC, cargar_respaldo

# Las frases de la presentacion del TP2 (corpus / fuera del corpus) y las dos pruebas de orden
FRASES = [
    ("enanos tienen una especie de", "Corpus"),
    ("llegaba a él por entre", "Corpus"),
    ("el viejo escritor pensaba en", "Fuera del corpus"),
    ("ella caminaba lentamente hacia la", "Fuera del corpus"),
    ("enanos tienen una especie de sexto sentido que les permite", "Corpus"),
    ("llegaba a él por entre grupos de lilas y extensos", "Corpus"),
    ("el viejo escritor pensaba en publicar su primer libro autobiográfico", "Fuera del corpus"),
    ("ella caminaba lentamente hacia la puerta principal buscando una salida", "Fuera del corpus"),
    ("los enanos tienen una especie", "Corpus"),
    ("especie una tienen enanos los", "Orden invertido"),
]


def tokenizar(texto: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", texto.lower())     # mismo patron que codigo/tokenizador.py


def cargar_pmc(directorio: Path) -> tuple[ModeloLenguajePMC, object, dict]:
    """Rearma el PMC: W del respaldo del CBOW + pesos guardados (sin pickle) del mejor modelo."""
    ruta = directorio / "modelo_pmc_mejor.npz"
    datos = np.load(ruta)
    configuracion = ast.literal_eval(str(datos["configuracion_json"]))
    respaldo = cargar_respaldo(ruta_raiz / configuracion["ruta_checkpoint"])
    modelo = ModeloLenguajePMC(respaldo["W"], configuracion["tamanio_contexto"], configuracion["modo"],
                               configuracion["capas_ocultas"], configuracion["normalizacion"],
                               configuracion["activacion"], configuracion["activacion_salida"])
    modelo.pmc.load_state_dict({c[len("pmc."):]: torch.tensor(datos[c]) for c in datos.files if c.startswith("pmc.")})
    modelo.eval()
    print(f"Modelo: {ruta} (época {int(datos['epoca'])}, modo {configuracion['modo']}, k = {modelo.k})")
    return modelo, respaldo["vocabulario_palabras"], configuracion


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser(description="Tablas de prediccion del PMC.")
    parser.add_argument("directorio", help="Carpeta de la corrida (con modelo_pmc_mejor.npz).")
    parser.add_argument("--frase", action="append", help="Frase propia (se puede repetir).")
    args = parser.parse_args()

    directorio = Path(args.directorio)
    modelo, vocabulario, _ = cargar_pmc(directorio)
    mapeo = {str(p): i for i, p in enumerate(vocabulario)}
    frases = [(f, "Propia") for f in args.frase] if args.frase else FRASES

    def contexto_de(ids):
        ids = [mapeo["."]] * max(0, modelo.k - len(ids)) + ids
        return torch.tensor([ids[-modelo.k:]])

    top = ["| Frase | Tipo | Opción 1 | Opción 2 | Opción 3 | Opción 4 | Opción 5 |", "|---|---|---|---|---|---|---|"]
    generacion = ["| Frase | Tipo | Salida (ansiosa, 5 tokens) |", "|---|---|---|"]
    for frase, tipo in frases:
        ids = [mapeo.get(t, 0) for t in tokenizar(frase)]
        u = modelo(contexto_de(ids))[0]
        u[0] = float("-inf") if modelo.modo == "vocabulario" else -2.0     # nunca <UNK>
        valores = modelo.probabilidades(u)
        mejores = valores.topk(5).indices.tolist()
        top.append(f"| {frase} | {tipo} | " + " | ".join(f"{vocabulario[i]} ({valores[i]:.2%})" for i in mejores) + " |")
        nuevos = []
        for _ in range(5):
            u = modelo(contexto_de(ids))[0]
            u[0] = float("-inf")
            ids.append(int(u.argmax()))
            nuevos.append(str(vocabulario[ids[-1]]))
        generacion.append(f"| {frase} | {tipo} | {' '.join(nuevos)} |")

    texto = "## Top 5 predicciones\n\n" + "\n".join(top) + "\n\n## Generación continua\n\n" + "\n".join(generacion) + "\n"
    print("\n" + texto)
    if not args.frase:
        (directorio / "predicciones.md").write_text(texto, encoding="utf-8")
        print(f"Guardado en: {directorio / 'predicciones.md'}")


if __name__ == "__main__":
    main()
