"""
Perceptron multicapa (PMC) como modelo de lenguaje: consignas 1 y 2 del TP2.

Idea: para predecir el token siguiente se toman los k tokens anteriores, se buscan sus representaciones
contextuales en la matriz W del CBOW del TP1 (d = 80) y se concatenan en un solo vector de k x d componentes.
Ese vector es la entrada del PMC. W NO es una capa Embedding: es un tensor fijo (no se entrena) del que
solo se leen filas (construir_entrada).

Dos variantes de salida (consigna 2):
    "vocabulario": tantas unidades como palabras (|V| = 53.409) + softmax; perdida = entropia cruzada.
    "vector":      tantas unidades como la dimension de la representacion (d = 80) con activacion tanh o
                   logistica; el objetivo es la representacion de la palabra siguiente, escalada (min-max) a la
                   imagen de la activacion: [-1, 1] para tanh, [0, 1] para logistica; perdida = error cuadratico.
                   Para saber "que palabra predijo" se buscan las filas de W escalada mas parecidas (coseno).

En las dos variantes se obtiene un puntaje por palabra del vocabulario (u), asi los contadores de ranking
(top-5, top-10, top-20) se calculan igual en las dos.
"""

import ast
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ACTIVACIONES = {"relu": nn.ReLU, "tanh": nn.Tanh, "logistica": nn.Sigmoid}
CORTES_RANKING = (5, 10, 20)     # contadores de acierto: la palabra correcta quedo entre las primeras 5, 10 o 20


def cargar_respaldo(ruta_archivo: str | Path) -> dict:
    """Lee un respaldo .npz del CBOW del TP1 (el formato de guardar_modelo) con NumPy."""
    datos = np.load(ruta_archivo, allow_pickle=True)   # allow_pickle: el vocabulario es un arreglo de cadenas (archivo propio)
    respaldo = {
        "W": datos["W"],
        "W_prima": datos["W_prima"],
        "vocabulario_palabras": datos["vocabulario_palabras"],
        "epoca_actual": int(datos.get("epoca_actual", 0)),
        "configuracion": {},
    }
    if "configuracion_json" in datos:
        # La configuracion se guardo como texto (str de un diccionario); literal_eval la vuelve diccionario
        respaldo["configuracion"] = ast.literal_eval(str(datos["configuracion_json"]))
    print(f"Modelo cargado exitosamente desde: '{ruta_archivo}' (Epoca cargada: {respaldo['epoca_actual']})", flush=True)
    return respaldo


def puesto_en_ranking(u: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
    """
    Puesto de la palabra correcta en el ranking de cada prediccion (1 = la mas probable).
    Es la cantidad de palabras con un puntaje u mayor o igual que el de la correcta (la correcta incluida).
    Ejemplo: si la correcta tiene el 3er puntaje mas alto, su puesto es 3 y cuenta como acierto
    en los cortes 5, 10 y 20 (pero no como acierto exacto).
    Ante un empate se cuenta el peor puesto (si todas empatan, el puesto es |V| y no un acierto falso).

    :param u: Puntajes (B x |V|).
    :param objetivo: Indice de la palabra correcta de cada prediccion (B).
    :return: Puesto de cada prediccion (B), entre 1 y |V|.
    """
    u_correcta = u.gather(1, objetivo[:, None])                     # (B, 1) puntaje de la palabra correcta
    return (u >= u_correcta).sum(dim=1)


def construir_entrada(W: torch.Tensor, contexto: torch.Tensor) -> torch.Tensor:
    """
    Busca la representacion de cada token del contexto y las concatena (aplana).

    :param W: Representaciones del CBOW (|V| x d), fijas.
    :param contexto: Indices de los k tokens anteriores (B x k).
    :return: Entradas del PMC (B x k*d). Ejemplo: k = 5, d = 80 -> 400 componentes.
    """
    return W[contexto].reshape(contexto.shape[0], -1)      # (B, k, d) -> (B, k*d)


def escalar_min_max(W: torch.Tensor, activacion_salida: str) -> torch.Tensor:
    """Escala todas las componentes de W a la imagen de la activacion de salida (min-max global)."""
    W01 = (W - W.min()) / (W.max() - W.min())               # a [0, 1]
    return W01 if activacion_salida == "logistica" else 2.0 * W01 - 1.0   # tanh: a [-1, 1]


class PerceptronMulticapa(nn.Module):
    """
    entrada (k*d) -> [Lineal -> (normalizacion por capa) -> activacion] por cada capa oculta -> Lineal de salida.

    :param dim_entrada: k * d.
    :param capas_ocultas: Unidades de cada capa oculta, por ejemplo [512] o [256, 256].
    :param dim_salida: |V| (variante "vocabulario") o d (variante "vector").
    :param normalizacion: Si es True, normalizacion por capa (LayerNorm) antes de cada activacion oculta.
    :param activacion: Activacion de las capas ocultas ("relu", "tanh" o "logistica").
    """

    def __init__(self, dim_entrada: int, capas_ocultas: list[int], dim_salida: int,
                 normalizacion: bool = True, activacion: str = "relu"):
        super().__init__()
        capas = []
        anterior = dim_entrada
        for unidades in capas_ocultas:
            capas.append(nn.Linear(anterior, unidades))
            if normalizacion:
                capas.append(nn.LayerNorm(unidades))
            capas.append(ACTIVACIONES[activacion]())
            anterior = unidades
        self.ocultas = nn.Sequential(*capas)
        self.salida = nn.Linear(anterior, dim_salida)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Devuelve la salida ANTES de la activacion final (logits o pre-activacion del vector)."""
        return self.salida(self.ocultas(x))


class ModeloLenguajePMC(nn.Module):
    """
    Junta W (fija), el PMC y la forma de puntuar cada palabra del vocabulario segun la variante.

    forward(contexto) devuelve u (B x |V|): un puntaje por palabra, mas alto = mas probable.
        "vocabulario": u son los logits (la softmax de u da las probabilidades).
        "vector":      u es el coseno entre el vector predicho y cada fila de W escalada.
    """

    def __init__(self, W: torch.Tensor, k: int, modo: str, capas_ocultas: list[int],
                 normalizacion: bool = True, activacion: str = "relu", activacion_salida: str = "tanh"):
        super().__init__()
        if modo not in ("vocabulario", "vector"):
            raise ValueError(f"modo desconocido: '{modo}' (vocabulario o vector)")
        self.k, self.modo, self.activacion_salida = k, modo, activacion_salida
        W = torch.as_tensor(W, dtype=torch.float32)
        V, d = W.shape
        # register_buffer: se mueve con .to(dispositivo) y se guarda, pero NO es un parametro (no se entrena)
        self.register_buffer("W", W)
        if modo == "vector":
            self.register_buffer("W_escalada", escalar_min_max(W, activacion_salida))
        self.pmc = PerceptronMulticapa(k * d, capas_ocultas, V if modo == "vocabulario" else d,
                                       normalizacion, activacion)

    def vector_predicho(self, contexto: torch.Tensor) -> torch.Tensor:
        """Variante "vector": salida del PMC con la activacion final (en la escala de W_escalada)."""
        z = self.pmc(construir_entrada(self.W, contexto))
        return torch.sigmoid(z) if self.activacion_salida == "logistica" else torch.tanh(z)

    def perdida(self, contexto: torch.Tensor, objetivo: torch.Tensor) -> torch.Tensor:
        """Entropia cruzada (vocabulario) o error cuadratico medio contra W_escalada[objetivo] (vector)."""
        if self.modo == "vocabulario":
            return F.cross_entropy(self.pmc(construir_entrada(self.W, contexto)), objetivo)
        return F.mse_loss(self.vector_predicho(contexto), self.W_escalada[objetivo])

    def forward(self, contexto: torch.Tensor) -> torch.Tensor:
        if self.modo == "vocabulario":
            return self.pmc(construir_entrada(self.W, contexto))
        y = F.normalize(self.vector_predicho(contexto), dim=1)
        return y @ F.normalize(self.W_escalada, dim=1).t()           # coseno con cada palabra

    def probabilidades(self, u: torch.Tensor) -> torch.Tensor:
        """Lo que se muestra como "%" en las tablas: softmax (vocabulario) o el coseno (vector), como en el TP de Pablo."""
        return torch.softmax(u, dim=-1) if self.modo == "vocabulario" else u
