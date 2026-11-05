"""
CBOW con una capa de auto-atencion, en PyTorch (el gradiente lo calcula el autograd).

En CBOW la capa oculta es el promedio del contexto, h = (1/C) sum_c v_Ic: una atencion con pesos fijos 1/C.
Aca esos pesos los decide una capa de atencion de H cabezas (d = N / H por cabeza). Para una muestra:

    X   = W[I_1..I_C] + P                      (C x N)  embeddings del contexto + posicionales
    q   = x_centro Wq                           (N)      consulta de la posicion central (vector aprendido)
    K   = X Wk,  V = X Wv                       (C x N)  se parten en H cabezas de d columnas
    s_hc = q_h . K_hc / sqrt(d),  s_h,nulo = b_nulo_h      puntuaciones (+ una "clave nula" por cabeza)
    a_h = softmax(s_h)                          (C + 1)  pesos de atencion
    z_h = sum_c a_hc V_hc + a_h,nulo v_nulo_h   (d)      salida de cada cabeza
    h   = [z_1 .. z_H] Wo                       (N)      reemplaza al promedio de CBOW
    u   = W'^t h,  y = softmax(u),  E = -log y_O         igual que en CBOW

La clave nula le da a cada cabeza un lugar donde "no mirar nada" sin tener que apoyarse en los signos.

Inicializacion: Wv = Wo = I, x_centro = 0 (=> q = 0 => atencion uniforme), P = 0, v_nulo = 0 y b_nulo = -10
(=> peso nulo ~ 0). Con eso h = promedio del contexto: el modelo arranca siendo exactamente el CBOW cargado.

Parametros y su forma (N = 80, C = 8, |V| = 53.409, H cabezas, d = N / H):
    W         (|V|, N)  embeddings de entrada del CBOW (fila = palabra)
    W_prima   (N, |V|)  pesos de salida del CBOW (columna = palabra a predecir)
    x_centro  (N,)      "embedding" de la posicion central: de ahi sale la consulta q
    P         (C, N)    un embedding posicional por posicion del contexto (-4..-1, +1..+4)
    Wq, Wk    (N, N)    proyecciones a consultas y claves (las H cabezas van concatenadas en las columnas)
    Wv, Wo    (N, N)    proyeccion a valores y mezcla final de las cabezas
    b_nulo    (H,)      puntaje de la clave nula de cada cabeza
    v_nulo    (H, d)    valor de la clave nula de cada cabeza

Comparado con microgpt.py: es la misma atencion (q . k / raiz(d), softmax, promedio ponderado de v, Wo),
pero aca la consulta es la posicion central y las claves son las C palabras del contexto (de los dos lados),
y no hay MLP, residual ni normalizacion: la salida h va directo a la capa de salida del CBOW.
"""

import ast
from pathlib import Path
import numpy as np
import torch
from torch import nn

# Nombres de los parametros de la capa de atencion (todo lo que no es W ni W_prima)
PARAMETROS_ATENCION = ("x_centro", "P", "Wq", "Wk", "Wv", "Wo", "b_nulo", "v_nulo")


class CBOWConAtencion(nn.Module):
    # nn.Module: clase base de PyTorch para modelos. Todo atributo nn.Parameter queda registrado como
    # parametro entrenable: modelo.parameters() los devuelve y .to("cuda") los mueve a la GPU
    def __init__(self, W: np.ndarray, W_prima: np.ndarray, C: int, H: int, semilla: int = 26):
        """
        :param W: Embeddings de entrada del CBOW (|V| x N).
        :param W_prima: Pesos de salida del CBOW (N x |V|).
        :param C: Cantidad de palabras de contexto.
        :param H: Cantidad de cabezas (N debe ser multiplo de H).
        """
        super().__init__()
        N = W.shape[1]
        if N % H != 0:
            raise ValueError(f"La dimension del embedding ({N}) debe ser multiplo de la cantidad de cabezas ({H}).")
        self.H, self.d = H, N // H                         # cantidad de cabezas y dimension de cada una
        generador = torch.Generator().manual_seed(semilla)  # numeros al azar reproducibles para Wq y Wk

        # Pesos del CBOW
        self.W = nn.Parameter(torch.tensor(W, dtype=torch.float32))
        self.W_prima = nn.Parameter(torch.tensor(W_prima, dtype=torch.float32))
        # Capa de atencion
        self.x_centro = nn.Parameter(torch.zeros(N))       # 0 => q = 0 => todos los puntajes iguales => atencion uniforme
        self.P = nn.Parameter(torch.zeros(C, N))           # 0 => al principio no cambia los embeddings
        # Wq y Wk al azar con desvio 1/raiz(N): si fueran 0 su gradiente seria 0 y nunca aprenderian
        self.Wq = nn.Parameter(torch.randn(N, N, generator=generador) / N**0.5)
        self.Wk = nn.Parameter(torch.randn(N, N, generator=generador) / N**0.5)
        self.Wv = nn.Parameter(torch.eye(N))               # identidad: los valores son los embeddings tal cual
        self.Wo = nn.Parameter(torch.eye(N))               # identidad: h = salida de la atencion tal cual
        self.b_nulo = nn.Parameter(torch.full((H,), -10.0))   # exp(-10) ~ 0: la clave nula arranca sin peso
        self.v_nulo = nn.Parameter(torch.zeros(H, self.d))

    def forward(self, contexto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Pasada hacia adelante para un lote de B muestras. Se llama como modelo(contexto).
        Solo se escribe esto: el autograd registra cada operacion y despues calcula los gradientes solo.

        :param contexto: Indices del contexto (B x C).
        :return: (u de forma B x |V|, pesos de atencion a de forma B x H x (C+1); la ultima columna es la nula).
        """
        B, C = contexto.shape
        H, d = self.H, self.d

        X = self.W[contexto] + self.P                                    # (B, C, N) embeddings + posicion
        q = (self.x_centro @ self.Wq).view(H, d)                         # (H, d) una consulta por cabeza (igual para todo el lote)
        K = (X @ self.Wk).view(B, C, H, d)                               # claves, partidas en H cabezas de d
        V = (X @ self.Wv).view(B, C, H, d)                               # valores, partidos igual

        # einsum: "hd,bchd->bhc" = para cada muestra b, cabeza h y posicion c, suma sobre d de q[h,d] * K[b,c,h,d]
        s = torch.einsum("hd,bchd->bhc", q, K) / d**0.5                  # (B, H, C) puntajes q . k / raiz(d)
        # Se agrega una columna mas con el puntaje de la clave nula (el mismo para todas las muestras)
        s = torch.cat([s, self.b_nulo.view(1, H, 1).expand(B, H, 1)], dim=2)          # (B, H, C+1)
        a = torch.softmax(s, dim=2)                                      # pesos: positivos y suman 1 sobre las C+1
        V = torch.cat([V, self.v_nulo.view(1, 1, H, d).expand(B, 1, H, d)], dim=1)    # (B, C+1, H, d)

        # Promedio de los valores ponderado por los pesos: z[b,h,:] = sum_c a[b,h,c] * V[b,c,h,:]
        z = torch.einsum("bhc,bchd->bhd", a, V).reshape(B, H * d)        # cabezas concatenadas
        h = z @ self.Wo                                                  # (B, N)
        u = h @ self.W_prima                                             # (B, |V|) puntaje de cada palabra
        return u, a


def cargar_respaldo(ruta_archivo: str | Path) -> dict:
    """Lee un respaldo .npz del CBOW (el formato de guardar_modelo) con NumPy."""
    datos = np.load(ruta_archivo, allow_pickle=True)   # allow_pickle: el vocabulario es un arreglo de cadenas
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


def guardar_respaldo(modelo: CBOWConAtencion, datos: dict, ruta_archivo: str | Path) -> None:
    """
    Guarda W, W' y el vocabulario con las mismas claves que guardar_modelo (asi cargar_modelo y
    scripts/pruebas.py pueden leer los embeddings), mas la atencion con prefijo "atencion_".
    """
    destino = Path(ruta_archivo)
    destino.parent.mkdir(parents=True, exist_ok=True)
    # .detach() saca el tensor del grafo del autograd, .cpu() lo trae de la GPU, .numpy() lo convierte
    W = modelo.W.detach().cpu().numpy()
    salida = {
        "W": W,
        "W_prima": modelo.W_prima.detach().cpu().numpy(),
        "vocabulario_palabras": np.asarray(datos["vocabulario_palabras"], dtype=object),
        "epoca_actual": int(datos["epoca_actual"]),
        "historial_perdida": np.asarray(datos["historial_perdida"], dtype=np.float32),
        "historial_accuracy": np.asarray(datos["historial_accuracy"], dtype=np.float32),
        "tamanio_vocabulario": int(W.shape[0]),
        "dimension_embedding": int(W.shape[1]),
        "cantidad_cabezas": int(modelo.H),
        "configuracion_json": str(datos["configuracion"]),
        "atencion_media": np.asarray(datos["atencion_media"]),   # (H, C+1) peso medio por posicion en la evaluacion
    }
    for nombre in PARAMETROS_ATENCION:
        salida[f"atencion_{nombre}"] = getattr(modelo, nombre).detach().cpu().numpy()
    np.savez_compressed(destino, **salida)
    print(f"Modelo guardado exitosamente en: '{destino}'", flush=True)
