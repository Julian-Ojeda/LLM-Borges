"""
Transformador para completar frases (predecir la palabra siguiente), en PyTorch, siguiendo el material de la
catedra (Transformadores a, b y c). Usa como matriz E los embeddings del CBOW entrenado en el TP.

Datos (Transformadores c, pag. 55-56): una ventana de N tokens recorre el corpus de a 1 token; el modelo
recibe los N tokens y tiene que predecir el siguiente:
    [volvió a su casa y se sentó en ... la] -> mesa          (N = 16)

Entrada (Transformadores c, pag. 16 y 36-37):
    x_i = raiz(d) rc_i + vp_i
                             rc_i = fila de E del token i (su representacion contextual del CBOW)
                             vp_i = vector posicional sinusoidal (fijo, no se aprende), pag. 7:
                                    vp(p, j) = sen(p / 10000^(2j/d))      si j es par
                                               cos(p / 10000^((2j-1)/d))  si j es impar      (p, j desde 1)

Bloque de transformacion (Transformadores b, pag. 15), se pueden apilar L bloques:
    T1 = autoAtencionMultiple(X)        causal: cada token solo ve a los anteriores (y a si mismo)
    T2 = X + T1                          conexion residual
    T3 = Normalizacion(T2)               normalizacion por capa (media, desvio, gamma y beta aprendidos)
    T4 = PMC(T3)                         d -> 4d -> d con GELU, el mismo PMC para cada token
    T5 = T3 + T4                         conexion residual
    H  = Normalizacion(T5)

Auto-atencion multiple causal (Transformadores a, pag. 32-37 y 51), para cada cabeza k = 1..h:
    Q_k = X W_Qk,  K_k = X W_Kk,  V_k = X W_Vk                       (N x dk), dk = dv = d / h
    Y_k = softmax(Q_k K_k^t / raiz(dk)) V_k      con -infinito arriba de la diagonal de Q_k K_k^t (mascara causal)
    A   = (Y_1 +' Y_2 +' ... +' Y_h) W_O         (+' = concatenacion)

Cabezal de modelado del lenguaje (Transformadores c, pag. 42-45): se toma la ultima representacion del
ultimo bloque, h_N^L, y se multiplica por E^t (la MISMA matriz E de la entrada):
    u = h_N^L E^t   (1 x |V|),   y = softmax(u)
Perdida: entropia cruzada L_EC = -log y_siguiente (pag. 52-53).

Escala de E: como E se usa en la entrada y en el cabezal, su escala importa. Los embeddings del CBOW tienen
largo ~15 y h_N^L (normalizado) ~raiz(d) = 9, asi que u = h E^t saldria con desvio ~15 y la softmax
quedaria casi "segura" de entrada (perdida inicial ~99, aprendizaje lento). Por eso:
    - E arranca como los embeddings del CBOW divididos por su largo medio (filas de largo ~1). Es el mismo
      factor para todas las filas: las direcciones y las similitudes del CBOW no cambian.
    - En la entrada se multiplica por raiz(d), como en el articulo original (Vaswani et al., 2017,
      citado en Transformadores a, pag. 26), que tambien comparte la matriz de entrada y la de salida.

Parametros (|V| = 53.409, d = 80, h cabezas, L bloques):
    E                     (|V|, d)   embeddings (del CBOW); se usan en la entrada y en el cabezal
    por bloque: W_Q, W_K, W_V, W_O (d, d), normalizaciones (gamma, beta de d), PMC (d x 4d y 4d x d, con sesgos)
"""

import math
import torch
from torch import nn
import torch.nn.functional as F

CORTES_RANKING = (5, 10, 20)     # contadores de acierto: la palabra correcta quedo entre las primeras 5, 10 o 20


def vectores_posicionales(N: int, d: int) -> torch.Tensor:
    """
    Codificacion de la posicion de la catedra (Transformadores c, pag. 7), con p y j desde 1.
    Reproduce el ejemplo de la pag. 15 (d = 32, p = 1: ... 0.31 0.97 0.10 1 0.03 1 ...).

    :return: Matriz (N, d): la fila p-1 es el vector posicional de la posicion p.
    """
    p = torch.arange(1, N + 1, dtype=torch.float64)[:, None]        # (N, 1) posiciones 1..N
    j = torch.arange(1, d + 1, dtype=torch.float64)[None, :]        # (1, d) elementos 1..d
    pares = torch.sin(p / 10000 ** (2 * j / d))
    impares = torch.cos(p / 10000 ** ((2 * j - 1) / d))
    return torch.where(j % 2 == 0, pares, impares).float()


class BloqueTransformacion(nn.Module):
    """Un bloque de transformacion: auto-atencion multiple causal + PMC, con residuales y normalizacion."""

    def __init__(self, d: int, h: int):
        super().__init__()
        self.h, self.dk = h, d // h
        # Las h matrices W_Qk (d x dk) van concatenadas en una sola de d x d (lo mismo con W_K y W_V)
        self.W_Q = nn.Linear(d, d, bias=False)
        self.W_K = nn.Linear(d, d, bias=False)
        self.W_V = nn.Linear(d, d, bias=False)
        self.W_O = nn.Linear(d, d, bias=False)                       # (h * dv) -> d
        self.normalizacion_1 = nn.LayerNorm(d)                       # gamma arranca en 1 y beta en 0
        self.normalizacion_2 = nn.LayerNorm(d)
        self.PMC_1 = nn.Linear(d, 4 * d)                             # capa oculta d_ff = 4d (512 -> 2048 en el articulo)
        self.PMC_2 = nn.Linear(4 * d, d)

    def auto_atencion(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param X: (B, N, d).
        :return: (A de forma (B, N, d), pesos alfa de forma (B, h, N, N)).
        """
        B, N, d = X.shape
        # (B, N, d) -> (B, h, N, dk): se separan las h cabezas
        Q = self.W_Q(X).view(B, N, self.h, self.dk).transpose(1, 2)
        K = self.W_K(X).view(B, N, self.h, self.dk).transpose(1, 2)
        V = self.W_V(X).view(B, N, self.h, self.dk).transpose(1, 2)
        S = Q @ K.transpose(-2, -1) / math.sqrt(self.dk)             # (B, h, N, N): elemento (i, j) = q_i k_j / raiz(dk)
        mascara = torch.ones(N, N, dtype=torch.bool, device=X.device).triu(1)   # True arriba de la diagonal (j > i)
        S = S.masked_fill(mascara, float("-inf"))                    # causal: el token i no ve a los j > i
        alfa = torch.softmax(S, dim=-1)                              # softmax de cada fila
        Y = alfa @ V                                                 # (B, h, N, dk)
        Y = Y.transpose(1, 2).reshape(B, N, d)                       # concatenacion de las h cabezas: (B, N, h*dv)
        return self.W_O(Y), alfa

    def forward(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        T1, alfa = self.auto_atencion(X)
        T2 = X + T1
        T3 = self.normalizacion_1(T2)
        # GELU(x) = 0,5 x (1 + tanh(raiz(2/pi) (x + 0,044715 x^3))), la definicion de la pag. 6 de Transformadores b
        T4 = self.PMC_2(F.gelu(self.PMC_1(T3), approximate="tanh"))
        T5 = T3 + T4
        H = self.normalizacion_2(T5)
        return H, alfa


class SiguientePalabra(nn.Module):
    def __init__(self, E, N: int, h: int, L: int):
        """
        :param E: Embeddings del CBOW (|V| x d), arreglo de NumPy.
        :param N: Largo de la ventana de entrada (cantidad de tokens anteriores).
        :param h: Cantidad de cabezas de atencion (d debe ser multiplo de h).
        :param L: Cantidad de bloques de transformacion apilados.
        """
        super().__init__()
        d = E.shape[1]
        if d % h != 0:
            raise ValueError(f"La dimension d ({d}) debe ser multiplo de la cantidad de cabezas ({h}).")
        self.N, self.h, self.L, self.d = N, h, L, d
        E = torch.tensor(E, dtype=torch.float32)
        self.E = nn.Parameter(E / E.norm(dim=1).mean())             # filas de largo medio 1 (ver "Escala de E")
        # buffer: se guarda y se mueve a la GPU con el modelo, pero no es un parametro (no se entrena)
        self.register_buffer("vp", vectores_posicionales(N, d))
        self.bloques = nn.ModuleList([BloqueTransformacion(d, h) for _ in range(L)])

    def forward(self, contexto: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        :param contexto: Indices de los N tokens de entrada (B x N), del mas lejano al mas cercano.
        :return: (u de forma B x |V|, pesos de atencion del ultimo token en el ultimo bloque, B x h x N).
        """
        X = self.E[contexto] * math.sqrt(self.d) + self.vp[: contexto.shape[1]]   # x_i = raiz(d) rc_i + vp_i
        for bloque in self.bloques:
            X, alfa = bloque(X)
        h_N = X[:, -1]                                               # ultima representacion del ultimo bloque
        u = h_N @ self.E.t()                                         # cabezal: producto con todas las filas de E
        return u, alfa[:, :, -1]


def cargar_transformador(ruta) -> tuple[SiguientePalabra, object, dict]:
    """
    Reconstruye el modelo desde un .npz guardado por scripts/entrenar_completar_torch.py (en CPU).

    :return: (modelo en modo evaluacion, vocabulario, configuracion con la que se entreno).
    """
    import ast
    import numpy as np
    datos = np.load(ruta, allow_pickle=True)
    N, h, L = int(datos["tamanio_contexto"]), int(datos["cantidad_cabezas"]), int(datos["cantidad_bloques"])
    modelo = SiguientePalabra(datos["W"], N, h, L)
    pesos = {clave[len("transformador."):]: torch.tensor(datos[clave])
             for clave in datos.files if clave.startswith("transformador.")}
    with torch.no_grad():
        modelo.E.copy_(torch.tensor(datos["W"]))                     # E tal cual se guardo (ya reescalada)
    modelo.load_state_dict({**pesos, "E": modelo.E.data, "vp": modelo.vp})
    modelo.eval()
    return modelo, datos["vocabulario_palabras"], ast.literal_eval(str(datos["configuracion_json"]))


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


@torch.no_grad()
def generar(modelo: SiguientePalabra, ids: list[int], cantidad: int, estrategia: str = "top_p",
            temperatura: float = 0.8, k: int = 40, p: float = 0.9, prohibidos: list[int] = (),
            generador: torch.Generator | None = None) -> list[int]:
    """
    Genera `cantidad` tokens a continuacion de `ids` (al menos N), uno por vez: predice, elige, agrega al final
    y corre la ventana. Estrategias de eleccion (Transformadores c, pag. 46-50):
        "ansiosa": siempre la mas probable (greedy decoding).
        "top_k":   al azar entre las k mas probables, segun sus probabilidades renormalizadas.
        "top_p":   al azar entre las mas probables hasta acumular una masa p de probabilidad (nucleus sampling).
    La temperatura (como en microgpt.py) divide a u antes de la softmax: < 1 concentra, > 1 reparte.

    :param prohibidos: Indices que nunca se eligen (por ejemplo <UNK>).
    :return: Los indices generados (sin el comienzo).
    """
    dispositivo = modelo.E.device
    ids = list(ids)
    nuevos = []
    for _ in range(cantidad):
        u, _ = modelo(torch.tensor([ids[-modelo.N:]], device=dispositivo))
        u = u[0] / temperatura
        if prohibidos:
            u[list(prohibidos)] = float("-inf")
        if estrategia == "ansiosa":
            elegido = int(u.argmax())
        else:
            probabilidades, indices = torch.softmax(u, dim=0).sort(descending=True)
            if estrategia == "top_k":
                cantidad_candidatas = k
            elif estrategia == "top_p":
                # cuantas hacen falta para acumular p (al menos 1)
                cantidad_candidatas = int((probabilidades.cumsum(0) < p).sum()) + 1
            else:
                raise ValueError(f"Estrategia desconocida: '{estrategia}' (ansiosa, top_k o top_p)")
            candidatas = probabilidades[:cantidad_candidatas]
            sorteada = torch.multinomial(candidatas / candidatas.sum(), 1, generator=generador)
            elegido = int(indices[sorteada])
        ids.append(elegido)
        nuevos.append(elegido)
    return nuevos


def unir_tokens(tokens: list[str]) -> str:
    """Une tokens en texto: sin espacio antes de , . ; : ! ? ) ” y sin espacio despues de ¿ ¡ ( “."""
    texto = ""
    for token in tokens:
        if texto and token not in {",", ".", ";", ":", "!", "?", ")", "”"} and texto[-1] not in "¿¡(“":
            texto += " "
        texto += token
    return texto
