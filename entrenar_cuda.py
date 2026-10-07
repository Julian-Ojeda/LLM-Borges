#!/usr/bin/env python3
"""
Entrenamiento CBOW con kernels CUDA propios (compilados en tiempo de ejecucion con CuPy RawModule).

Usa los mismos parametros (configuracion.yaml y argumentos de consola), el mismo formato de respaldos (.npz)
y el mismo archivo de metricas (CSV) que entrenar.py, asi que los pesos son intercambiables: se puede reanudar
desde un respaldo generado por entrenar.py y viceversa.

La logica es la de las clases, muestra a muestra (sin lotes). Para cada muestra (contexto, palabra central O):

    h = (1/C) (v_I1 + ... + v_IC)                  h  (N x 1)
    u = W'^t h,   y = softmax(u),   E = -log y_O   u, y (|V| x 1)
    e = y - t,    EH = W' e                        e (|V| x 1), EH (N x 1), con W' anterior a la actualizacion
    w'_ij  <- w'_ij  - eta e_j h_i                 toda la matriz W' (N x |V|)
    W_Ic,. <- W_Ic,. - eta (1/C) EH^t              cada palabra Ic del contexto (fila de W, |V| x N)

Esas cuentas se reparten en 3 kernels por muestra:
    1. cbow_salida:             h y u = W'^t h, mas el maximo y la suma parcial de la softmax de cada bloque.
    2. cbow_actualizar_salida:  y, e, EH (con W' anterior) y la actualizacion de W'; acumula E y el acierto.
    3. cbow_actualizar_entrada: la actualizacion de las filas de W del contexto.

Solo implementa el modo softmax completo (cantidad_negativos: 0).
"""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm

# Asegurar acceso a los modulos del directorio codigo/
ruta_raiz = Path(__file__).resolve().parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))

from codigo.configuracion import cargar_configuracion
from codigo.tokenizador import tokenizar_corpus
from codigo.generador_vocabulario import construir_vocabulario
from codigo.entrenador_cbow import preparar_muestras, imprimir_resumen, TOKEN_DESCONOCIDO
from codigo.modelo_cbow_cupy import inicializar_pesos, guardar_modelo, cargar_modelo, USAR_CUPY, xp

HILOS = 256

CODIGO_CUDA = r"""
#define HILOS 256
#define MENOS_INFINITO __int_as_float(0xff800000)

// Maximo (y su indice, el menor ante empates) de red_val/red_ind en el bloque. Resultado en la posicion 0.
__device__ void reducir_maximo(float* red_val, int* red_ind, int tid) {
    for (int paso = HILOS / 2; paso > 0; paso >>= 1) {
        if (tid < paso) {
            float a = red_val[tid], b = red_val[tid + paso];
            int ia = red_ind[tid], ib = red_ind[tid + paso];
            if (b > a || (b == a && ib < ia)) { red_val[tid] = b; red_ind[tid] = ib; }
        }
        __syncthreads();
    }
}

// Suma de red_val en el bloque. Resultado en la posicion 0.
__device__ void reducir_suma(float* red_val, int tid) {
    for (int paso = HILOS / 2; paso > 0; paso >>= 1) {
        if (tid < paso) red_val[tid] += red_val[tid + paso];
        __syncthreads();
    }
}

// Kernel 1: h = (1/C) sum_c W_Ic,.  y  u_j = v'_j^t h. Un hilo por palabra j de la salida.
extern "C" __global__ void cbow_salida(
    const float* __restrict__ W, const float* __restrict__ Wp, const int* __restrict__ contexto,
    const int idx, const int V, const int N, const int C,
    float* __restrict__ u, float* __restrict__ h_global,
    float* __restrict__ max_parcial, int* __restrict__ arg_parcial, float* __restrict__ suma_parcial)
{
    extern __shared__ float h[];                       // (N)
    __shared__ float red_val[HILOS];
    __shared__ int red_ind[HILOS];
    const int tid = threadIdx.x;
    const int j = blockIdx.x * HILOS + tid;
    const int* ctx = contexto + (long long)idx * C;

    for (int i = tid; i < N; i += HILOS) {
        float s = 0.0f;
        for (int c = 0; c < C; ++c) s += W[(long long)ctx[c] * N + i];
        h[i] = s / C;
        if (blockIdx.x == 0) h_global[i] = h[i];
    }
    __syncthreads();

    float uj = MENOS_INFINITO;
    if (j < V) {
        float s = 0.0f;
        for (int i = 0; i < N; ++i) s += Wp[(long long)i * V + j] * h[i];
        uj = s;
        u[j] = uj;
    }

    // Maximo del bloque (estabilidad numerica de la softmax y prediccion argmax)
    red_val[tid] = uj;
    red_ind[tid] = j;
    __syncthreads();
    reducir_maximo(red_val, red_ind, tid);
    const float m = red_val[0];
    const int arg = red_ind[0];
    __syncthreads();

    // Suma parcial de exp(u_j - m) del bloque
    red_val[tid] = (j < V) ? expf(uj - m) : 0.0f;
    __syncthreads();
    reducir_suma(red_val, tid);
    if (tid == 0) {
        max_parcial[blockIdx.x] = m;
        arg_parcial[blockIdx.x] = arg;
        suma_parcial[blockIdx.x] = red_val[0];
    }
}

// Kernel 2: y = softmax(u), e = y - t, EH = W' e (con W' anterior) y w'_ij <- w'_ij - eta e_j h_i.
extern "C" __global__ void cbow_actualizar_salida(
    float* __restrict__ Wp, const float* __restrict__ u, const float* __restrict__ h_global,
    const int* __restrict__ objetivo, const int idx, const int V, const int N, const int nb, const float eta,
    const float* __restrict__ max_parcial, const int* __restrict__ arg_parcial, const float* __restrict__ suma_parcial,
    float* __restrict__ EH_parcial, double* __restrict__ perdida_acumulada, unsigned long long* __restrict__ aciertos)
{
    extern __shared__ float compartida[];              // h (N) y EH parcial del bloque (N)
    float* h = compartida;
    float* EH = compartida + N;
    __shared__ float red_val[HILOS];
    __shared__ int red_ind[HILOS];
    __shared__ float M_s, S_s;
    __shared__ int arg_s;
    const int tid = threadIdx.x;
    const int j = blockIdx.x * HILOS + tid;
    const int O = objetivo[idx];

    for (int i = tid; i < N; i += HILOS) { h[i] = h_global[i]; EH[i] = 0.0f; }

    // Maximo global de u y su indice, a partir de los parciales de los bloques
    float mv = MENOS_INFINITO;
    int mi = 0x7fffffff;
    for (int b = tid; b < nb; b += HILOS) {
        float v = max_parcial[b];
        int ib = arg_parcial[b];
        if (v > mv || (v == mv && ib < mi)) { mv = v; mi = ib; }
    }
    red_val[tid] = mv;
    red_ind[tid] = mi;
    __syncthreads();
    reducir_maximo(red_val, red_ind, tid);
    if (tid == 0) { M_s = red_val[0]; arg_s = red_ind[0]; }
    __syncthreads();
    const float M = M_s;

    // S = sum_j exp(u_j - M)
    float s = 0.0f;
    for (int b = tid; b < nb; b += HILOS) s += suma_parcial[b] * expf(max_parcial[b] - M);
    red_val[tid] = s;
    __syncthreads();
    reducir_suma(red_val, tid);
    if (tid == 0) S_s = red_val[0];
    __syncthreads();
    const float S = S_s;

    // e_j = y_j - t_j
    float ej = 0.0f;
    if (j < V) ej = expf(u[j] - M) / S - (j == O ? 1.0f : 0.0f);

    // Una sola pasada por W': EH_i += w'_ij e_j (valor anterior) y w'_ij <- w'_ij - eta e_j h_i
    const int carril = tid & 31;
    for (int i = 0; i < N; ++i) {
        float aporte = 0.0f;
        if (j < V) {
            const long long pos = (long long)i * V + j;
            const float w = Wp[pos];
            aporte = w * ej;
            Wp[pos] = w - eta * ej * h[i];
        }
        for (int d = 16; d > 0; d >>= 1) aporte += __shfl_down_sync(0xffffffff, aporte, d);
        if (carril == 0) atomicAdd(&EH[i], aporte);
    }
    __syncthreads();
    for (int i = tid; i < N; i += HILOS) EH_parcial[(long long)blockIdx.x * N + i] = EH[i];

    // Metricas de la muestra: E = -log y_O (acotada como en la version CuPy) y acierto argmax(y) = O
    if (blockIdx.x == 0 && tid == 0) {
        perdida_acumulada[0] += (double)fminf(M + logf(S) - u[O], 27.6310211f);
        aciertos[0] += (arg_s == O) ? 1ULL : 0ULL;
    }
}

// Kernel 3: EH = suma de los parciales de los bloques y W_Ic,. <- W_Ic,. - eta (1/C) EH^t para cada c.
// Cada hilo recorre las C palabras en orden, asi una palabra repetida en el contexto acumula ambas actualizaciones.
extern "C" __global__ void cbow_actualizar_entrada(
    float* __restrict__ W, const int* __restrict__ contexto, const int idx,
    const int N, const int C, const int nb, const float eta, const float* __restrict__ EH_parcial)
{
    const int* ctx = contexto + (long long)idx * C;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float EH = 0.0f;
        for (int b = 0; b < nb; ++b) EH += EH_parcial[(long long)b * N + i];
        for (int c = 0; c < C; ++c) W[(long long)ctx[c] * N + i] -= eta / C * EH;
    }
}
"""


class PasoCBOWCuda:
    """
    Ejecuta un paso de descenso por el gradiente (softmax completo) por muestra con los 3 kernels CUDA.
    La perdida y los aciertos se acumulan en la GPU y se leen solo cuando se piden.
    """

    def __init__(self, tamanio_vocabulario: int, dimension_embedding: int, cantidad_contexto: int):
        modulo = xp.RawModule(code=CODIGO_CUDA)
        self.kernel_salida = modulo.get_function("cbow_salida")
        self.kernel_actualizar_salida = modulo.get_function("cbow_actualizar_salida")
        self.kernel_actualizar_entrada = modulo.get_function("cbow_actualizar_entrada")

        self.V = np.int32(tamanio_vocabulario)
        self.N = np.int32(dimension_embedding)
        self.C = np.int32(cantidad_contexto)
        self.bloques = (tamanio_vocabulario + HILOS - 1) // HILOS
        self.nb = np.int32(self.bloques)

        self.u = xp.empty(tamanio_vocabulario, dtype=xp.float32)
        self.h = xp.empty(dimension_embedding, dtype=xp.float32)
        self.max_parcial = xp.empty(self.bloques, dtype=xp.float32)
        self.arg_parcial = xp.empty(self.bloques, dtype=xp.int32)
        self.suma_parcial = xp.empty(self.bloques, dtype=xp.float32)
        self.EH_parcial = xp.empty((self.bloques, dimension_embedding), dtype=xp.float32)
        self.perdida_acumulada = xp.zeros(1, dtype=xp.float64)
        self.aciertos = xp.zeros(1, dtype=xp.uint64)

    def reiniciar_metricas(self) -> None:
        self.perdida_acumulada.fill(0)
        self.aciertos.fill(0)

    def leer_metricas(self) -> tuple[float, int]:
        return float(self.perdida_acumulada[0]), int(self.aciertos[0])

    def __call__(self, W, W_prima, contexto, objetivo, indice: int, eta: float) -> None:
        idx = np.int32(indice)
        bytes_h = int(self.N) * 4
        self.kernel_salida(
            (self.bloques,), (HILOS,),
            (W, W_prima, contexto, idx, self.V, self.N, self.C,
             self.u, self.h, self.max_parcial, self.arg_parcial, self.suma_parcial),
            shared_mem=bytes_h,
        )
        self.kernel_actualizar_salida(
            (self.bloques,), (HILOS,),
            (W_prima, self.u, self.h, objetivo, idx, self.V, self.N, self.nb, np.float32(eta),
             self.max_parcial, self.arg_parcial, self.suma_parcial,
             self.EH_parcial, self.perdida_acumulada, self.aciertos),
            shared_mem=2 * bytes_h,
        )
        self.kernel_actualizar_entrada(
            (1,), (128,),
            (W, contexto, idx, self.N, self.C, self.nb, np.float32(eta), self.EH_parcial),
        )


def entrenar_cuda(configuracion_dict: dict) -> dict:
    """
    Misma logica que codigo.entrenador_cbow.entrenar (reanudacion, muestras, orden aleatorio por época,
    metricas, respaldos), con el paso de cada muestra calculado por los kernels CUDA.

    :param configuracion_dict: Diccionario que contiene todos los hiperparametros y rutas.
    :return: Diccionario 'modelo' con W, W_prima, vocabulario_palabras, historiales y metadatos.
    """
    if not USAR_CUPY:
        raise RuntimeError("entrenar_cuda.py necesita una GPU NVIDIA con CuPy; en CPU use entrenar.py.")
    if configuracion_dict["cantidad_negativos"] > 0:
        raise ValueError("entrenar_cuda.py solo implementa softmax completo; ponga cantidad_negativos: 0 o use entrenar.py.")

    tiempo_inicio_total = time.time()
    configuracion = dict(configuracion_dict)

    ruta_corpus = configuracion["ruta_corpus"]
    directorio_respaldos = Path(configuracion["directorio_respaldos"])
    tamanio_ventana = configuracion["tamanio_ventana"]
    tasa_aprendizaje = configuracion["tasa_aprendizaje"]
    cantidad_epocas = configuracion["cantidad_epocas"]
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

    W = xp.ascontiguousarray(W, dtype=xp.float32)
    W_prima = xp.ascontiguousarray(W_prima, dtype=xp.float32)
    tamanio_vocabulario, dimension_embedding = W.shape
    if tamanio_vocabulario != len(vocabulario_palabras) or W_prima.shape != (dimension_embedding, tamanio_vocabulario):
        raise ValueError(f"Dimensiones inconsistentes: W {W.shape}, W' {W_prima.shape}, |V| = {len(vocabulario_palabras)}")

    # 2. Muestras (contexto, palabra central) como indices, cargadas una sola vez en la GPU
    mapeo_vocabulario = {palabra: indice for indice, palabra in enumerate(vocabulario_palabras)}
    _, contexto_cpu, objetivo_cpu = preparar_muestras(tokens_corpus, mapeo_vocabulario, tamanio_ventana)
    contexto = xp.ascontiguousarray(xp.asarray(contexto_cpu, dtype=xp.int32))
    objetivo = xp.ascontiguousarray(xp.asarray(objetivo_cpu, dtype=xp.int32))
    total_muestras = len(objetivo_cpu)

    paso = PasoCBOWCuda(tamanio_vocabulario, dimension_embedding, contexto.shape[1])

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
        "Modo de salida": "Softmax completo sobre todo |V| (kernels CUDA)",
        "Tasa de aprendizaje (eta)": tasa_aprendizaje,
        "Épocas a entrenar": cantidad_epocas,
        "Semilla aleatoria": semilla_aleatoria,
        "Cómputo": "GPU (kernels CUDA propios vía CuPy)",
        "Métricas por época": str(ruta_metricas),
    }
    imprimir_resumen("CONFIGURACIÓN DEL ENTRENAMIENTO (CUDA)", datos_resumen)

    # 3. Bucle de entrenamiento: una actualización de W y W' por cada muestra
    epoca_final = epoca_inicial + cantidad_epocas
    for epoca in range(epoca_inicial + 1, epoca_final + 1):
        tiempo_inicio = time.time()
        orden = np.random.default_rng(semilla_aleatoria + epoca).permutation(total_muestras)
        paso.reiniciar_metricas()

        barra_progreso = tqdm(orden, desc=f"Epoca {epoca}/{epoca_final}", unit="muestra", mininterval=2.0)
        for procesadas, indice in enumerate(barra_progreso, start=1):
            paso(W, W_prima, contexto, objetivo, int(indice), tasa_aprendizaje)

            if procesadas % 20000 == 0:
                perdida_acumulada, aciertos = paso.leer_metricas()
                barra_progreso.set_postfix(
                    {"Perdida": f"{perdida_acumulada / procesadas:.4f}", "Acc": f"{aciertos / procesadas:.3f}"}
                )

        perdida_acumulada, aciertos = paso.leer_metricas()
        duracion_epoca = time.time() - tiempo_inicio
        historial_perdida.append(perdida_acumulada / total_muestras)
        historial_accuracy.append(aciertos / total_muestras)
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
    imprimir_resumen("RESUMEN FINAL DE CONFIGURACIÓN Y RESULTADOS (CUDA)", datos_resumen)

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


def parsear_argumentos() -> argparse.Namespace:
    """
    Parsea los mismos argumentos de linea de comandos que entrenar.py.

    :return: Objeto Namespace con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(description="Entrenamiento CBOW con kernels CUDA propios (softmax completo).")
    parser.add_argument("--config", type=str, default="configuracion_cuda.yaml",
                        help="Ruta al archivo YAML de configuracion (por defecto: configuracion_cuda.yaml).")
    parser.add_argument("--epocas", type=int, default=None, help="Sobrescribe la cantidad de epocas a entrenar.")
    parser.add_argument("--tasa", type=float, default=None, help="Sobrescribe la tasa de aprendizaje eta.")
    parser.add_argument("--reanudar", action="store_true", help="Activa el modo de reanudacion desde un checkpoint .npz.")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Ruta al archivo .npz desde el cual reanudar (requiere --reanudar).")
    return parser.parse_args()


def main() -> None:
    args = parsear_argumentos()

    print("==========================================================================")
    print("        ENTRENAMIENTO CBOW CON KERNELS CUDA - LINEA DE COMANDOS           ")
    print("==========================================================================")

    configuracion = cargar_configuracion(args.config)
    if args.epocas is not None:
        configuracion["cantidad_epocas"] = args.epocas
    if args.tasa is not None:
        configuracion["tasa_aprendizaje"] = args.tasa
    if args.reanudar:
        configuracion["reanudar_entrenamiento"] = True
    if args.checkpoint is not None:
        configuracion["ruta_checkpoint"] = args.checkpoint

    modelo = entrenar_cuda(configuracion)

    print("\n==========================================================================")
    print("                     ENTRENAMIENTO FINALIZADO CON ÉXITO                   ")
    print("==========================================================================")
    print(f"Pérdida final alcanzada: {modelo['historial_perdida'][-1]:.4f}")
    print(f"Accuracy final alcanzado: {modelo['historial_accuracy'][-1]:.4f}")
    print(f"Tiempo total transcurrido: {modelo['duracion_total_segundos']:.2f} s")
    print("==========================================================================")


if __name__ == "__main__":
    main()
