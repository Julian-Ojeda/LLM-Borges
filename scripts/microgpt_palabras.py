"""
Consigna 3 del TP2: microgpt.py de Karpathy (material/autograd/microgpt.py) adaptado para que el modelo de
lenguaje sea de PALABRAS del corpus del TP1 en lugar de letras de nombres.

Se mantiene todo el algoritmo de microgpt (autograd escalar con la clase Value, rmsnorm, atencion multi-cabeza
causal, MLP con ReLU, Adam con decaimiento lineal). Los cambios estan marcados con "# CAMBIO:".
    1. Datos: los tokens (palabras y signos) de una fraccion del comienzo del corpus, con el mismo tokenizador
       del TP1. Cada oracion (termina en . ? !) es un "documento" que empieza con BOS, y se corta en tramos de
       block_size + 1 tokens (block_size predicciones por tramo).
    2. Vocabulario: las palabras distintas de esa fraccion + BOS.
    3. wte: se inicializa con las representaciones W del CBOW del TP1 (época 254, d = 80), reducidas con PCA
       a n_embd = --dimension (por defecto 40; la consigna sugiere alrededor de 50) para que entrene en horas.
       BOS (y una palabra que no estuviera en el CBOW) recibe un vector al azar como los de microgpt.
    4. Validacion: el ultimo 10% de las oraciones de la fraccion no se usa para entrenar.
    5. Se guardan los pesos cada tanto (y se puede reanudar) y la perdida de cada paso en un CSV.
    6. La generacion une palabras con espacios.
La evaluacion (contadores top-5/10/20 y tablas de prediccion) esta en scripts/probar_microgpt.py, que lee
los pesos guardados y hace la misma cuenta con NumPy (mucho mas rapido que con Value).

Uso:
    python scripts/microgpt_palabras.py --fraccion 0.005 --bloque 8 --cabezas 4 --pasos 3000
    python scripts/microgpt_palabras.py ... --reanudar          # sigue desde el ultimo respaldo de la carpeta
"""

import argparse
import math
import random
import re
import sys
import time
from pathlib import Path
import numpy as np

ruta_raiz = Path(__file__).resolve().parent.parent
sys.path.append(str(ruta_raiz))
from codigo.tokenizador import tokenizar_corpus

# CAMBIO: los hiperparametros se pasan por linea de comandos
parser = argparse.ArgumentParser(description="microgpt adaptado a palabras del corpus del TP1.")
parser.add_argument("--fraccion", type=float, default=0.005, help="Fraccion del comienzo del corpus (0.01 = 1%%).")
parser.add_argument("--bloque", type=int, default=8, help="block_size N: tokens de contexto.")
parser.add_argument("--dimension", type=int, default=40, help="n_embd d: W del CBOW (80) se reduce con PCA a esta dimension.")
parser.add_argument("--cabezas", type=int, default=4, help="n_head H (d tiene que ser multiplo).")
parser.add_argument("--capas", type=int, default=1, help="n_layer.")
parser.add_argument("--pasos", type=int, default=3000, help="num_steps: un tramo de oracion por paso.")
parser.add_argument("--tasa", type=float, default=0.01, help="learning_rate inicial de Adam.")
parser.add_argument("--checkpoint", default="respaldos/modelo_cbow_w4_epoca_254.npz", help="CBOW del TP1 (de ahi sale wte).")
parser.add_argument("--directorio", default=None, help="Carpeta de salida (por defecto respaldos/microgpt/<parametros>).")
parser.add_argument("--cada", type=int, default=250, help="Guarda los pesos cada tantos pasos.")
parser.add_argument("--reanudar", action="store_true")
args = parser.parse_args()

random.seed(42) # Let there be order among chaos

# ===== 1. DATOS Y TOKENIZADOR =====

# CAMBIO: corpus del TP1 tokenizado en palabras y signos; se toma solo una fraccion del comienzo
tokens_corpus = tokenizar_corpus(ruta_raiz / "datos/corpus_ApAvAu.txt", True)
tokens_corpus = tokens_corpus[: int(len(tokens_corpus) * args.fraccion)]
print(f"tokens de la fraccion: {len(tokens_corpus)}")

# CAMBIO: el vocabulario son las palabras distintas (antes: las letras); BOS va al final
uchars = sorted(set(tokens_corpus))
palabra_a_id = {w: i for i, w in enumerate(uchars)}   # (uchars.index seria muy lento con miles de palabras)
BOS = len(uchars)
vocab_size = len(uchars) + 1
print(f"vocab size: {vocab_size}")

# CAMBIO: documentos = oraciones. Cada una empieza con BOS y termina con BOS despues del . ? !
# Las oraciones largas se cortan en tramos de block_size + 1 tokens que se pisan en 1 token,
# asi todas las palabras son objetivo alguna vez.
block_size = args.bloque
oraciones, actual = [], [BOS]
for w in tokens_corpus:
    actual.append(palabra_a_id[w])
    if w in (".", "?", "!"):
        oraciones.append(actual + [BOS])
        actual = [BOS]
corte = int(len(oraciones) * 0.9)          # CAMBIO: el ultimo 10% de las oraciones queda para validacion
def tramos(lista):
    return [o[i: i + block_size + 1] for o in lista for i in range(0, len(o) - 1, block_size)]
docs, docs_validacion = tramos(oraciones[:corte]), tramos(oraciones[corte:])
random.shuffle(docs)
print(f"num docs: {len(docs)} de entrenamiento, {len(docs_validacion)} de validacion")

# ===== 2. AUTOGRAD ESCALAR (sin cambios) =====

class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads')

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.grad = 0
        self._children = children
        self._local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

sys.setrecursionlimit(100000)   # CAMBIO: el grafo es mucho mas profundo que con nombres (build_topo es recursiva)

# ===== 3. PARAMETROS =====

# CAMBIO: n_embd sale del CBOW del TP1 (d = 80, o menos con --dimension)
cbow = np.load(ruta_raiz / args.checkpoint, allow_pickle=True)   # respaldo propio (el vocabulario es un arreglo de objetos)
W_cbow = cbow["W"]
if args.dimension and args.dimension < W_cbow.shape[1]:
    # CAMBIO: W (d = 80) se reduce a las `dimension` direcciones de mayor varianza (PCA): se centra W y se
    # proyecta sobre los primeros vectores singulares. Conserva lo mas posible de las distancias entre palabras.
    W_centrada = W_cbow - W_cbow.mean(axis=0)
    _, valores_singulares, Vt = np.linalg.svd(W_centrada, full_matrices=False)
    varianza = (valores_singulares[:args.dimension] ** 2).sum() / (valores_singulares ** 2).sum()
    W_cbow = W_centrada @ Vt[:args.dimension].T
    print(f"W del CBOW reducida de {W_centrada.shape[1]} a {args.dimension} dimensiones (PCA, {varianza:.1%} de la varianza)")
indice_cbow = {str(p): i for i, p in enumerate(cbow["vocabulario_palabras"])}
n_layer = args.capas
n_embd = W_cbow.shape[1]
n_head = args.cabezas
assert n_embd % n_head == 0, f"n_embd = {n_embd} tiene que ser multiplo de n_head = {n_head}"
head_dim = n_embd // n_head

matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
# CAMBIO: cada fila de wte toma el vector de esa palabra en el CBOW (BOS queda con el vector al azar)
sin_cbow = 0
for palabra, i in palabra_a_id.items():
    if palabra in indice_cbow:
        state_dict['wte'][i] = [Value(float(x)) for x in W_cbow[indice_cbow[palabra]]]
    else:
        sin_cbow += 1
print(f"wte: {vocab_size - 1 - sin_cbow} filas copiadas del CBOW, {sin_cbow} palabras + BOS al azar")
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)
params = [p for mat in state_dict.values() for row in mat for p in row]
print(f"num params: {len(params)}")

# ===== 4. MODELO (sin cambios) =====

def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]

def softmax(logits):
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    tok_emb = state_dict['wte'][token_id]
    pos_emb = state_dict['wpe'][pos_id]
    x = [t + p for t, p in zip(tok_emb, pos_emb)]
    x = rmsnorm(x)
    for li in range(n_layer):
        x_residual = x
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq'])
        k = linear(x, state_dict[f'layer{li}.attn_wk'])
        v = linear(x, state_dict[f'layer{li}.attn_wv'])
        keys[li].append(k)
        values[li].append(v)
        x_attn = []
        for h in range(n_head):
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])
        x = [a + b for a, b in zip(x, x_residual)]
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])
        x = [xi.relu() for xi in x]
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])
        x = [a + b for a, b in zip(x, x_residual)]
    logits = linear(x, state_dict['lm_head'])
    return logits

# ===== 5. ENTRENAMIENTO =====

learning_rate, beta1, beta2, eps_adam = args.tasa, 0.85, 0.99, 1e-8
m = [0.0] * len(params)
v = [0.0] * len(params)
num_steps = args.pasos

# CAMBIO: carpeta de salida, respaldos periodicos (pesos + buffers de Adam) y CSV con la perdida de cada paso
nombre = f"f{args.fraccion:g}_d{n_embd}_N{block_size}_H{n_head}_L{n_layer}_pasos{num_steps}"
directorio = ruta_raiz / (args.directorio or f"respaldos/microgpt/{nombre}")
directorio.mkdir(parents=True, exist_ok=True)
ruta_pesos, ruta_perdidas = directorio / "microgpt_palabras.npz", directorio / "perdidas.csv"

def guardar(paso_hecho):
    """Pesos como matrices de NumPy, vocabulario como texto (se lee sin pickle) y estado de Adam."""
    datos = {nombre: np.array([[p.data for p in fila] for fila in mat]) for nombre, mat in state_dict.items()}
    datos.update(adam_m=np.array(m), adam_v=np.array(v), paso=np.array(paso_hecho),
                 vocabulario=np.array(uchars + ["<BOS>"]),
                 hiperparametros=np.array(repr({"fraccion": args.fraccion, "block_size": block_size, "n_head": n_head,
                                                "n_layer": n_layer, "n_embd": n_embd, "num_steps": num_steps,
                                                "learning_rate": learning_rate, "checkpoint": args.checkpoint})))
    temporal = directorio / "microgpt_palabras_tmp.npz"
    np.savez(temporal, **datos)
    temporal.replace(ruta_pesos)       # se reemplaza de una vez: un corte de luz no deja el respaldo a medias

paso_inicial = 0
if args.reanudar and ruta_pesos.exists():
    guardado = np.load(ruta_pesos)     # sin allow_pickle: solo numeros y texto
    for nombre_mat, mat in state_dict.items():
        for fila, valores in zip(mat, guardado[nombre_mat]):
            for p, x in zip(fila, valores):
                p.data = float(x)
    m, v = guardado["adam_m"].tolist(), guardado["adam_v"].tolist()
    paso_inicial = int(guardado["paso"])
    print(f"reanudado desde el paso {paso_inicial}")
elif ruta_perdidas.exists():
    raise FileExistsError(f"'{directorio}' ya tiene un entrenamiento. Use --reanudar u otra carpeta.")
if paso_inicial == 0:
    ruta_perdidas.write_text("paso,perdida,segundos\n", encoding="utf-8")

inicio = time.time()
for step in range(paso_inicial, num_steps):
    # CAMBIO: el documento ya viene tokenizado (un tramo de oracion con BOS)
    tokens = docs[step % len(docs)]
    n = min(block_size, len(tokens) - 1)

    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = (1 / n) * sum(losses)

    loss.backward()

    lr_t = learning_rate * (1 - step / num_steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0

    segundos = time.time() - inicio
    with open(ruta_perdidas, "a", encoding="utf-8") as archivo:
        archivo.write(f"{step + 1},{loss.data:.6f},{segundos:.1f}\n")
    faltan = segundos / (step + 1 - paso_inicial) * (num_steps - step - 1) / 3600
    print(f"step {step+1:5d} / {num_steps:5d} | loss {loss.data:.4f} | faltan ~{faltan:.1f} h", end='\r', flush=True)
    if (step + 1) % args.cada == 0 or step + 1 == num_steps:
        guardar(step + 1)

# ===== 6. INFERENCIA =====

temperature = 0.5
print("\n--- inference (oraciones nuevas) ---")
for sample_idx in range(10):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {' '.join(sample)}")   # CAMBIO: palabras unidas por espacios
print(f"\npesos en: {ruta_pesos}")
