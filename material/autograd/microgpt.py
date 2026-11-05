"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy

------------------------------------------------------------------------------------------
Resumen (anotaciones en español)

Que hace: entrena un GPT chiquito que aprende a generar nombres letra por letra. Cada "documento" es
un nombre ("emma"), cada token es una letra, y la tarea es predecir la letra siguiente dada las anteriores.
Despues del entrenamiento, genera nombres nuevos letra por letra.

Partes, en el orden del archivo:
    1. Datos y tokenizador: letras <-> numeros enteros (ids).
    2. Autograd escalar (clase Value): cada numero recuerda de donde salio, para despues aplicar
       la regla de la cadena hacia atras y obtener el gradiente de la perdida respecto de cada parametro.
    3. Parametros: embeddings de letras y de posiciones, y por cada capa las matrices de atencion y del MLP.
    4. Modelo (funcion gpt): embedding -> [atencion multi-cabeza + MLP] x n_layer -> logits.
    5. Entrenamiento con Adam: perdida = entropia cruzada de predecir la letra siguiente.
    6. Inferencia: generar nombres muestreando de la softmax.

Notacion de la catedra (distinta de la del CBOW del TP, donde N era la dimension del embedding):
    d = n_embd (dimension del embedding), N = block_size (largo maximo de contexto),
    H = n_head (cabezas), dk = dv = head_dim = d / H.

Diferencia con el CBOW + atencion del TP: aca la atencion es causal (cada letra solo ve las anteriores,
porque las claves y valores se van agregando de a una) y el modelo es generativo (predice la siguiente).
En el CBOW + atencion la palabra central ve el contexto de los dos lados y no se genera texto.
------------------------------------------------------------------------------------------
"""

import os       # os.path.exists
import math     # math.log, math.exp
import random   # random.seed, random.choices, random.gauss, random.shuffle

random.seed(42) # Let there be order among chaos
                # Semilla fija: los pesos iniciales, el orden de los datos y los nombres generados se repiten

# ===== 1. DATOS Y TOKENIZADOR =====

# Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
# Si no existe input.txt, se descarga la lista de nombres de makemore (un nombre por linea)
if not os.path.exists('input.txt'):
    import urllib.request
    names_url = 'https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt'
    urllib.request.urlretrieve(names_url, 'input.txt')
docs = [line.strip() for line in open('input.txt') if line.strip()]   # un nombre por linea, sin lineas vacias
docs = docs[0 : 100]    # solo los primeros 100 nombres, para que el entrenamiento escalar termine rapido
random.shuffle(docs)    # se mezclan para no entrenar siempre en el mismo orden alfabetico
print(f"num docs: {len(docs)}")

# Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
uchars = sorted(set(''.join(docs))) # unique characters in the dataset become token ids 0..n-1
                                    # Ejemplo: ['a', 'b', ..., 'z'] -> 'a' es el id 0, 'b' el 1, ...
BOS = len(uchars) # token id for a special Beginning of Sequence (BOS) token
                  # BOS marca el principio y el final de cada nombre (el modelo aprende a "cortar" en BOS)
vocab_size = len(uchars) + 1 # total number of unique tokens, +1 is for BOS
print(f"vocab size: {vocab_size}")

# ===== 2. AUTOGRAD ESCALAR =====

# Let there be Autograd to recursively apply the chain rule through a computation graph
# Cada operacion entre Values crea un Value nuevo que guarda:
#   - su resultado (data),
#   - de que Values salio (_children),
#   - la derivada local del resultado respecto de cada hijo (_local_grads).
# Ejemplo: c = a * b  ->  c._children = (a, b), c._local_grads = (b.data, a.data)  porque dc/da = b, dc/db = a
class Value:
    __slots__ = ('data', 'grad', '_children', '_local_grads') # Python optimization for memory usage

    def __init__(self, data, children=(), local_grads=()):
        self.data = data                # scalar value of this node calculated during forward pass
        self.grad = 0                   # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children       # children of this node in the computation graph
        self._local_grads = local_grads # local derivative of this node w.r.t. its children

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)   # permite sumar un numero comun: v + 2
        return Value(self.data + other.data, (self, other), (1, 1))   # d(a+b)/da = 1, d(a+b)/db = 1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))   # d(ab)/da = b, d(ab)/db = a

    # Derivadas locales de cada operacion elemental:
    def __pow__(self, other): return Value(self.data**other, (self,), (other * self.data**(other-1),))   # d(a^n)/da = n a^(n-1)
    def log(self): return Value(math.log(self.data), (self,), (1/self.data,))                          # d(log a)/da = 1/a
    def exp(self): return Value(math.exp(self.data), (self,), (math.exp(self.data),))                  # d(e^a)/da = e^a
    def relu(self): return Value(max(0, self.data), (self,), (float(self.data > 0),))                  # 1 si a > 0, 0 si no
    # El resto se arma con las de arriba (resta = suma con el opuesto, division = multiplicar por a^-1, ...)
    def __neg__(self): return self * -1
    def __radd__(self, other): return self + other      # 2 + v  (el numero comun a la izquierda)
    def __sub__(self, other): return self + (-other)
    def __rsub__(self, other): return other + (-self)
    def __rmul__(self, other): return self * other
    def __truediv__(self, other): return self * other**-1
    def __rtruediv__(self, other): return other * self**-1

    def __repr__(self) -> str:
        return f"{self.data}"

    def backward(self):
        # 1. Orden topologico: lista de todos los nodos del grafo, cada uno despues de sus hijos
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        # 2. Regla de la cadena desde la perdida hacia atras:
        #    d(perdida)/d(perdida) = 1, y cada nodo le pasa a sus hijos  grad_hijo += derivada_local * grad_nodo.
        #    Se usa += porque un nodo puede influir en la perdida por varios caminos (y se suman)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad

# ===== 3. PARAMETROS =====

# Initialize the parameters, to store the knowledge of the model
n_layer = 1     # depth of the transformer neural network (number of layers)
n_embd = 16     # width of the network (embedding dimension). Es lo que llamamos d
block_size = 16 # maximum context length of the attention window
                # (note: the longest name is 15 characters). Es lo que llamamos N
n_head = 4      # number of attention heads. Es lo que llamamos H
head_dim = n_embd // n_head # derived dimension of each head.
                            # Es lo que llamamos kv y aquí, se toma kd = kv

# matrix(nout, nin): matriz de nout filas x nin columnas de Values al azar, normal con media 0 y desvio 0.08
matrix = lambda nout, nin, std=0.08: [[Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)]
# wte: embedding de cada token (una fila por letra)        (vocab_size x d)
# wpe: embedding de cada posicion 0..N-1                    (N x d)
# lm_head: proyeccion final de d a un logit por token       (vocab_size x d)
state_dict = {'wte': matrix(vocab_size, n_embd), 'wpe': matrix(block_size, n_embd), 'lm_head': matrix(vocab_size, n_embd)}
# attn_wq debería ser de d X dk pero, tomando dk como head_dim
# attn_wq se la puedde hacer de n_embd X n_embd donde estarán
# concatenadas las matrices WQ para cada cabezal
# Lo mismo pasa con WK y WV
# En esta implementación dk = dv = d / H, donde H es el número de cabezales
for i in range(n_layer):
    state_dict[f'layer{i}.attn_wq'] = matrix(n_embd, n_embd)    # consultas (queries)
    state_dict[f'layer{i}.attn_wk'] = matrix(n_embd, n_embd)    # claves (keys)
    state_dict[f'layer{i}.attn_wv'] = matrix(n_embd, n_embd)    # valores (values)
    state_dict[f'layer{i}.attn_wo'] = matrix(n_embd, n_embd)    # mezcla la salida concatenada de las H cabezas
    state_dict[f'layer{i}.mlp_fc1'] = matrix(4 * n_embd, n_embd)   # MLP: d -> 4d
    state_dict[f'layer{i}.mlp_fc2'] = matrix(n_embd, 4 * n_embd)   # MLP: 4d -> d
params = [p for mat in state_dict.values() for row in mat for p in row] # flatten params into a single list[Value]
                                                                        # (lista plana: Adam recorre esta lista)
print(f"num params: {len(params)}")

# ===== 4. MODELO =====

# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x, w):
    # Corresponde a multiplicar la matriz w por el vector x donde
    # si w es de A x B, x es de B x 1
    # Da como resultado un vector de A x 1
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]
    # Alternativamente (sin usar zip) se podría haber escrito:
    # return [sum(x[i] * wo[i] for i in range(len(x))) for wo in w]

def softmax(logits):
    # Convierte puntajes en probabilidades positivas que suman 1: exp(l_i) / sum_j exp(l_j).
    # Se resta el maximo antes del exp para no desbordar (el resultado no cambia)
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]

def rmsnorm(x):
    # Es una alternativa de normalización a la de LayerNormalization
    # Se eleva al cuadrado cada uno de los componentes del vector
    # Se suman todos los cuadrados obtenidos y se lo divide por la cantidad
    # A la suma se le calcula la raíz cuadrada y con este valor (rms)
    # se divide cada uno de los componentes del vector
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5     # 1 / raiz(ms); el 1e-5 evita dividir por 0
    return [xi * scale for xi in x]

def gpt(token_id, pos_id, keys, values):
    # Procesa UN token (el de la posicion pos_id) y devuelve los logits de cual sera el siguiente.
    # keys y values son la "memoria" de las posiciones anteriores (una lista por capa): cada llamada
    # agrega la clave y el valor de este token, asi la atencion solo puede mirar hacia atras (causal).
    tok_emb = state_dict['wte'][token_id] # token embedding
    pos_emb = state_dict['wpe'][pos_id] # position embedding
    x = [t + p for t, p in zip(tok_emb, pos_emb)] # joint token and position embedding
    x = rmsnorm(x) # note: not redundant due to backward pass via the residual connection

    for li in range(n_layer):
        # 1) Multi-head Attention block
        x_residual = x      # se guarda la entrada para la conexion residual (x + atencion(x))
        x = rmsnorm(x)
        q = linear(x, state_dict[f'layer{li}.attn_wq']) # WQ * x
        k = linear(x, state_dict[f'layer{li}.attn_wk']) # WK * x
        v = linear(x, state_dict[f'layer{li}.attn_wv']) # WV * x
        keys[li].append(k)      # la clave y el valor de este token quedan disponibles
        values[li].append(v)    # para él y para los tokens que vengan despues
        x_attn = []
        for h in range(n_head):
            # Cada cabeza trabaja con su porcion de head_dim componentes de q, k y v
            hs = h * head_dim
            q_h = q[hs : hs + head_dim]
            k_h = [ki[hs : hs + head_dim] for ki in keys[li]]     # claves de todas las posiciones vistas (0..pos_id)
            v_h = [vi[hs : hs + head_dim] for vi in values[li]]
            # Puntaje de atencion con cada posicion t: q_h . k_t / raiz(dk)
            attn_logits = [sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5 for t in range(len(k_h))]
            attn_weights = softmax(attn_logits)     # pesos de atencion: positivos y suman 1
            # Salida de la cabeza: promedio de los valores ponderado por los pesos
            head_out = [sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h))) for j in range(head_dim)]
            x_attn.extend(head_out)
            # x_attn.extend(head_out) es equivalente a
            # x_attn = x_attn + head_out
        x = linear(x_attn, state_dict[f'layer{li}.attn_wo'])   # mezcla las H cabezas concatenadas
        x = [a + b for a, b in zip(x, x_residual)]             # conexion residual
        # 2) MLP block
        x_residual = x
        x = rmsnorm(x)
        x = linear(x, state_dict[f'layer{li}.mlp_fc1'])   # d -> 4d
        x = [xi.relu() for xi in x]                       # no linealidad
        x = linear(x, state_dict[f'layer{li}.mlp_fc2'])   # 4d -> d
        x = [a + b for a, b in zip(x, x_residual)]        # conexion residual

    logits = linear(x, state_dict['lm_head'])   # un puntaje por token del vocabulario
    return logits

# ===== 5. ENTRENAMIENTO =====

# Let there be Adam, the blessed optimizer and its buffers
# learning_rate: tasa inicial; beta1, beta2: cuanto "recuerdan" los promedios de m y v; eps_adam: evita dividir por 0
learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
m = [0.0] * len(params) # first moment buffer    (promedio movil del gradiente: la direccion)
v = [0.0] * len(params) # second moment buffer   (promedio movil del gradiente al cuadrado: la escala)

# Repeat in sequence
num_steps = 1000 # number of training steps
for step in range(num_steps):

    # Take single document, tokenize it, surround it with BOS special token on both sides
    # Ejemplo: "ana" -> [BOS, a, n, a, BOS]. Pares (entrada -> objetivo): BOS->a, a->n, n->a, a->BOS
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)    # cantidad de predicciones (sin pasarse de block_size posiciones)

    # Forward the token sequence through the model, building up the computation graph all the way to the loss
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]   # memoria vacia al empezar cada nombre
    losses = []
    for pos_id in range(n):
        token_id, target_id = tokens[pos_id], tokens[pos_id + 1]   # entrada: este token; objetivo: el siguiente
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()    # entropia cruzada: -log de la probabilidad del token correcto
        losses.append(loss_t)
    loss = (1 / n) * sum(losses) # final average loss over the document sequence. May yours be low.

    # Backward the loss, calculating the gradients with respect to all model parameters
    loss.backward()     # llena p.grad de cada parametro

    # Adam optimizer update: update the model parameters based on the corresponding gradients
    lr_t = learning_rate * (1 - step / num_steps) # linear learning rate decay
                                                  # (baja de 0.01 a 0 a lo largo del entrenamiento)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad ** 2
        # Correccion de sesgo: m y v arrancan en 0 y al principio subestiman; se dividen por (1 - beta^t)
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        # Paso: direccion m_hat dividida por la escala tipica del gradiente (cada parametro se mueve ~lr_t)
        p.data -= lr_t * m_hat / (v_hat ** 0.5 + eps_adam)
        p.grad = 0      # el gradiente se pone en 0 para el proximo paso (si no, se acumularia)

    print(f"step {step+1:4d} / {num_steps:4d} | loss {loss.data:.4f}", end='\r')

# ===== 6. INFERENCIA =====

# Inference: may the model babble back to us
temperature = 0.5 # in (0, 1], control the "creativity" of generated text, low to high
                  # (los logits se dividen por la temperatura: menor => la letra mas probable sale mas seguido)
print("\n--- inference (new, hallucinated names) ---")
for sample_idx in range(20):
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    token_id = BOS      # se arranca con BOS: "empieza un nombre"
    sample = []
    for pos_id in range(block_size):
        logits = gpt(token_id, pos_id, keys, values)
        probs = softmax([l / temperature for l in logits])
        # Se sortea la letra siguiente segun las probabilidades (no siempre la mas probable)
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:     # BOS = el modelo decidio que el nombre termino
            break
        sample.append(uchars[token_id])
    print(f"sample {sample_idx+1:2d}: {''.join(sample)}")
