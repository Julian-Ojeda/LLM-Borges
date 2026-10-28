# Código de AutpGrad escalar
# Aprendizaje automatico avanzado, UNAHUR
import math
import random
from graphviz import Digraph

import os
os.environ["PATH"] += os.pathsep + r"C:\Program Files\Graphviz\bin"

class Var:
    def __init__(self, data: int, nombre: str, _hijos = (), _op: str = ""):
        self.data = data
        self.nombre = nombre
        self._hijos = set(_hijos)
        self._op = _op
        # Agregado para Autograd
        self.grad = 0

    def __repr__(self) -> str:
        return f"{self.nombre} = {self.data}"
    
    def __add__(self, otro_operando):
        # Convertir el otro_operando a un objeto Var si otro_operando no es un objeto Var
        otro_operando = otro_operando if isinstance(otro_operando, Var) else Var(
            data = otro_operando, nombre = str(otro_operando))
        salida = Var(
            data = self.data + otro_operando.data,
            nombre = "", _hijos = (self, otro_operando), _op = "+")
        return salida 

    # Agregado para Autograd
    def _retroceso(self):
        if self._op == "+":
            list(self._hijos)[0].grad += self.grad * 1
            list(self._hijos)[1].grad += self.grad * 1
            informar(list(self._hijos)[0], self, list(self._hijos)[1])
        if self._op == "*":
            list(self._hijos)[0].grad += self.grad * list(self._hijos)[1].data
            list(self._hijos)[1].grad += self.grad * list(self._hijos)[0].data
            informar(list(self._hijos)[0], self, list(self._hijos)[1])
        if '**' in self._op:
            list(self._hijos)[0].grad += self.grad * float(self._op[2:]) * list(self._hijos)[0].data**(float(self._op[2:]) - 1)
            informar(list(self._hijos)[0], self)
        if self._op == "exp":
            list(self._hijos)[0].grad += self.grad * math.exp(list(self._hijos)[0].data)
            informar(list(self._hijos)[0], self)
        if self._op == "tanh":
            list(self._hijos)[0].grad += self.grad * (1 - math.tanh(list(self._hijos)[0].data) ** 2)
            informar(list(self._hijos)[0], self)
        if self._op == "relu":
            list(self._hijos)[0].grad += self.grad * (0 if list(self._hijos)[0].data < 0 else 1)
            informar(list(self._hijos)[0], self)

    def __radd__(self, otro_operando):
        # Esto será otro_operando + self 
        return self + otro_operando
    
    def __neg__(self):
        return self * -1

    # Substraction 
    # (Special case of addition and negation)
    
    def __sub__(self, otro_operando): 
        return self + (-otro_operando)

    def __rsub__(self, otro_operando): 
        return otro_operando + (-self)

    # Multiplication 
    def __mul__(self, otro_operando):
        otro_operando = otro_operando if isinstance(otro_operando, Var) else Var(
            data = otro_operando,
            nombre = str(otro_operando)
        )
        salida = Var(
            data = self.data * otro_operando.data,
            nombre="",
            _hijos=(self, otro_operando),
            _op="*"
        )
        return salida

    def __rmul__(self, otro_operando): # otro_operando * self
        return self * otro_operando

    # Power
    def __pow__(self, otro_operando):
        salida = Var(
            data = self.data**otro_operando,
            nombre = "",
            _hijos = (self,),
            _op = f'**{otro_operando}'
        )
        return salida 

    # Division  (Special case of division and ^-1)
    def __truediv__(self, otro_operando): 
        return self * otro_operando**-1

    def __rtruediv__(self, otro_operando):
        return otro_operando * self**-1
    
    # Activation functions
    # Exponent function
    def exp(self):
        salida = Var(data = math.exp(self.data), nombre = f"exp({self.data})",
                    _hijos = (self,), _op="exp")
        return salida

    # Tanh function
    def tanh(self):
        resultado = (math.exp(2 * self.data) - 1) / (math.exp(2 * self.data) + 1)
        salida = Var(resultado, nombre = f"tanh({self.nombre})", _op = "tanh",
                    _hijos = (self, ))
        return salida

    # ReLU or Rectified Linear Unit function
    def relu(self):
        resultado = 0 if self.data <= 0 else self.data
        salida = Var(resultado, nombre = f"relu({self.nombre})", _op = "relu", _hijos = (self, ))
        return salida
    
# Crear conjuntos de nodos y arcos
def obtener_nodos_y_arcos(root: Var):
    # Inicializar los conjuntos de arcos y nodos
    nodos, arcos = set(), set()
    # Definir la función recursiva
    def insertar(nodo_actual: Var):
        # Si el nodo no fue visitado, agregarlo
        if nodo_actual not in nodos:
            nodos.add(nodo_actual)
            # Iterar sobre los hijos
            for hijo in nodo_actual._hijos:
                arcos.add((hijo, nodo_actual))
                # Hacer recursión sobre cada hijo
                insertar(hijo)

    # LLmara a la función insertar sobre el nodo raíz
    insertar(root)
    return nodos, arcos

# Dibujar grafo indicando la raíz
def DibujarGrafo(root: Var):
    dot = Digraph(format = "png", graph_attr = {'rankdir': 'LR'})
    nodos, arcos = obtener_nodos_y_arcos(root)
    # Iterar sobre todos los nodos
    for nodo in nodos:
        # Definir la variable nodo_uuid que usa graphviz
        nodo_uuid = str(id(nodo))
        # Dibujar el nodo
        dot.node(name = nodo_uuid, label = "{ %s | data %.2f | grad %.2f}" % (nodo.nombre, nodo.data, nodo.grad),
                 shape = "record")
        # Si el nodo tiene una operación asociada a él
        # se dibuja un nodo adicional (auxiliar) donde está explicitada 
        # la operación y un arco desde el nodo actual y el de 
        # la operación
        if nodo._op:
            dot.node(name = nodo_uuid + nodo._op, label = f"{nodo._op}")
            dot.edge(tail_name = nodo_uuid + nodo._op, head_name = nodo_uuid)
    # Iterar sobre todos los arcos usando los nodos que vinculan cada arco
    for nodo_1, nodo_2 in arcos:
        dot.edge(tail_name = str(id(nodo_1)), head_name = str(id(nodo_2)) + nodo_2._op)
    #dot.format = 'png'
    #dot.view()
    dot.render('Graph', view = True)    
    return dot

def informar(hijo_0: Var, ppal: Var, hijo_1: Var = None):
    if ppal._op == '+' or ppal._op == '*' or ppal._op == 'tanh':
        map_ = {"+": "Suma", "*": "Multiplicación", "tanh": "Tanh"}
        if hijo_1:
            print(f"{ppal.nombre} como {map_[ppal._op]} entre {hijo_0.nombre} y {hijo_1.nombre }")
        else:
            print(f"{ppal.nombre} como {map_[ppal._op]} ({hijo_0.nombre})")
    else: 
        print(f"Ir para atrás desde {ppal.nombre}")
    print("-----------------")
    print(f"{ppal.nombre} | Grad:", ppal.grad)
    print(f"{hijo_0.nombre} | Grad: ", hijo_0.grad)
    if hijo_1 is not None:
        print(f"{hijo_1.nombre} | Grad: ", hijo_1.grad)
    print(" ")
def obtener_ordenamiento_topologico(root: Var):
    # Obtener el ordenamiento topológico para los nodos en el GDA
    topo = []
    visitados = set()
    # Definir 
    def construir_ordenamiento_topologico(v):
        # Add the current node v if it is not in the set
        if v not in visitados:
            visitados.add(v)
            # Hacer recursión sobre cada uno de los hijos
            # hasta encontrar un nodo hoja
            # donde no se aplicará la recursión
            # y se devolverá una lista
            for hijo in v._hijos:
                construir_ordenamiento_topologico(hijo)
            topo.append(v)
    construir_ordenamiento_topologico(root)
    return topo


def retroceso(self):
    # Construir el ordenamiento topológico de acuerdo al GDA
    topo = []
    visitados = set()
    def construir_ordenamiento_topologico(v):
        if v not in visitados:
            visitados.add(v)
            for hijo in v._hijos:
                construir_ordenamiento_topologico(hijo)
            topo.append(v)
    construir_ordenamiento_topologico(self)
    # Invertir la lista topo y proceder
    # hacia atrás un nodo a la vez aplicando la regla de la cadena
    # para obtener su gradiente
    self.grad = 1
    for v in reversed(topo):
        v._retroceso()

# # Programa 1
# a = Var(3, nombre = "a")
# b = a * 3
# b.nombre = "b"
# c = b.exp()
# c.nombre = "c"
# # Calcular los gradientes
# c.grad = 1.0
# # Dibujar el grafo
# DibujarGrafo(c)
# c._retroceso()
# # Dibujar el grafo
# DibujarGrafo(c)

# # Programa 2
# # Entradas
# x1 = Var(2.0, "x1")
# x2 = Var(0.0, "x2")
# # Pesos
# w1 = Var(-3.0, "w1")
# w2 = Var(1.0, "w2")
# b = Var(6.8813735870195432, "b")
# # Multiplicaciones y suma
# w1x1 = w1 * x1
# w1x1.nombre = "w1 * x1"
# w2x2 = w2 * x2
# w2x2.nombre = "w2 * x2"
# x1w1x2w2 = w1x1 + w2x2
# x1w1x2w2.nombre = "x1w1 + x2w2"
# z = x1w1x2w2 + b
# z.nombre = "z"
# # Salida
# salida = z.tanh()
# salida.nombre = "salida"

# # Dibujar el grafo
# DibujarGrafo(salida)

# salida.grad = 1
# # Pasos hacia atrás
# salida._retroceso()
# z._retroceso()
# x1w1x2w2._retroceso()
# w2x2._retroceso()
# w1x1._retroceso()
# b._retroceso()
# w2._retroceso()
# w1._retroceso()
# x2._retroceso()
# x1._retroceso()

# # Dibujar el grafo
# DibujarGrafo(salida)

# # Programa 3
# a = Var(2, nombre = "a")
# b = Var(3, nombre = "b")
# c = a + b
# c.nombre = "c"
# d = a * c
# d.nombre = "d"

# # Dibujar el grafo
# DibujarGrafo(d)
# d.grad = 1.0
# d._retroceso()
# c._retroceso()
# b._retroceso()
# a._retroceso()

# # Dibujar el grafo
# DibujarGrafo(d)

# # Programa 4
# a = Var(2, nombre = "a")
# b = Var(3, nombre = "b")
# e = Var(10, nombre = "e")
# c = a + b
# c.nombre = "c"
# d = a * c + e
# d.nombre = "d"
# # Retropropagar
# d.grad = 1.0
# d._retroceso()
# a._retroceso()
# c._retroceso()
# DibujarGrafo(d)

# # Programa 5 (igual que el 4 pero usando retroceso
# # la función retroceso asegura calcular todos las componentes del gradiente
# # ya que en el progema 4 omitía calcular el grad de e y de a * c como nodo 
# a = Var(2, nombre = "a")
# b = Var(3, nombre = "b")
# e = Var(10, nombre = "e")
# c = a + b
# c.nombre = "c"
# d = a * c + e
# d.nombre = "d"
# # Retropropagar
# retroceso(d)
# DibujarGrafo(d)

# # Programa 6
# # Entradas x1 y x2
# x1 = Var(2.0, "x1")
# x2 = Var(0.0, "x2")
# # Pesos
# w1 = Var(-3.0, "w1")
# w2 = Var(1.0, "w2")
# b = Var(6.88, "b")
# # Operaciones de multiplicación y suma
# w1x1 = w1 * x1
# w1x1.nombre = "w1 * x1"
# w2x2 = w2 * x2
# w2x2.nombre = "w2 * x2"
# x1w1x2w2 = w1x1 + w2x2
# x1w1x2w2.nombre = "x1w1 + x2w2"
# z = x1w1x2w2 + b
# z.nombre = "z"
# # Salida
# salida = z.tanh()
# salida.nombre = "salida"

# retroceso(salida)

# # Dibujar el grafo
# DibujarGrafo(salida)

# # Programa 7
# # Entradas x1 y x2
# x1 = Var(2.0, "x1")
# x2 = Var(0.0, "x2")
# # Pesos
# w1 = Var(-3.0, "w1")
# w2 = Var(1.0, "w2")
# b = Var(6.88, "b")
# # Operaciones de multiplicación y suma
# z = w1 * x1 + w2 * x2 + b
# z.nombre = "z"
# # Salida
# salida = z.tanh()
# salida.nombre = "salida"
# retroceso(salida)

# # Dibujar el grafo
# DibujarGrafo(salida)

# Programa 8
# Entrada
def InicializarPesosYUmbrales(pesos_1, pesos_2, b_1, b_2):
    for j in range(2):
        for k in range(3):
            pesos_1[j][k].data = random.random() - 0.5
    for i in range(2):
            pesos_2[i].data = random.random() - 0.5
    for i in range(2):
        b_1[i].data = random.random() - 0.5
    b_2.data = random.random() - 0.5
    return

def PonerEntrada(entrada, valores):
    for k in range(3):
        entrada[k].data = valores[k]
    return

def Mult(v_1, v_2):
    aux = Var(0, "aux")
    for i in range(len(v_1)):
        aux = aux + (v_1[i] * v_2[i])
    return aux
    
# Inicializar variables
entrada = []
for k in range(3):
    entrada.append(Var(0, "x" + str(k)))
    
W = []
for j in range(2):
    W.append([])
    for k in range(3):
        W[j].append(Var(0, "W" + str(j) + str(k)))

U = []
for j in range(2):
    U.append(Var(0, "U" + str(j)))

B = []
for j in range(2):
    B.append(Var(0, "B" + str(j)))

V = []
for j in range(2):
    V.append(Var(0, "V" + str(j)))

w = []
for i in range(2):
    w.append(Var(0, "w" + str(i)))

b = Var(0, "b")

u = Var(0, "u")
    
o = Var(0, "o")

InicializarPesosYUmbrales(W, w, B, b)

# Paso hacia adelante
#PonerEntrada(entrada, [0.5, 1.2, -0.7])
PonerEntrada(entrada, [1, 1, 1])
for j in range(2):
    U[j] = Mult(entrada, W[j])
    U[j].nombre = "U" + str(j)
    V[j] = (U[j] + B[j]).relu()
    V[j].nombre = "V" + str(j)
u = Mult(V, w)
u.nombre = "u"
o = (u + b).relu()
o.nombre = "o"

print(o)

# Calcular gradientes
retroceso(o)

# Dibujar el grafo
DibujarGrafo(o)
 