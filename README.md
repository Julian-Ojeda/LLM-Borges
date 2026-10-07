# TP1: Representación Contextual de Palabras con CBOW

**Asignatura**: Aprendizaje Automático Avanzado  
**Carrera**: Tecnicatura Universitaria en Inteligencia Artificial (UNAHUR)  
**Profesor**: Juan Miguel Santos (LIDEC)  

Este repositorio contiene la implementación del modelo de redes neuronales **CBOW (Continuous Bag-of-Words)** para Procesamiento de Lenguaje Natural (PLN), acelerado por GPU NVIDIA mediante **Python, CuPy y NumPy**.

---

## 🚀 Instalación y Configuración del Entorno con `uv`

El proyecto utiliza el gestor de paquetes de alto rendimiento **`uv`**.

### 1. Clonar el Repositorio
```bash
git clone https://github.com/opablon/cbow_nlp.git
cd cbow_nlp
```

### 2. Crear e Instalar el Entorno Virtual
```bash
# Crear el entorno virtual con uv
uv venv .venv

# Activar el entorno virtual (Linux/macOS)
source .venv/bin/activate

# Activar el entorno virtual (Windows PowerShell)
# .venv\Scripts\activate

# Instalar las dependencias declaradas en pyproject.toml
uv pip install -e .
```

---

## 📁 Organización del Proyecto

```text
cbow_nlp/
├── README.md                               # Instrucciones de instalacion, configuracion y uso
├── pyproject.toml                          # Dependencias del proyecto para uv
├── configuracion.yaml                      # Hiperparametros centralizados por defecto
├── entrenar.py                             # Script ejecutable por consola para entrenamiento local
├── TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb  # Notebook interactivo para analisis y evaluacion
└── codigo/                                 # Modulos Python reutilizables
    ├── __init__.py                         # Identificador del paquete de Python
    ├── configuracion.py                    # Carga y guardado de hiperparametros YAML
    ├── tokenizador.py                      # Tokenizacion por Palabra (Regex)
    ├── generador_vocabulario.py            # Construccion del vocabulario (indices one-hot)
    ├── modelo_cbow_cupy.py                 # Red CBOW matricial acelerada por GPU (CuPy / NumPy)
    └── entrenador_cbow.py                  # Bucle de entrenamiento por lotes y resguardos
```

> **Nota**: El archivo del corpus de entrenamiento (`datos/corpus.txt`) se ubica en el directorio `datos/` y está excluido del control de versiones mediante `.gitignore`.

---

## ⚙️ Configuración del Corpus y Parámetros del Proyecto

Todos los parámetros del proyecto (ruta del corpus, tamaño de lote, semilla aleatoria, tamaño de ventana, etc.) están centralizados en `configuracion.yaml`.

### 1. Selección de Fuente del Corpus de Texto
Para entrenar el modelo con tu propio corpus de texto o con el archivo provisto por la cátedra:
1. Coloca tu archivo de texto dentro de la carpeta `datos/` (por ejemplo `datos/corpus.txt`).
2. Modifica el atributo `ruta_corpus` en `configuracion.yaml`:
   ```yaml
   ruta_corpus: datos/corpus.txt
   ```

### 2. Parámetros Principales de Entrenamiento
- **`incluir_puntuacion_y_numeros`**: Incluir signos de puntuación y números en la tokenización (`true` o `false`).
- **`tamanio_ventana`**: Tamaño de la ventana de contexto $C/2$ a izquierda y a derecha (ej. `5` para un contexto total $C = 10$).
- **`dimension_embedding`**: Dimensión de la capa oculta o vector de embedding $N$ (ej. `100`).
- **`tasa_aprendizaje`**: Tasa de aprendizaje $\eta$ para la actualización de gradientes (ej. `0.2`).
- **`cantidad_epocas`**: Número total de épocas de entrenamiento (ej. `10`).

### 3. Modelo CBOW
Para cada palabra central $p_O$ con $C = 2m$ palabras de contexto:
- Capa oculta: $h = \frac{1}{C}(v_{I_1} + \dots + v_{I_C})$, el promedio de las filas de $W$ del contexto (equivale a $W^t x$ con entradas *one-hot*, sin armar esas matrices).
- Excitación de salida: $u = W'^t h$.

**Softmax completo** (`cantidad_negativos: 0`):
- Salida: $y = \text{softmax}(u)$, pérdida $E = -\log y_O$, error $e = y - t$, $EH = W'e$.
- Actualización: $v'_j \leftarrow v'_j - \eta\, e_j h$ para toda columna $j$ de $W'$, y $v_{I_c} \leftarrow v_{I_c} - \eta \frac{1}{C} EH$ solo para las filas del contexto.

**Muestreo negativo** (`cantidad_negativos: K > 0`):
- $P_{sel}$: $K$ palabras por muestra tomadas de $U(p)^{3/4}/Z$, sin incluir a $p_O$.
- Pérdida $E = -\log\sigma(v'_O \cdot h) - \sum_{j \in P_{sel}} \log\sigma(-v'_j \cdot h)$; solo se actualizan las columnas de $W'$ de $\{p_O\} \cup P_{sel}$.

El entrenamiento es por descenso por el gradiente **muestra a muestra** (sin lotes): cada palabra central produce una actualización de $W$ y $W'$, con las dimensiones de las clases ($h \in \mathbb{R}^{N \times 1}$, $u, y, e \in \mathbb{R}^{|V| \times 1}$, $EH \in \mathbb{R}^{N \times 1}$). Las muestras se recorren en orden aleatorio en cada época.

### 4. Métricas: pérdida y accuracy
Por cada época se registra la pérdida promedio y el *accuracy* (fracción de muestras en que la palabra central fue la más probable según la red, medido sobre cada muestra antes de su actualización). Se guardan en el `.npz` (`historial_perdida`, `historial_accuracy`) y en `respaldos/metricas_w{ventana}.csv`, que se actualiza al terminar cada época:
```bash
uv run python graficar_metricas.py respaldos/metricas_w4.csv   # genera respaldos/metricas_w4.png
```

---

## 💻 Flujo de Entrenamiento y Análisis

### 1. Entrenamiento Local por Consola
Para ejecutar el pipeline de entrenamiento desde la terminal:
```bash
# Ejecutar entrenamiento con la configuracion por defecto
uv run python entrenar.py

# O bien especificando parametros de consola
uv run python entrenar.py --epocas 10 --tasa 0.05
```
Al finalizar, se guardará automáticamente la copia de respaldo `.npz` en la carpeta `respaldos/`.

### 2. Análisis y Evaluación en Jupyter Notebook
Abre el notebook `TP1_CBOW_Procesamiento_Lenguaje_Natural.ipynb` para:
1. Cargar modelos resguardados `.npz`.
2. Graficar las curvas de pérdida por época.
3. Buscar palabras más similares mediante producto interno o similaridad de coseno sobre la matriz $W$.
4. Comparar curvas de entrenamiento entre distintos experimentos.

---

## 🔄 Reanudación del Entrenamiento desde un Checkpoint (.npz)

Para continuar entrenando un modelo existente a partir de una copia de seguridad en formato `.npz`:

```bash
uv run python entrenar.py --reanudar --checkpoint respaldos/modelo_cbow_w5_epoca_10.npz --epocas 5
```

Al reanudar, se restaurarán las matrices de pesos $W$ y $W'$, el vocabulario de palabras, la época actual y el historial de pérdida acumulado.
