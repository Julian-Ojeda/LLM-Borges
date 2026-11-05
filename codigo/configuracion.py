"""
Modulo de configuracion del modelo CBOW.
Permite cargar y guardar hiperparametros desde y hacia archivos YAML.

Rol en el proyecto: todos los scripts (entrenar.py, entrenar_cuda.py, ajustar_puntuacion_cuda.py,
entrenar_atencion_torch.py) arrancan llamando a cargar_configuracion(...) con su YAML de config/.
El resultado es un diccionario comun {nombre_parametro: valor} que despues se pasa al entrenamiento.
Los argumentos de consola (--epocas, --tasa, ...) pisan valores de ese diccionario.
"""

from pathlib import Path
import yaml


def cargar_configuracion(ruta_yaml: str | Path = "configuracion.yaml") -> dict:
    """
    Carga los parametros de configuracion desde un archivo YAML y devuelve un diccionario.
    Establece valores por defecto en español para cualquier parametro omitido.

    :param ruta_yaml: Ruta al archivo YAML de configuracion (por defecto 'configuracion.yaml').
    :return: Diccionario con la totalidad de los hiperparametros de entrenamiento y modelo.
    """
    # Valores por defecto: se usan solo si el YAML no trae ese parametro.
    # (Los YAML de config/ traen todos, asi que en la practica mandan los del YAML.)
    configuracion_por_defecto = {
        "ruta_corpus": "datos/corpus_prueba.txt",          # archivo de texto plano con el corpus
        "directorio_respaldos": "respaldos",               # carpeta de los .npz y del CSV de metricas
        "incluir_puntuacion_y_numeros": True,              # True: '.', ',' y numeros son tokens; False: solo palabras
        "frecuencia_minima": 1,                            # apariciones minimas para entrar al vocabulario (si no, <UNK>)
        "tamanio_ventana": 5,                              # m: palabras a cada lado de la central (C = 2m de contexto)
        "dimension_embedding": 100,                        # N: tamaño de la capa oculta = largo de cada embedding
        "tasa_aprendizaje": 0.2,                           # eta: cuanto se mueven los pesos en cada paso
        "cantidad_negativos": 0,                           # 0 = softmax completo; K > 0 = muestreo negativo con K palabras
        "cantidad_epocas": 10,                             # pasadas completas por el corpus en esta corrida
        "hacer_respaldo": True,                            # guardar .npz periodicamente
        "frecuencia_respaldo": 2,                          # cada cuantas épocas se guarda un .npz
        "semilla_aleatoria": 26,                           # fija los numeros aleatorios (reproducibilidad)
        "reanudar_entrenamiento": False,                   # True: parte de ruta_checkpoint en lugar de pesos al azar
        "ruta_checkpoint": "respaldos/modelo_cbow_w5_epoca_10.npz",   # .npz desde el que se reanuda
    }

    path_yaml = Path(ruta_yaml)
    if path_yaml.exists():
        with open(path_yaml, "r", encoding="utf-8") as archivo:
            datos_cargados = yaml.safe_load(archivo)          # YAML -> diccionario de Python
            if isinstance(datos_cargados, dict):
                configuracion_por_defecto.update(datos_cargados)   # lo del YAML pisa a los valores por defecto

    return configuracion_por_defecto


def guardar_configuracion(
    configuracion_dict: dict, ruta_yaml: str | Path = "configuracion.yaml"
) -> None:
    """
    Guarda un diccionario de configuracion en un archivo YAML.

    :param configuracion_dict: Diccionario que contiene los parametros a exportar.
    :param ruta_yaml: Ruta del archivo YAML de destino (por defecto 'configuracion.yaml').
    :return: None.
    """
    destino = Path(ruta_yaml)
    destino.parent.mkdir(parents=True, exist_ok=True)       # crea la carpeta si no existe
    with open(destino, "w", encoding="utf-8") as archivo:
        yaml.safe_dump(
            configuracion_dict,
            archivo,
            default_flow_style=False,   # un parametro por linea (no todo en una linea entre llaves)
            sort_keys=False,            # respeta el orden del diccionario
            allow_unicode=True,         # escribe tildes y ñ tal cual
        )
