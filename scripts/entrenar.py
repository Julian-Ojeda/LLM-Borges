#!/usr/bin/env python3
"""
Script de ejecucion autonoma para el entrenamiento de la red neuronal CBOW.
Permite iniciar un entrenamiento desde cero o reanudar un checkpoint (.npz) desde la terminal.

Es la version "de referencia": cada paso usa operaciones de CuPy/NumPy (codigo/modelo_cbow_cupy.py).
Soporta softmax completo y muestreo negativo. Para softmax completo mas rapido, ver entrenar_cuda.py.

Uso:
    python scripts/entrenar.py                                   # todo segun config/configuracion.yaml
    python scripts/entrenar.py --epocas 5 --tasa 0.025           # pisa esos dos valores del YAML
    python scripts/entrenar.py --negativos 10                    # muestreo negativo con 10 palabras
    python scripts/entrenar.py --reanudar --checkpoint respaldos/modelo_cbow_w4_epoca_254.npz

OJO al reanudar: los respaldos nuevos se numeran desde la época del checkpoint en adelante y se guardan
en directorio_respaldos. Si ya existen respaldos de épocas posteriores, se sobrescriben.
"""

import argparse
import os
import sys
from pathlib import Path

# Asegurar acceso a los modulos del directorio codigo/ y trabajar desde la raiz del proyecto,
# para que las rutas relativas (config/, datos/, respaldos/) funcionen desde cualquier carpeta
ruta_raiz = Path(__file__).resolve().parent.parent
if str(ruta_raiz) not in sys.path:
    sys.path.append(str(ruta_raiz))
os.chdir(ruta_raiz)

from codigo.configuracion import cargar_configuracion
from codigo.entrenador_cbow import entrenar


def parsear_argumentos() -> argparse.Namespace:
    """
    Parsea los argumentos de linea de comandos para el entrenamiento por consola.

    :return: Objeto Namespace con los argumentos parseados.
    """
    parser = argparse.ArgumentParser(
        description="Entrenamiento independiente por consola de redes neuronales CBOW (CuPy / NumPy)."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/configuracion.yaml",
        help="Ruta al archivo YAML de configuracion (por defecto: config/configuracion.yaml).",
    )
    parser.add_argument(
        "--epocas",
        type=int,
        default=None,
        help="Sobrescribe la cantidad de epocas a entrenar.",
    )
    parser.add_argument(
        "--tasa",
        type=float,
        default=None,
        help="Sobrescribe la tasa de aprendizaje eta (ejemplo: 0.2).",
    )
    parser.add_argument(
        "--negativos",
        type=int,
        default=None,
        help="Sobrescribe la cantidad de ejemplos negativos (0 = softmax completo).",
    )
    parser.add_argument(
        "--reanudar",
        action="store_true",
        help="Activa el modo de reanudacion desde un checkpoint .npz.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Ruta al archivo .npz desde el cual reanudar (requiere --reanudar).",
    )
    return parser.parse_args()


def main() -> None:
    """
    Funcion principal de punto de entrada para ejecutar el pipeline de entrenamiento CBOW.

    :return: None.
    """
    args = parsear_argumentos()

    print("==========================================================================")
    print("      ENTRENAMIENTO CBOW LOCAL - LINEA DE COMANDOS (TERMINAL IDE)        ")
    print("==========================================================================")

    # 1. Carga de configuracion desde YAML
    configuracion = cargar_configuracion(args.config)

    # Sobrescrituras opcionales desde CLI (None = no se paso ese argumento, queda el valor del YAML)
    if args.epocas is not None:
        configuracion["cantidad_epocas"] = args.epocas
    if args.tasa is not None:
        configuracion["tasa_aprendizaje"] = args.tasa
    if args.negativos is not None:
        configuracion["cantidad_negativos"] = args.negativos
    if args.reanudar:
        configuracion["reanudar_entrenamiento"] = True
    if args.checkpoint is not None:
        configuracion["ruta_checkpoint"] = args.checkpoint

    # 2. Ejecucion del entrenamiento matricial
    modelo = entrenar(configuracion)

    # 3. Resumen final del entrenamiento
    print("\n==========================================================================")
    print("                     ENTRENAMIENTO FINALIZADO CON ÉXITO                   ")
    print("==========================================================================")
    print(f"Pérdida final alcanzada: {modelo['historial_perdida'][-1]:.4f}")
    print(f"Accuracy final alcanzado: {modelo['historial_accuracy'][-1]:.4f}")
    if "duracion_total_segundos" in modelo:
        duracion = modelo["duracion_total_segundos"]
        minutos = int(duracion // 60)
        segundos_resto = duracion % 60
        if minutos > 0:
            print(f"Tiempo total transcurrido: {duracion:.2f} s ({minutos} m {segundos_resto:.2f} s)")
        else:
            print(f"Tiempo total transcurrido: {duracion:.2f} s")

    hacer_respaldo = configuracion.get("hacer_respaldo", True)
    frecuencia_respaldo = configuracion.get("frecuencia_respaldo", 2)
    directorio_respaldos = configuracion.get("directorio_respaldos", "respaldos")

    if hacer_respaldo:
        print(f"Copias de respaldo: Activadas (frecuencia: cada {frecuencia_respaldo} épocas en '{directorio_respaldos}')")
    else:
        print("Copias de respaldo: Desactivadas")
    print("==========================================================================")


if __name__ == "__main__":
    main()
