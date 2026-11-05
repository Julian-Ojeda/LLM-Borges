# Paquete de modulos Python en espanol para el TP1 CBOW
#
#   configuracion.py          lee los YAML de config/ y completa valores por defecto
#   tokenizador.py            texto -> lista de tokens
#   generador_vocabulario.py  tokens -> vocabulario (indice de cada palabra)
#   modelo_cbow_cupy.py       pesos W y W', pasos de gradiente del CBOW, guardar/cargar .npz
#   entrenador_cbow.py        bucle de entrenamiento del CBOW (épocas, metricas, respaldos)
#   modelo_atencion_torch.py  CBOW con una capa de auto-atencion en PyTorch (autograd)
#   modelo_atencion_cupy.py   el mismo modelo en CuPy con el gradiente escrito a mano
