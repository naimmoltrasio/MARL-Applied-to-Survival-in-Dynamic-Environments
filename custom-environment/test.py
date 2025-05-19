import torch
print("Versión de torch:", torch.__version__)
print("Versión de CUDA:", torch.version.cuda)
print("¿CUDA disponible?:", torch.cuda.is_available())
print("Dispositivo por defecto:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
print("Nombre GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No detectada")

import sys
print("Ruta de Python en uso:", sys.executable)
