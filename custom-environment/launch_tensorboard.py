from tensorboard import program
import time

logdir = "logs"  # o la ruta completa si estás en otro directorio
tb = program.TensorBoard()
tb.configure(argv=[None, '--logdir', logdir, '--port', '7007'])  # puerto alternativo
url = tb.launch()
print(f"TensorBoard está corriendo en: {url}")

# Mantener el proceso vivo
try:
    while True:
        time.sleep(60)
except KeyboardInterrupt:
    print("TensorBoard detenido.")
