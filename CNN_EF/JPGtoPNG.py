#-----------------------------------------------------
# Programa de la fase de preprocesamiento de imágenes
# que convierte las extensiones jpg a png
#------------------------------------------------
import os
from PIL import Image

# Ruta de la carpeta que contiene las imágenes en formato JPG
input_folder = 'Database_CNNEF_3000_CV/test/surprise'

# Ruta de la carpeta de salida donde se guardarán las imágenes en formato PNG
output_folder = 'Database_CNNEF_3000_CV_png/test/surprise'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Obtener la lista de archivos en la carpeta de entrada
files = os.listdir(input_folder)

# Iterar sobre los archivos en la carpeta de entrada
for file_name in files:
    # Construir la ruta completa de entrada
    input_path = os.path.join(input_folder, file_name)
    
    # Abrir la imagen en formato JPG
    image = Image.open(input_path)
    
    # Construir la ruta completa de salida
    output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + '.png')
    
    # Guardar la imagen en formato PNG
    image.save(output_path, "PNG")

print("Las imágenes se han convertido correctamente de JPG a PNG.")
