#-----------------------------------------------------
# Programa de la fase de preprocesamiento de imágenes
# que convierte imágenes RGB en escala de grises
#-----------------------------------------------------

import os
import cv2

# Ruta de la carpeta de entrada que contiene las imágenes en color
input_folder = 'Database_CNNEF_3000_CV/test/surprise'

# Ruta de la carpeta de salida donde se guardarán las imágenes en blanco y negro
output_folder = 'Database_CNNEF_3000_CV_gray/test/surprise'

# Crear la carpeta de salida si no existe
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Obtener la lista de archivos en la carpeta de entrada
files = os.listdir(input_folder)

# Iterar sobre los archivos en la carpeta de entrada
for file_name in files:
    # Construir la ruta completa de entrada
    input_path = os.path.join(input_folder, file_name)
    input_path= input_path.replace("\\", "/")
    
    # Leer la imagen en color
    image = cv2.imread(input_path)
    
    # Convertir la imagen a blanco y negro
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Construir la ruta completa de salida
    output_path = os.path.join(output_folder, file_name)
    
    # Guardar la imagen en blanco y negro en la carpeta de salida
    cv2.imwrite(output_path, grayscale_image)

print("Imágenes convertidas con éxito a blanco y negro.")
