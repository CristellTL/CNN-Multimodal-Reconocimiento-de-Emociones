#-----------------------------------------------
#          PREPROCESAMIENTO DE IMÁGENES
#   Concatenación de características antes 
#       de ingresar a la red neuronal
#-----------------------------------------------

import os
from PIL import Image

def unir_imagenes(imagen1_path, imagen2_path, output_path):
    # Abre las imágenes
    imagen1 = Image.open(imagen1_path)
    imagen2 = Image.open(imagen2_path)

    nueva_ancho = imagen1.width + imagen2.width
    nueva_alto = max(imagen1.height, imagen2.height)
    nueva_imagen = Image.new('RGB', (nueva_ancho, nueva_alto))
        
    # Pega las imágenes una al lado de la otra
    nueva_imagen.paste(imagen1, (0, 0))
    nueva_imagen.paste(imagen2, (imagen1.width, 0))

    # Guarda la imagen resultante
    nueva_imagen.save(output_path)

def unir_carpetas(carpeta1, carpeta2, carpeta_salida, direccion='horizontal'):
    # Obtener listas de archivos de imagen en ambas carpetas
    imagenes1 = sorted([os.path.join(carpeta1, f) for f in os.listdir(carpeta1) if f.endswith(('jpg', 'jpeg', 'png'))])
    imagenes2 = sorted([os.path.join(carpeta2, f) for f in os.listdir(carpeta2) if f.endswith(('jpg', 'jpeg', 'png'))])

    if len(imagenes1) != len(imagenes2):
        raise ValueError("Las carpetas deben contener el mismo número de imágenes")

    # Crear carpeta de salida si no existe
    os.makedirs(carpeta_salida, exist_ok=True)

    # Unir cada par de imágenes
    for img1, img2 in zip(imagenes1, imagenes2):
        nombre_salida = os.path.join(carpeta_salida, os.path.basename(img1))
        unir_imagenes(img1, img2, nombre_salida, direccion)
        print(f"Imagen guardada en {nombre_salida}")

path_1 = "EMO-MX-NAO-EF-CNN/val/06-sorpresa"
path_2 = "NAO-REspectrograms/val/06-sorpresa"
path_to = "DatabaseNAO/val/06-sorpresa"
unir_carpetas(path_1, path_2, path_to)
