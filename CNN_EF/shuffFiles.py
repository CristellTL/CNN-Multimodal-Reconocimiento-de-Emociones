#-------------------------------------------------------
# Programa de la fase de distribución del dataset
# que divide una carpeta en dos directorios de acuerdo
# a la proporción establecida
#-------------------------------------------------------

import os
import random
import shutil

def split_folder(source_folder, destination_folder_1, destination_folder_2, split_ratio=0.8):
    # Crear las carpetas de destino si no existen
    if not os.path.exists(destination_folder_1):
        os.makedirs(destination_folder_1)
    if not os.path.exists(destination_folder_2):
        os.makedirs(destination_folder_2)

    # Obtener la lista de archivos en la carpeta de origen
    file_list = os.listdir(source_folder)
    # Mezclar la lista de archivos de forma aleatoria
    random.shuffle(file_list)

    # Calcular el número de archivos para cada subcarpeta
    num_files_1 = int(len(file_list) * split_ratio)
    num_files_2 = len(file_list) - num_files_1

    # Copiar los archivos a las subcarpetas
    for i, file_name in enumerate(file_list):
        source_path = os.path.join(source_folder, file_name)
        source_path= source_path.replace("\\", "/")
        if i < num_files_1:
            destination_path = os.path.join(destination_folder_1, file_name)
            destination_path= destination_path.replace("\\", "/")
        else:
            destination_path = os.path.join(destination_folder_2, file_name)
            destination_path= destination_path.replace("\\", "/")
        shutil.copyfile(source_path, destination_path)

# Carpeta de origen
source_folder = "EMO-MX-NAO-EF-CNN/val_/05-Tristeza"

# Carpetas de destino
destination_folder_1 = "EMO-MX-NAO-EF-CNN/test/05-Tristeza"
destination_folder_2 = "EMO-MX-NAO-EF-CNN/val/05-Tristeza"

# Proporción para la primera carpeta (70%)
split_ratio = 0.7

# Llamar a la función para dividir la carpeta
split_folder(source_folder, destination_folder_1, destination_folder_2, split_ratio)

print("Carpeta dividida exitosamente.")
