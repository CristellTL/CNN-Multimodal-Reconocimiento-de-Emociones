#----------------------------------------
#      PREPROCESAMIENTO DE IMÁGENES
#  Recorte de espectrogramas para filtrar 
#     únicamente información relevante
#----------------------------------------
 
import cv2
import os
import shutil

# Directorio destino
to_dir = "NAO-REspectrograms/train"

# Dirección de la carpeta principal
db_image_directory = "Databases/NAO-Espectrograms/train"
img_files = os.listdir(db_image_directory)

indices = 0
emotions = ['01-Enojo', '02-Aversion', '03-Miedo', '04-Alegria', '05-Tristeza','06-Sorpresa']

# Para cada carpeta de emociones en la carpeta de DB
for dir in img_files:
  
  img_files_path = os.path.join(db_image_directory, dir)
  img_files_path= img_files_path.replace("\\", "/")
  img_files_2 = os.listdir(img_files_path)
  i = 0

  # Para cada imagen de cada carpeta de emociones
  for file in img_files_2: 
     img_files_path2_ = os.path.join(img_files_path, file)
     img_files_path2= img_files_path2_.replace("\\", "/")
     
     # Cargar la imagen
     image = cv2.imread(img_files_path2)

      # Recortar la región de interés (cara)
     roi = image[58:428, 79:479]
     roi = cv2.resize(roi, (192,192))
       
     # Guardar la imagen recortada en un archivo
     file_name = str(i) +".jpg"
     cv2.imwrite(file_name, roi)

     path1 = file_name                         
     path2 = to_dir + '/' + emotions[indices]                        
     path3 = to_dir + '/' + emotions[indices] + '/' + file_name   
        
        # Checa si la ruta de la carpeta/directorio existen antes de moverlo
        # Si no crea una nueva carpeta/directorio y mueve la imagen
     if os.path.exists(path2):
        print("Movviendo " + file_name + ".....")
        # Mueve de path1 ---> path3
        shutil.move(path1, path3)
     else:
        os.makedirs(path2)
        print("Moviendo " + file_name + ".....")
        shutil.move(path1, path3)
     i = i +1
  indices = indices +1
