#----------------------------------------
#      PREPROCESAMIENTO DE IMÁGENES
# Incluye recorte de rostro y conversión
#      de RGB a escala de grises
#----------------------------------------
 
import cv2
import os
import shutil

# Directorio destino
to_dir = "EMO-MX-NAO-EF"

# Dirección de la carpeta principal
db_image_directory = "PreprocessingImage"
img_files = os.listdir(db_image_directory)

# Cargar el clasificador Haar para detectar caras
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

indices = 0
emotions = ['01-Enojo', '02-Aversion', '03-Miedo', '04-Alegria', '05-Tristeza', '06-Sorpresa']

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

     # Convertir la imagen a escala de grises
     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

     # Detectar caras en la imagen
     faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
     
     j=0
     # Iterar sobre las caras detectadas y recortarlas
     for (x, y, w, h) in faces:
  
        # Recortar la región de interés (cara)
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (192,192))
        # Guardar la cara recortada en un archivo
        file_name = str(i) +'_'+ str(j)+".jpg"
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
        j = j+1
     i = i +1
  indices = indices +1