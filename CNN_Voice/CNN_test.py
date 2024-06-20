#------------------------------------------------
# Programa que mide el accuracy de la CNNVoice
#------------------------------------------------

import os
import numpy as np
from matplotlib import pyplot
from matplotlib.image import imread
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

#--------------------
# Cargar testImages
#--------------------
testing_image_directory = "NAO-REspectrograms/test"
img_files = os.listdir(testing_image_directory)

#-------------------------
# Cargar el modelo de CNN
#-------------------------
model = tf.keras.models.load_model('CNNNAO_16_200_7CC.h5')

#------------------------------------------------------------
# Cuantificar la exactitud de la CNN en una testImage
#------------------------------------------------------------
"""""
#Ruta completa de testImage
img_files_path= "Database_CKModified/test/aversion/19.png"

# Cargar imagen 
img_1 = load_img(img_files_path,target_size=(48,48))

# Convertir la imagen en matriz y aumentar la dimensión
img_2 = img_to_array(img_1)
img_3 = np.expand_dims(img_2, axis=0)
  
# Predecir la clase de la imagen que no se ha visto
prediction = model.predict(img_3)
predict_class = np.argmax(prediction, axis=1)
print(predict_class)

#Mostrar la imagen en pantalla
pyplot.imshow(img_1)
pyplot.title(predict_class[0])
pyplot.axis('off')
pyplot.show()
"""
#-----------------------------------------------------------------
# Graficar la exactitud de la CNN con 15 testImages de cada grupo
#-----------------------------------------------------------------
""""
indices = 0
emotions = ["anger", "aversion", "fear", "happyness", "sadness", "surprise"]
test_accuracy = [0, 0, 0, 0, 0, 0]
for dir in img_files:
  
  img_files_path = os.path.join(testing_image_directory, dir)
  img_files_path= img_files_path.replace("\\", "/")
  img_files_2 = os.listdir(img_files_path)
  i= 0
  tc = 0

  for file in img_files_2[15:29]: 

    # Ruta completa de testImage
    img_files_path2 = os.path.join(img_files_path, file)
    img_files_path2= img_files_path2.replace("\\", "/")
    print("Nombre del archivo" + img_files_path2)
    
    # Cargar imagen 
    img_1 = load_img(img_files_path2,target_size=(48,48))

    # Convertir la imagen en matriz
    img_2 = img_to_array(img_1)
    img_2 /= 255.0
    # Aumentar la dimensión
    img_3 = np.expand_dims(img_2, axis=0)
    
    # Predecir la clase de la imagen que no se ha visto
    prediction = model.predict(img_3)
    # print(prediction)

    predict_class = np.argmax(prediction, axis=1)
    # print(predict_class)

    # Graficar la imagen usando subplot
    pyplot.subplot(3, 5, i+1)
    pyplot.imshow(img_2.astype('uint8'))
    
    # Agregar título a la imagen como el valor de la clase predecida
    pyplot.title(predict_class[0])

    # No mostrar los ejes x, y en la imagen
    pyplot.axis('off')

    i=i+1

    if predict_class==indices: 
      tc = tc + 1

  pyplot.savefig(dir+ '.png')
  pyplot.show()
  test_accuracy[indices] = tc/i
  indices = indices + 1

print (emotions)
print (test_accuracy)

"""
#-------------------------------------------------------------------
# Graficar la exactitud de la CNN con todas las imágenes de testing
#-------------------------------------------------------------------

indices = 0
emotions = ["anger", "aversion", "fear", "happyness", "sadness", "surprise"]
test_accuracy = []
x_predict = []
x_true = []

for dir in img_files:
  
  img_files_path = os.path.join(testing_image_directory, dir)
  img_files_path= img_files_path.replace("\\", "/")
  img_files_2 = os.listdir(img_files_path)

  for file in img_files_2: 

    # Ruta completa de testImage
    img_files_path2 = os.path.join(img_files_path, file)
    img_files_path2= img_files_path2.replace("\\", "/")
    
    # Cargar imagen 
    img_1 = load_img(img_files_path2,target_size=(192,192))

    # Convertir la imagen en matriz
    img_2 = img_to_array(img_1)
    img_2 /= 255.0
    # Aumentar la dimensión
    img_3 = np.expand_dims(img_2, axis=0)
    
    # Predecir la clase de la imagen que no se ha visto
    prediction = model.predict(img_3)
    # print(prediction)

    predict_class = np.argmax(prediction, axis=1)
    # print(predict_class)

    x_predict.append(predict_class)
    x_true.append(indices)

  indices = indices +1


# Calcular la matriz de confusión
cm = confusion_matrix(x_true, x_predict)  

# Configurar el gráfico
pyplot.figure(figsize=(12, 10))
    
# Crear el mapa de calor
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=emotions, yticklabels= emotions)
    
# Añadir etiquetas y título
pyplot.xlabel('Predicted Label')
pyplot.ylabel('True Label')
pyplot.title('Confusion Matrix')
pyplot.savefig('mc.png')
# Mostrar el gráfico
pyplot.show()


