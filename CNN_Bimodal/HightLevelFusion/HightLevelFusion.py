#--------------------------------------------------------
# Programa que mide el accuracy de la CNN_HightLevelFusion
#--------------------------------------------------------

import os
import numpy as np
from matplotlib import pyplot
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.metrics import accuracy_score, confusion_matrix
import seaborn as sns

#--------------------
# Cargar testImages
#--------------------
testing_imageVoice_directory = "NAO-REspectrograms/test"
imgEF_files = os.listdir(testing_imageVoice_directory)

testing_imageEF_directory = "EMO-MX-NAO-EF-CNN/test"
imgVoice_files = os.listdir(testing_imageVoice_directory)

#-------------------------
# Cargar los modelos de CNN
#-------------------------
modelVoice = tf.keras.models.load_model('Voice.h5')
modelEF = tf.keras.models.load_model('FacialExpresion.h5')

#----------------------------------------
# Predicción de emociones en voz
#----------------------------------------

indices = 0
emotions = ["anger", "aversion", "fear", "happyness", "sadness", "surprise"]
test_accuracy = []
x_predict = []
prediction_voice = []
x_true = []

for dir in imgVoice_files:
  
  img_files_path = os.path.join(testing_imageVoice_directory, dir)
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
    prediction = modelVoice.predict(img_3)
    prediction_voice.append(prediction)
    # print(prediction)

    predict_class = np.argmax(prediction, axis=1)
    # print(predict_class)

    x_predict.append(predict_class)
    x_true.append(indices)

  indices = indices +1

#---------------------------------------------------
# Predicción de emociones en expresiones faciales
#---------------------------------------------------

x_predict = []
prediction_ef = []

for dir in imgEF_files:
  
  img_files_path = os.path.join(testing_imageEF_directory, dir)
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
    prediction = modelEF.predict(img_3)
    prediction_ef.append(prediction)
    # print(prediction)

    predict_class = np.argmax(prediction, axis=1)
    # print(predict_class)

    x_predict.append(predict_class)

#-----------------------
# Predicción ponderada
#-----------------------

# Sumar ponderaciones de las predicciones

weight_EF = 0.95
weight_Voice = 0.05
x_predict_aux = []
x_predict = []

for i in range(0,66):
  x_predict = []
  for j in range (0,6):
    x = prediction_ef[i][0][j]* weight_EF + prediction_voice[i][0][j] * weight_Voice
    x_predict.append(x)
  x_predict_aux.append(x_predict)

x_predict = np.argmax(x_predict_aux, axis=1)

# Calcular precisión
accuracy = accuracy_score(x_true, x_predict)
print(f'Precisión del modelo combinado: {accuracy * 100:.2f}%')

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