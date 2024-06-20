#--------------------------------------
# Programa que muestra la arquitectura
# del modelo de CNN cargado
#--------------------------------------

import os
import tensorflow as tf

#-------------------------
# Cargar el modelo de CNN
#-------------------------
model = tf.keras.models.load_model('Voice.h5')

model.summary()