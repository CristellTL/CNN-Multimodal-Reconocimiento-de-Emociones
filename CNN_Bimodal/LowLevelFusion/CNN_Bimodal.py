#-----------------------------------------------------------------
# CNN para la clasificación de emociones con fusión en bajo nivel
#-----------------------------------------------------------------

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, Callback 
from keras.losses import sparse_categorical_crossentropy
import PIL
from PIL import Image

#--------------------------------------------------------------
# Aumento de datos aleatorio 
#(Cambio de tamaño, rotación, giros, zoom, transformaciones) 
#usando ImageDataGenerator 
#--------------------------------------------------------------

#----------------#
# TRAINING DATA  #
#----------------#

training_data_generator = ImageDataGenerator(
    rescale = 1.0/255)

# Directorio de imágenes de entrenamiento
#training_image_directory = "DataBases/Database_CKModified/train"
training_image_directory = "DatabaseNAO/train"

# Generación de aumento de datos procesados
training_augmented_images = training_data_generator.flow_from_directory(
    training_image_directory,
    target_size=(384,192))


#------------------#
# VALIDATION DATA  #
#------------------#

validation_data_generator = ImageDataGenerator(
    rescale = 1.0/255)

# Directorio de imágenes de validación
validation_image_directory = "DatabaseNAO/val"

# Generación de aumento de datos procesados
validation_augmented_images = validation_data_generator.flow_from_directory(
    validation_image_directory,
    target_size=(384,192))


print(training_augmented_images.class_indices)
print(validation_augmented_images.class_indices)

#--------------------------------#
# Definiendo el modelo de CNN    # 
# CNN de 3 capas convolucionales #
#--------------------------------#

model = tf.keras.models.Sequential([
    
    tf.keras.layers.Conv2D(32, (3,3), padding = 'same', activation='relu', input_shape=(384, 192,3)),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3,3), padding = 'same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(64, (3,3), padding = 'same',activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5),

    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])

# Resumen del modelo
model.summary()

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=14, verbose=True, min_delta=1e-4)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=True, min_delta=1e-4)  
callbacks_list = [early_stop, reduce_lr] 

# Ajustar el modelo
history = model.fit(
    training_augmented_images, 
    batch_size=64, 
    epochs=200, 
    validation_data = validation_augmented_images, 
    verbose=True, 
    callbacks= callbacks_list)


# Guardar el modelo CNNEFP_BatchSize_Epoch
model.save("CNNBimNAO_3CC_64_200.h5")

plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.savefig('training_accuracy.png')
plt.show()

plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Loss')
plt.legend()
plt.savefig('training_loss.png')
plt.show()

