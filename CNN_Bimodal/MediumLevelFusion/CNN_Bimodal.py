#-----------------------------------------------------------------
# CNN para la clasificación de emociones con fusión en nivel medio
#-----------------------------------------------------------------

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, concatenate, Dropout
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Aumento de datos para el conjunto de entrenamiento
train_datagen = ImageDataGenerator(rescale=1./255)

# Solo reescalar para el conjunto de validación y prueba
val_datagen = ImageDataGenerator(rescale=1./255)

# Directorios de los datos
trainEF_dir = 'EMO-MX-NAO-EF-CNN/train'
valEF_dir = 'EMO-MX-NAO-EF-CNN/val'
testEF_dir = 'EMO-MX-NAO-EF-CNN/test'

trainVoice_dir = 'NAO-REspectrograms/train'
valVoice_dir = 'NAO-REspectrograms/val'
testVoice_dir = 'NAO-REspectrograms/test'

# Generadores de datos para el primer tipo de imagen
train_generatorEF = train_datagen.flow_from_directory(
    trainEF_dir,
    target_size=(192,192)
)

val_generatorEF = val_datagen.flow_from_directory(
    valEF_dir,
    target_size=(192,192)
)

test_generatorEF = val_datagen.flow_from_directory(
    testEF_dir,
    target_size=(192,192)
)

# Generadores de datos para el segundo tipo de imagen
train_generatorVoice = train_datagen.flow_from_directory(
    trainVoice_dir,
    target_size=(192,192)
)

val_generatorVoice = val_datagen.flow_from_directory(
    valVoice_dir,
    target_size=(192,192)
)

test_generatorVoice = val_datagen.flow_from_directory(
    testVoice_dir,
    target_size=(192,192)
)

def combined_generator(gen1, gen2):
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        yield [X1i[0], X2i[0]], X1i[1]

train_generator = combined_generator(train_generatorEF, train_generatorVoice)
val_generator = combined_generator(val_generatorEF, val_generatorVoice)
test_generator = combined_generator(test_generatorEF, test_generatorVoice)


#----------------------------------------------------------------------
#   Creación de sub-redes para expresiones faciales y emociones en voz
#_---------------------------------------------------------------------

def create_subnetworkEF(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(64, (3, 3),padding ="same", activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding ="same",activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3),padding ="same", activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return input_layer, x

def create_subnetworkVoice(input_shape):
    input_layer = Input(shape=input_shape)
    x = Conv2D(32, (3, 3),padding ="same", activation='relu')(input_layer)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(64, (3, 3), padding ="same",activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Conv2D(128, (3, 3), padding ="same",activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    return input_layer, x

input_shape1 = (192,192,3)  
input_shape2 = (192,192,3)  

input_layer1, output1 = create_subnetworkEF(input_shape1)
input_layer2, output2 = create_subnetworkVoice(input_shape2)


merged = concatenate([output1, output2])
x = Dense(256, activation='relu')(merged)
x = Dropout(0.5)(x)
output_layer = Dense(6, activation='softmax')(x)  


model = Model(inputs=[input_layer1, input_layer2], outputs=output_layer)

# Compilar el modelo
model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])

model.summary()


early_stop = EarlyStopping(monitor='val_loss', patience=14, verbose=True, min_delta=1e-4)  
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=7, verbose=True, min_delta=1e-4)  
callbacks_list = [early_stop, reduce_lr] 

history = model.fit(
    train_generator,
    steps_per_epoch=30,
    batch_size=16,
    epochs=100,
    validation_data=val_generator,
    validation_steps=20,
    verbose=True ,
    callbacks= callbacks_list
)

# Guardar el modelo CNNEFP_BatchSize_Epoch
model.save("CNNEFP4_16_100.h5")

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
