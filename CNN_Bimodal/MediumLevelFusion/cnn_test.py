#-------------------------------------------------------------
# Programa que mide el accuracy de la CNN_MediumLevelFusion
#-------------------------------------------------------------

import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Cargar el modelo entrenado
model = load_model('CNNEFP4_16_100.h5')

# Crear instancias de ImageDataGenerator
test_datagen = ImageDataGenerator(rescale=1./255)

# Directorio de datos de prueba
test_dir1 = 'EMO-MX-NAO-EF-CNN/test'
test_dir2 = 'NAO-REspectrograms/test'

# Crear generadores de datos de prueba
test_generator1 = test_datagen.flow_from_directory(
    test_dir1,
    target_size=(192,192),
    batch_size=11,
    class_mode='sparse',
    shuffle=False
)

test_generator2 = test_datagen.flow_from_directory(
    test_dir2,
    target_size=(192,192),
    batch_size=11,
    class_mode='sparse',
    shuffle=False
)
# Generador combinado
def combined_generator(gen1, gen2):
    while True:
        X1i = gen1.next()
        X2i = gen2.next()
        yield [X1i[0], X2i[0]], X1i[1]

test_generator = combined_generator(test_generator1, test_generator2)

# Obtener etiquetas reales y predicciones
y_true = []
y_pred = []

steps = test_generator1.samples // test_generator1.batch_size
print(steps)
for _ in range(steps):
    # Obtener lote de datos
    [X1_batch, X2_batch], labels = next(test_generator)
    
    # Predecir con el modelo
    preds = model.predict([X1_batch, X2_batch])
    
    # Obtener la clase predicha (índice del valor máximo en las predicciones)
    predicted_classes = np.argmax(preds, axis=1)
    
    # Añadir etiquetas y predicciones a las listas
    y_true.extend(labels)
    y_pred.extend(predicted_classes)

# Convertir a arrays numpy
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Crear la matriz de confusión
cm = confusion_matrix(y_true, y_pred)

# Etiquetas de las clases
labels = ['angry', 'aversion', 'fear', 'happy','sad','surprise']  

# Visualizar la matriz de confusión
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap=plt.cm.Blues)

plt.title('Matriz de Confusión')
plt.show()