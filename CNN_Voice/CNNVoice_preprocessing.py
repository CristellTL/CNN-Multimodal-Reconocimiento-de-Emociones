#-----------------------------------------------
# Preprocesamiento de los archivos de audio
#-----------------------------------------------

import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import time
import numpy as np

# Directorio de base de datos
audio_directory = "Databases/EMO-MX-NAO-SP-CNN/train"
audio_files = os.listdir(audio_directory)


for dir in audio_files:
  
  audio_files_path = os.path.join(audio_directory, dir)
  audio_files_path= audio_files_path.replace("\\", "/")
  audio_files_2 = os.listdir(audio_files_path)
  index = 0
  print(audio_files_2)
  
  for file in audio_files_2: 

    # Ruta completa de testImage
    audio_files_path2 = os.path.join(audio_files_path, file)
    audio_files_path2= audio_files_path2.replace("\\", "/")
    
    # Cargar el archivo de audio
    print("Cargando audio")
    y, sr = librosa.load(audio_files_path2)

    # Calcular el espectrograma
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)

    # Mostrar el espectrograma
    librosa.display.specshow(librosa.power_to_db(spectrogram, ref=np.max), sr=sr, x_axis='time', y_axis='mel')
    
    plt.colorbar(format='%+2.0f dB')
    plt.title('Espectrograma')
    plt.savefig(audio_files_path2+ str(index) + '.png')
    plt.show()
    
    index = index +1
  
  
