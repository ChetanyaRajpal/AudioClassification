import os 
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from python_speech_features import mfcc, logfbank
import librosa

#Getting the information about the directories such as :
#dirpath - the path to the directory
#dirnames - the names of the directory
#filenames - the names of the file in it
for dirpath, dirnames, filenames in os.walk('F:/Code/Python/AudioClassification/AudioClassification/wavfiles'):
    print(f"there are {len(dirnames)} directories and {len(filenames)} .wav files in '{dirpath}'.")

#Getting the class names
dir_names = []

for dirpath, dirnames, filenames in os.walk('F:\Code\Python\AudioClassification\AudioClassification\wavfiles'):
    dir_names.append(dirnames)
dir_names = dir_names[:1]
class_names = [element for innerList in dir_names for element in innerList]
print(class_names)

from playsound import playsound

# playsound(r'F:/Code/Python/AudioClassification/AudioClassification/wavfiles/Acoustic_guitar/0eeaebcb.wav')
import random
def play_random_sound(target_dir):
    random_sound = random.sample(os.listdir(target_dir),1)
    target_folder = target_dir + "/" + random_sound[0]
    
    random_file = random.sample(os.listdir(target_folder),1)
    
    target_file = target_folder + "/" + random_file[0]
    return target_file, random_sound

# play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')

#Plotting .wav files
import os 
import matplotlib.pyplot as plt
from scipy.io import wavfile
import kapre
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Permute
import librosa
import librosa.display
import matplotlib.pyplot as plt
from kapre import composed

plt.style.use('ggplot')


def show_spectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y,sr = librosa.load(audio_path[0], sr=16000) # Loads the audio file. The sampling rate is set to 16000.
    print(y)
    print(sr)

    D = librosa.stft(y) # Computes the Short Time Fourier Transform
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis='time', y_axis='log') # librosa.amplitude_to_db() converts the magnitude of the spectrogram into decibels (dB) for visualization purposes.
    #librosa.display.specshow() visualizes the spectrogram with amplitude in dB.
    plt.colorbar(format='%+2.0f dB')
    plt.title(audio_path[1])
    plt.tight_layout()
    plt.show()

def return_spectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y,sr = librosa.load(audio_path[0], sr = 16000)
    spec = librosa.stft(y)
    spec_db = librosa.amplitude_to_db(np.abs(spec), ref = np.max)
    return spec_db

def show_melspectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y,sr = librosa.load(audio_path[0], sr = 16000)
    print(y)
    print(sr)
    
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    melspec_db = librosa.amplitude_to_db(melspec, ref=np.max)

    plt.figure(figsize=(10, 6))
    librosa.display.specshow(melspec_db, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(audio_path[1])
    plt.tight_layout()
    plt.show()

def return_melspectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y, sr = librosa.load(audio_path[0], sr = 16000)
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length = 512, n_mels = 128)
    melspec_db = librosa.amplitude_to_db(melspec, ref=np.max)
    return melspec_db

#SpecAugment is a simple data augmentation method for speech recognition.
import tensorflow as tf
from tensorflow.keras.models import Sequential
import librosa
from kapre.augmentation import SpecAugment
from kapre.composed import get_melspectrogram_layer
from kapre.time_frequency import Magnitude, MagnitudeToDecibel
import matplotlib.pyplot as plt

def show_kapre_melspectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y, sr = librosa.load(audio_path[0], sr = 16000)
    audio_tensor = tf.reshape(tf.cast(y, tf.float32), (1, -1, 1))
    input_shape = y.reshape(-1, 1).shape
    
    melgram = get_melspectrogram_layer(input_shape = input_shape, 
                                       n_fft = 2048,
                                       return_decibel = True, 
                                       n_mels = 128,
                                       input_data_format = 'channels_last',
                                       output_data_format = 'channels_last')
    model = Sequential()
    model.add(melgram)
    original_spectrogram = model(audio_tensor)
    plt.figure(figsize=(5,5))
    plt.title(audio_path[1])
    plt.imshow(original_spectrogram[0, :, :, :])
    plt.show()

def return_kapre_melspectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y, sr = librosa.load(audio_path[0], sr = 16000)
    audio_tensor = tf.reshape(tf.cast(y, tf.float32), (1, -1, 1))
    input_shape = y.reshape(-1, 1).shape
    
    melgram = get_melspectrogram_layer(input_shape = input_shape, 
                                       n_fft = 2048,
                                       return_decibel = True, 
                                       n_mels = 128,
                                       input_data_format = 'channels_last',
                                       output_data_format = 'channels_last')
    model = Sequential()
    model.add(melgram)
    original_spectrogram = model(audio_tensor)
    return original_spectrogram
    

audio_path = ('AudioClassification/wavfiles/Clarinet/3c66098d.wav', 'Clarinet')  
# kapre_melspectrogram(audio_path=audio_path)
# show_melspectrogram(audio_path=audio_path)
def show_kapre_augmented_melspectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y, sr = librosa.load(audio_path[0], sr = 16000)
    audio_tensor = tf.reshape(tf.cast(y, tf.float32), (1, -1, 1))
    input_shape = y.reshape(-1, 1).shape
    
    melgram = get_melspectrogram_layer(input_shape = input_shape, 
                                       n_fft = 2048,
                                       return_decibel = True, 
                                       n_mels = 128,
                                       input_data_format = 'channels_last',
                                       output_data_format = 'channels_last')
    
    spec_augment = SpecAugment(freq_mask_param=5,
                           time_mask_param=10,
                           n_freq_masks=2,
                           n_time_masks=3,
                           mask_value=-100) 
    
    model = Sequential()
    model.add(melgram)
    model.add(spec_augment)
    original_spectrogram = model(audio_tensor, training=True)
    plt.figure(figsize=(5,5))
    plt.title(audio_path[1])
    plt.imshow(original_spectrogram[0, :, :, :])
    plt.show()

def return_kapre_augmented_melspectrogram(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y, sr = librosa.load(audio_path[0], sr = 16000)
    audio_tensor = tf.reshape(tf.cast(y, tf.float32), (1, -1, 1))
    input_shape = y.reshape(-1, 1).shape
    
    melgram = get_melspectrogram_layer(input_shape = input_shape, 
                                       n_fft = 2048,
                                       return_decibel = True, 
                                       n_mels = 128,
                                       input_data_format = 'channels_last',
                                       output_data_format = 'channels_last')
    
    spec_augment = SpecAugment(freq_mask_param=5,
                           time_mask_param=10,
                           n_freq_masks=2,
                           n_time_masks=3,
                           mask_value=-100) 
    
    model = Sequential()
    model.add(melgram)
    model.add(spec_augment)
    original_spectrogram = model(audio_tensor, training=True)
    return original_spectrogram

def show_kapre_augmented_melspectrogram_DB(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y, sr = librosa.load(audio_path[0], sr = 16000)
    audio_tensor = tf.reshape(tf.cast(y, tf.float32), (1, -1, 1))
    input_shape = y.reshape(-1, 1).shape
    
    melgram = get_melspectrogram_layer(input_shape = input_shape, 
                                       n_fft = 2048,
                                       return_decibel = True, 
                                       n_mels = 128,
                                       input_data_format = 'channels_last',
                                       output_data_format = 'channels_last')
    
    spec_augment = SpecAugment(freq_mask_param=5,
                           time_mask_param=10,
                           n_freq_masks=2,
                           n_time_masks=3,
                           mask_value=-100) 
    
    model = Sequential()
    model.add(melgram)
    model.add(spec_augment)
    model.add(Magnitude())
    model.add(MagnitudeToDecibel())
    original_spectrogram = model(audio_tensor, training=True)
    plt.figure(figsize=(5,5))
    plt.title(audio_path[1])
    plt.imshow(original_spectrogram[0, :, :, :])
    plt.show()

def return_kapre_augmented_melspectrogram_DB(audio_path = play_random_sound('F:/Code/Python/AudioClassification/AudioClassification/wavfiles')):
    y, sr = librosa.load(audio_path[0], sr = 16000)
    audio_tensor = tf.reshape(tf.cast(y, tf.float32), (1, -1, 1))
    input_shape = y.reshape(-1, 1).shape
    
    melgram = get_melspectrogram_layer(input_shape = input_shape, 
                                       n_fft = 2048,
                                       return_decibel = True, 
                                       n_mels = 128,
                                       input_data_format = 'channels_last',
                                       output_data_format = 'channels_last')
    
    spec_augment = SpecAugment(freq_mask_param=5,
                           time_mask_param=10,
                           n_freq_masks=2,
                           n_time_masks=3,
                           mask_value=-100) 
    
    model = Sequential()
    model.add(melgram)
    model.add(spec_augment)
    model.add(Magnitude())
    model.add(MagnitudeToDecibel())
    original_spectrogram = model(audio_tensor, training=True)
    return original_spectrogram


# show_kapre_augmented_melspectrogram(audio_path=audio_path)
# show_kapre_augmented_melspectrogram_DB(audio_path=audio_path)
# show_kapre_melspectrogram(audio_path=audio_path)
# show_melspectrogram(audio_path=audio_path)
    







