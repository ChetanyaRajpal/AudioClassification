# Import required libraries
import os
import librosa
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import StandardScaler
import random
import DataAugmentationFunc
import ImportsandFunctions

# Define random augmentation functions
def random_augmentation(waveform, sr):
    augmentations = [DataAugmentationFunc.change_pitch_speed, DataAugmentationFunc.change_pitch_only, DataAugmentationFunc.change_speed_only,
                    DataAugmentationFunc.value_augmentation] 
                    # DataAugmentationFunc.add_distribution_noise, DataAugmentationFunc.random_shifting, DataAugmentationFunc.shift_silent_right]
                     
    num_augmentations = random.randint(1, len(augmentations))
    selected_augmentations = random.sample(augmentations, num_augmentations)

    y_aug = waveform.copy()
    for aug_func in selected_augmentations:
        y_aug = aug_func(y_aug, sr)
    return y_aug

# Function to extract features
def extract_features(waveform, sr):
    try:
        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=waveform, sr=16000, n_mfcc=13)
        mfccs_mean = np.mean(mfccs.T, axis=0)

        # Extract additional features
        chroma = librosa.feature.chroma_stft(y=waveform, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)

        spectral_contrast = librosa.feature.spectral_contrast(y=waveform, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)

        # Combine features
        features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean))
        return features
    except Exception as e:
        print(f"Error extracting features: {e}")
        return None

# Dataset path
dataset_path = "F:/Code/Python/AudioClassification/AudioClassification/wavfiles"
sample_rate = 16000
data = []

# Iterate through dataset
for category in os.listdir(dataset_path):
    category_path = os.path.join(dataset_path, category)

    if os.path.isdir(category_path):
        print(f"Processing category: {category}")
        for file_name in os.listdir(category_path):
            file_path = os.path.join(category_path, file_name)

            if file_path.endswith(".wav"):
                try:
                    # Load original waveform
                    waveform, sr = librosa.load(file_path, sr=sample_rate)
                    
                    # Extract features from original audio
                    original_features = extract_features(waveform, sr)
                    if original_features is not None:
                        data.append([original_features, category])
                    
                    # Generate augmented waveform
                    augmented_waveform = random_augmentation(waveform, sr)
                    
                    # Extract features from augmented audio
                    augmented_features = extract_features(augmented_waveform, sr)
                    if augmented_features is not None:
                        data.append([augmented_features, category])
                except Exception as e:
                    print(f"Error processing file {file_path}: {e}")

def extract_features2(file_path):
    try:
        # Load the audio file
        y,sr = librosa.load(file_path, sr=16000)
                # Extract features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)  # MFCC features
        mfccs_mean = np.mean(mfccs.T, axis=0)  # Mean of MFCCs across time
        
        # Extract additional features if needed
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        chroma_mean = np.mean(chroma.T, axis=0)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        spectral_contrast_mean = np.mean(spectral_contrast.T, axis=0)
        
        # Combine features
        features = np.concatenate((mfccs_mean, chroma_mean, spectral_contrast_mean))
        
        return features
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
# Convert to DataFrame
columns = [f"feature_{i+1}" for i in range(len(data[0][0]))] + ["label"]
df = pd.DataFrame(data, columns=["features", "label"])

# Save the features and labels
df_expanded = pd.DataFrame(df["features"].tolist(), columns=columns[:-1])
df_expanded["label"] = df["label"]

# Save as CSV
output_csv = "augmented_audio_features_random.csv"
df_expanded.to_csv(output_csv, header=False, index=False)
print(f"Extracted features saved to {output_csv}")

dataset = pd.read_csv('augmented_audio_features_random.csv', header=None)
print(dataset.head())

features = dataset.drop(32, axis = 1)
print(features.head())

#Normalization of features
from sklearn.compose import make_column_transformer, make_column_selector
from sklearn.preprocessing import MinMaxScaler, StandardScaler
ct = make_column_transformer(
    (MinMaxScaler(), make_column_selector(dtype_include='number')) 
)

transformed_data = ct.fit_transform(features)

labels = dataset[32]
print(labels.head())

label_encoder = LabelEncoder()
labels = label_encoder.fit_transform(labels)
labels_one_hot = tf.one_hot(labels, depth = 10)
print(labels_one_hot)

features = np.array(transformed_data)
print(features[0].shape)
labels_one_hot = labels_one_hot.numpy()
X_train, X_test, y_train, y_test = train_test_split(features, labels_one_hot, random_state=42, test_size=0.2, shuffle=True)

print(len(X_train))
print(len(X_test))
print(X_train.shape)
print(y_train.shape)
print(y_test)


model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dense(10, activation = 'softmax')
])

model.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

history = model.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test))

model2 = tf.keras.Sequential([
    tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train.shape[1], 1)),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')
])

model2.compile(loss = tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.Adam(),
              metrics = ['accuracy'])

history2 = model2.fit(X_train, y_train, epochs = 100, validation_data = (X_test, y_test))

audio_path = ('AudioClassification/wavfiles/Clarinet/3c66098d.wav', 'Clarinet')  
test_case = extract_features2(audio_path[0])
print(test_case.shape)

model.evaluate(X_test, y_test)
model2.evaluate(X_test, y_test)

def model_prediction(model, audio_path=ImportsandFunctions.play_random_sound("F:/Code/Python/AudioClassification/AudioClassification/wavfiles")):
    extracted_features = extract_features2(audio_path[0])
    df = pd.DataFrame(extracted_features)
    df_transposed = df.transpose()
    features_normalized = ct.transform(df_transposed)
    features = np.array(features_normalized)
    predictions = model.predict(features)
    predicted_class = label_encoder.inverse_transform([np.argmax(predictions)])
    print(audio_path[1])
    return predicted_class
audio_path = ('C:/Users/HP/Downloads/dizi-flute-02-72563.wav', 'acoustic-guitar')
print(model_prediction(model2, audio_path=audio_path))
print(model_prediction(model, audio_path=audio_path))

