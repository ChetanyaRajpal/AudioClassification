import numpy as np
import librosa
import soundfile as sf
import sounddevice as sd

def change_pitch_speed(y, sr=16000):
    # Create a copy of the input signal
    y_pitch_speed = y.copy()
    
    # Randomly vary speed and pitch
    length_change = np.random.uniform(low=0.8, high=1.2)
    speed_fac = 1.0 / length_change
    
    # Interpolate to adjust pitch and speed
    tmp = np.interp(np.arange(0, len(y_pitch_speed), speed_fac),
                    np.arange(0, len(y_pitch_speed)),
                    y_pitch_speed)
    
    # Truncate or pad to match the original length
    minlen = min(len(y_pitch_speed), len(tmp))
    y_pitch_speed[:minlen] = tmp[:minlen]
    y = y_pitch_speed
    
    # Return the modified signal
    return y

def change_pitch_only(y, sr=16000):
    y_pitch = y.copy()
    bins_per_octave = 12
    pitch_pm = 2
    pitch_change =  pitch_pm * 2*(np.random.uniform())   
    y_pitch = librosa.effects.pitch_shift(y_pitch, n_steps=pitch_change, 
                                        bins_per_octave=bins_per_octave, sr=16000)
    return y_pitch

def change_speed_only(y,sr=16000):
    y_speed = y.copy()
    speed_change = np.random.uniform(low=0.9,high=1.1)
    print("speed_change = ",speed_change)
    tmp = librosa.effects.time_stretch(y_speed, rate=speed_change)
    minlen = min(y_speed.shape[0], tmp.shape[0])
    y_speed *= 0 
    y_speed[0:minlen] = tmp[0:minlen]
    return y_speed

def value_augmentation(y,sr=16000):
    y_aug = y.copy()
    dyn_change = np.random.uniform(low=1.5,high=3)
    y_aug = y_aug * dyn_change
    return y_aug

def add_distribution_noise(y,sr=16000):
    y_noise = y.copy()
# you can take any distribution from https://docs.scipy.org/doc/numpy-1.13.0/reference/routines.random.html
    noise_amp = 0.005*np.random.uniform()*np.amax(y_noise)
    y_noise = y_noise + noise_amp * np.random.normal(size=y_noise.shape[0])
    return y_noise

def random_shifting(y,sr=16000):
    y_shift = y.copy()
    timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length

    start = int(y_shift.shape[0] * timeshift_fac)
    
    if (start > 0):
        y_shift = np.pad(y_shift,(start,0),mode='constant')[0:y_shift.shape[0]]
    else:
        y_shift = np.pad(y_shift,(0,-start),mode='constant')[0:y_shift.shape[0]]
    return y_shift

def shift_silent_right(y,sr=16000):
    sampling=y[(y > 200) | (y < -200)]
    shifted_silent =sampling.tolist()+np.zeros((y.shape[0]-sampling.shape[0])).tolist()
    return shifted_silent

def stretching(y,sr=16000):
    input_length = len(y)
    streching = y.copy()
    streching = librosa.effects.time_stretch(streching, rate=1.1)
    if len(streching) > input_length:
        streching = streching[:input_length]
    else:
        streching = np.pad(streching, (0, max(0, input_length - len(streching))), "constant")
    return streching


# Load audio file
# y, sr = librosa.load('C:/Users/HP/Downloads/dizi-flute-02-72563.wav', sr=16000)

# # Modify pitch and speed
# y_modified = change_pitch_speed(y, sr)

# # Save the modified audio to a file
# output_path = 'C:/Users/HP/Downloads/modified_audio.wav'
# sf.write(output_path, y_modified, sr)

# # Play the modified audio
# print(f"Playing modified audio: {output_path}")
# print(y[0])
# print(change_pitch_only(y,sr))
# # sd.play(y_modified, sr)
# sd.wait()  
# sd.play(y,sr)
# sd.wait()
