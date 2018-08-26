import librosa, os
import numpy as np
from tensorflow.python.platform import gfile

class AudioAugmentation:
    def read_audio_file(self, file_path):
        input_length = 16000
        data = librosa.core.load(file_path)[0]
        if len(data) > input_length:
            first = len(data) / 2 - input_length / 2
            end = len(data) / 2 + input_length / 2
            data = data[first:end]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

    def write_audio_file(self, file, data, sample_rate=16000):
        librosa.output.write_wav(file, data, sample_rate)

    def add_noise(self, data):
        noise = np.random.randn(len(data))
        data_noise = data + 0.005 * noise
        return data_noise

    def shift(self, data):
        return np.roll(data, 1600)

    def stretch(self, data, rate=1):
        input_length = 16000
        data = librosa.effects.time_stretch(data, rate)
        if len(data) > input_length:
            data = data[:input_length]
        else:
            data = np.pad(data, (0, max(0, input_length - len(data))), "constant")
        return data

# Set the path for the target class
paths = os.path.join('./2', '*.wav')
waves = gfile.Glob(paths)

for path in waves:
    # Create a new instance from AudioAugmentation class
    aa = AudioAugmentation()

    # Read and show cat sound
    data = aa.read_audio_file(path)
    aa.write_audio_file(path, data)

    # Adding noise to sound
    data_noise = aa.add_noise(data)

    # Shifting the sound
    data_roll = aa.shift(data)

    # Stretching the sound
    data_stretch = aa.stretch(data, 0.8)

    # Write generated cat sounds
    aa.write_audio_file(path[:-4] + '_1.wav', data_noise)
    aa.write_audio_file(path[:-4] + '_2.wav', data_roll)
    aa.write_audio_file(path[:-4] + '_3.wav', data_stretch)
