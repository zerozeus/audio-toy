import matplotlib.pyplot as plt
import numpy as np
import librosa.display

class AudioSpecPloter(object):
    
    def plot(self, path, sample_rate, offset, duration):
        y, _ = librosa.load(path = path, sr = sample_rate, offset=offset, duration=duration)
        plt.figure(figsize=(12, 8))
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, y_axis='linear', x_axis='time')
        plt.colorbar(format='%+2.0f dB')
        plt.title('Linear-frequency power spectrogram')
        plt.show()
