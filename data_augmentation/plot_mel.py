import librosa
import librosa.display
from tools.features_log_mel_bands import feature_extraction
import matplotlib.pyplot as plt

if __name__ == '__main__':
    wav_file =  r'./07 ambient bell.wav'
    y = librosa.load(path=wav_file, sr=44100, mono=True)[0]

    mel = feature_extraction(y, sr=44100, nb_fft=1024, hop_size=512, nb_mels=64, window_function='hann', center=True,
                             f_min=.0, f_max=None, htk=False, power=1.0, norm=1)

    fig = plt.figure(figsize=(10, 4))
    librosa.display.specshow(mel.T, x_axis='time',
                             y_axis='mel', sr=44100,
                             fmax=None)
    plt.tight_layout()
    plt.show()
    fig.savefig(r'mel.png', dpi=400)