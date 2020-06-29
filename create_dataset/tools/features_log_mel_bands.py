#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from librosa.feature import melspectrogram
import librosa

__author__ = 'Konstantinos Drossos -- Tampere University'
__docformat__ = 'reStructuredText'
__all__ = ['feature_extraction']


def feature_extraction(audio_data: np.ndarray,
                       sr: int,
                       nb_fft: int,
                       hop_size: int,
                       nb_mels: int,
                       f_min: float,
                       f_max: float,
                       htk: bool,
                       power: float,
                       norm: bool,
                       window_function: str,
                       center: bool)\
        -> np.ndarray:
    """Feature extraction function.

    :param audio_data: Audio signal.
    :type audio_data: numpy.ndarray
    :param sr: Sampling frequency.
    :type sr: int
    :param nb_fft: Amount of FFT points.
    :type nb_fft: int
    :param hop_size: Hop size in samples.
    :type hop_size: int
    :param nb_mels: Amount of MEL bands.
    :type nb_mels: int
    :param f_min: Minimum frequency in Hertz for MEL band calculation.
    :type f_min: float
    :param f_max: Maximum frequency in Hertz for MEL band calculation.
    :type f_max: float|None
    :param htk: Use the HTK Toolbox formula instead of Auditory toolkit.
    :type htk: bool
    :param power: Power of the magnitude.
    :type power: float
    :param norm: Area normalization of MEL filters.
    :type norm: bool
    :param window_function: Window function.
    :type window_function: str
    :param center: Center the frame for FFT.
    :type center: bool
    :return: Log mel-bands energies of shape=(t, nb_mels)
    :rtype: numpy.ndarray
    """
    y = audio_data
    mel_bands = melspectrogram(
        y=y, sr=sr, n_fft=nb_fft, hop_length=hop_size, win_length=nb_fft,
        window=window_function, center=center, power=power, n_mels=nb_mels,
        fmin=f_min, fmax=f_max, htk=htk, norm=norm).T
    logmel_spectrogram = librosa.core.power_to_db(
            mel_bands, ref=1.0, amin=1e-10, 
            top_db=None)
    logmel_spectrogram = logmel_spectrogram.astype(np.float32)        
    return logmel_spectrogram


# EOF
