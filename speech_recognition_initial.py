import os
import time

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
from sklearn.decomposition import PCA

from utils import log_spectrogram, plot_spectrogram, load_audio_data

DATA_DIR = './data/train/'
file_name = '/audio/yes/0a7c2a8d_nohash_0.wav'
sample_rate, samples = wavfile.read(str(DATA_DIR) + file_name)
assert sample_rate == 16000, 'Unexpected sample rate, should be 16 kHz.'

freqs, times, spectrogram = log_spectrogram(samples, sample_rate)
plot_spectrogram(file_name, sample_rate, times, freqs, samples, spectrogram)

train, validation = load_audio_data(DATA_DIR)
assert train is not [], 'Error: data was not read correctly'