import os
import re
from glob import glob

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import signal
from scipy.io import wavfile
from scipy.fftpack import fft
from sklearn.decomposition import PCA

def log_spectrogram(audio_input, sample_rate, window_size = 20,
                    step_size = 10, eps = 1e-10):
    
    samples_per_segment = int(round(window_size*sample_rate / 1e3))
    n_overlap = int(round(step_size*sample_rate / 1e3))
    frequencies, segment_times, spectrogram = \
        signal.spectrogram(audio_input,
                           fs = sample_rate,
                           window = 'hann',
                           nperseg = samples_per_segment,
                           noverlap = n_overlap,
                           detrend = False)

    return frequencies, segment_times, np.log(spectrogram.T.astype(np.float32) + eps)

def plot_spectrogram(input_name, sample_rate, sample_times, frequencies, samples, spectrogram):
    fig = plt.figure(figsize = (14, 8))
    ax1 = fig.add_subplot(211)
    ax1.set_title('Raw wave of file ' + input_name)
    ax1.set_ylabel('Amplitude')
    ax1.plot(np.linspace(0, sample_rate/len(samples), sample_rate), samples)

    ax2 = fig.add_subplot(212)
    ax2.imshow(spectrogram.T, aspect = 'auto', origin = 'lower',
               extent = [sample_times.min(), sample_times.max(), frequencies.min(), frequencies.max()])
    ax2.set_yticks(frequencies[::16])
    ax2.set_xticks(sample_times[::16])
    ax2.set_title('Spectrogram of file ' + input_name)
    ax2.set_ylabel('Frequency [Hz]')
    ax2.set_xlabel('Time [s]')

    plt.show()

def load_audio_data(data_dir):
    POSSIBLE_LABELS = 'yes no up down left right on off stop go silence unknown'.split()
    id2name = {i: name for i, name in enumerate(POSSIBLE_LABELS)}
    name2id = {name: i for i, name in id2name.items()}

    pattern = re.compile("(.+\/)?(\w+)\/([^_]+)_.+wav")
    all_files = glob(os.path.join(data_dir, 'audio/*/*wav'))

    with open(os.path.join(data_dir, 'validation_list.txt'), 'r') as f_val:
        validation_files = f_val.readlines()
    validation_set = set()
    for f in validation_files:
        r = re.match(pattern, f)
        if r:
            validation_set.add(r.group(3))
    
    possible_labels = set(POSSIBLE_LABELS)
    train, val = [], []

    for f in all_files:
        r = re.match(pattern, f)
        if r:
            label, uid = r.group(2), r.group(3)
            if label ==  '_background_noise_':
                label = 'silence'
            if label not in possible_labels:
                label = 'unknown'
            label_id = name2id[label]

            sample = (label_id, uid, f)
            if uid in validation_set:
                val.append(sample)
            else: train.append(sample)
    print('Data split into {} training and {} validation samples'.format(len(train), len(val)))
    return train, val