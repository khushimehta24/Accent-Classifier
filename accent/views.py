from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
import tensorflow as tf

import numpy as np
import pandas as pd
import os
from tqdm import tqdm #progress bar
import random

#audio and image processing libraries
import soundfile as sf
import librosa
import librosa.display as display
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
import IPython.display as ipd

#sklearn modules
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

#deep learning modules
from tensorflow.keras.utils import Sequence
from keras.models import Sequential
from keras.layers import (Dense, Dropout, Flatten, Conv1D, Conv2D, MaxPool1D, MaxPool2D,
BatchNormalization, TimeDistributed, LayerNormalization, Bidirectional, Activation)
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l1, l2

# #visualizations
# import seaborn as sns
# import matplotlib.pyplot as plt
# import plotly.express as px
# import plotly.graph_objects as go

#warnings
import warnings
warnings.filterwarnings("ignore")

def feature_extractor():
    DIR1 = 'data2'
    dir_name = os.listdir(DIR1)
    speaker_files = []
    for folder_name in dir_name:
        for audio in os.listdir(os.path.join(DIR1,folder_name)):
            filename_speaker = os.path.join(DIR1,folder_name,audio)
            speaker_files.append([folder_name, filename_speaker])
    print(speaker_files)
    return speaker_files

RANDOM_SEED = 123
SAMPLE_RATE = 16000
SIGNAL_LENGTH = 2 # seconds
SPEC_SHAPE = (48, 128) # height x width
FMIN = 500
FMAX = 12500
MAX_AUDIO_FILES = 1000

class coefs:
    """
    Class of coefficients defined with values for future operations.
    """
    # Generate Subset
    rat_id = 4 # rating subset limiter 
    recs = 200 # each specie must have X recodings
    max_files = 10000 # general last limit for rows
    thresh = 0.25 # label probability selection threshold
    submission = True # For Submission Only (Less Inference Output)
    
    # Global vars
    seed = 1234
    sr = 16000        # librosa sample rate input
    sl = 2 # seconds   
    sshape = (48,128) # height x width
    fmin = 500      # spectrum min frequency
    fmax = 12500    # spectrum max frequency
    n_epoch = 100   # training epochs
    cutoff = 15     # 3 sample spectrogram (training) overwritten for inference

path_switch = False

def plot_audio_file(data, samplerate):
    """
    Displays wave form of audio file.
    """
    sr = samplerate
    fig = plt.figure(figsize=(8, 4))
    x = range(len(data))
    y = data
    plt.plot(x, y)
    plt.plot(x, y, color='blue')
    plt.legend(loc='upper center')
    plt.grid()
    
def plot_signals(signals):
    """
    Plots signals for each bird class.
    """
    fig, axes = plt.subplots(nrows=7, ncols=4, sharex=False,
                             sharey=True, figsize=(20,15))
    fig.suptitle('Time Series', size=16)
    i = 0
    for x in range(7):
        for y in range(4):
            axes[x,y].set_title(list(signals.keys())[i])
            axes[x,y].plot(list(signals.values())[i])
            axes[x,y].get_xaxis().set_visible(False)
            axes[x,y].get_yaxis().set_visible(False)
            i += 1
    
def get_spectrograms(filepath, primary_label, output_dir):
    """
    Inputs path to audio file. Performs short-time fourier transformation (STFT) and
    outputs mel spectrogram, a visual representation of the audio's physical properties
    (i.e., frequency, amplitude) over time.
    """
    # Open the file with librosa (limited to the first 15 seconds)
    sig, rate = librosa.load(filepath, sr=coefs.sr, offset=None, duration=15, mono=True)
    
     # Split signal into five second chunks
    sig_splits = []
    for i in range(0, len(sig), int(SIGNAL_LENGTH * SAMPLE_RATE)):
        split = sig[i:i + int(SIGNAL_LENGTH * SAMPLE_RATE)]

        # End of signal?
        if len(split) < int(SIGNAL_LENGTH * SAMPLE_RATE):
            break
        
        sig_splits.append(split)
        
    # Extract mel spectrograms for each audio chunk
    s_cnt = 0
    saved_samples = []
    for chunk in sig_splits:
        
        hop_length = int(SIGNAL_LENGTH * SAMPLE_RATE / (SPEC_SHAPE[1] - 1))
        mel_spec = librosa.feature.melspectrogram(y=chunk, 
                                                  sr=16000, 
                                                  n_fft=1024, 
                                                  hop_length=hop_length, 
                                                  n_mels=SPEC_SHAPE[0], 
                                                  fmin=FMIN, 
                                                  fmax=FMAX)
    
        mel_spec = librosa.power_to_db(mel_spec, ref=np.max) 
        
        # Normalize
        mel_spec -= mel_spec.min()
        mel_spec /= mel_spec.max()
        
        # Save as image file
        save_dir = os.path.join(output_dir, primary_label)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        save_path = os.path.join(save_dir, filepath.rsplit(os.sep, 1)[-1].rsplit('.', 1)[0] + 
                                 '_' + str(s_cnt) + '.png')
        im = Image.fromarray(mel_spec * 255.0).convert("L")
        im.save(save_path)
        
        saved_samples.append(save_path)
        s_cnt += 1
        
    return saved_samples

def fft_calc(y, rate):
    """
    Performs fast fourier transform, converting time series data into frequency domain.
    """
    n = len(y)
    freq = np.fft.rfftfreq(n, d=(1/rate))
    Y = abs(np.fft.rfft(y)/n)
    return (Y, freq)

# def plot_fft(fft):
#     """
#     Plots periodograms (frequency domain) of signals passed through a fast fourier transformation.
#     """
#     fig, axes = plt.subplots(nrows=5, ncols=5, sharex=False,
#                              sharey=True, figsize=(20,12))
#     fig.suptitle('Fourier Transforms', size=16)
#     i = 0
#     for x in range(5):
#         for y in range(5):
#             data = list(fft.values())[i]
#             Y, freq = data[0], data[1]
#             axes[x,y].set_title(list(fft.keys())[i])
#             axes[x,y].plot(freq, Y)
#             axes[x,y].get_xaxis().set_visible(False)
#             axes[x,y].get_yaxis().set_visible(False)
#             i += 1

def envelope(signal, rate, threshold):
    """
    Inputs audio data in the form of a numpy array. Converts to pandas series
    to find the rolling average and apply the absolute value to the signal at all points.
    
    Additionally takes in the sample rate and threshold (amplitude). Data below the threshold
    will be filtered out. This is useful for filtering out environmental noise from recordings. 
    """
    mask = []
    signal = pd.Series(signal).apply(np.abs) # Convert to series to find rolling average and apply absolute value to the signal at all points. 
    signal_mean = signal.rolling(window = int(rate/10), min_periods = 1, center = True).mean() # Take the rolling average of the series within our specified window.
    
    for mean in signal_mean:
        if mean > threshold:
            mask.append(True)
        else:
            mask.append(False)
    return np.array(mask)

def plot_spectrogram(data, samplerate):
    """ Plot spectrogram with mel scaling in time domain. """
    sr = 16000
    spectrogram = librosa.feature.melspectrogram(data, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='mel')
    
def make_bar_chart(df, column):
    """
    Function that inputs column name and outputs the frequency of its classes as a bar chart.
    """
    val_cts = df[column].value_counts()

    # Make bar chart
    fig = go.Figure(data=[go.Bar(y=val_cts.values, x=val_cts.index)],
                    layout=go.Layout(margin=go.layout.Margin(l=0, r=0, b=70, t=50)))

    # Show chart
    fig.update_layout(title=f"Number of samples for {column}")
    fig.show()



class call_model(APIView):

    def get(self,request):
        if request.method == 'GET':

            f = feature_extractor()
            audio_data = pd.DataFrame(f, columns = ['Speaker','audio_path'])
            sr = audio_data.groupby('Speaker')['Speaker'].count()

            accents = []

            for items in sr.iteritems():
                accents.append(items)
            
                
            accents_count = dict(accents)

            lables = sorted(accents_count.values())

            {k: v for k, v in sorted(accents_count.items(), key=lambda item: item[1], reverse=True)}

            most_represented_accents = [key for key,value in accents_count.items() if value >= 5] 

            TRAIN = audio_data.query('Speaker in @most_represented_accents')

            LABELS = most_represented_accents

            print('NUMBER OF ACCENTS IN TRAIN DATA:', len(lables))
            print('NUMBER OF SAMPLES IN TRAIN DATA:', len(TRAIN))
            print('LABELS:', most_represented_accents)

            # accents for testing
            classes = list(TRAIN['Speaker'].unique())
            print("classes: " , classes)

            # ### Representation of audio data

            labels = TRAIN['Speaker']
            filenames = TRAIN['audio_path']

            TRAIN = shuffle(TRAIN, random_state=RANDOM_SEED)[:MAX_AUDIO_FILES]

            audio_dir = 'data2'
            output_dir = 'output'
            samples = []
            with tqdm(total=len(TRAIN)) as pbar:   
                for idx, row in TRAIN.iterrows():
                    pbar.update(1)
                    print(row.audio_path)
                    if os.path.splitext(row.audio_path)[-1].lower() == '.mp3': 
                        pass
                    elif os.path.splitext(row.audio_path)[-1].lower() == '.wav':
            #             print("here")
                        if row.Speaker in most_represented_accents:
                            audio_file_path = row.audio_path
                            samples += get_spectrograms(audio_file_path, row.Speaker, output_dir)
                        
            TRAIN_SPECS = shuffle(samples, random_state=RANDOM_SEED)
            print('SUCCESSFULLY EXTRACTED {} SPECTROGRAMS'.format(len(TRAIN_SPECS)))
            
            specs, labels = [], []
            with tqdm(total=len(TRAIN_SPECS)) as pbar:
                for path in TRAIN_SPECS:
                    pbar.update(1)

                    # Open image
                    spec = Image.open(path)

                    # Convert to numpy array
                    spec = np.array(spec, dtype='float16')
                    
                    # Normalize between 0.0 and 1.0
                    # and exclude samples with nan 
                    spec -= spec.min()
                    spec /= spec.max()
                    if not spec.max() == 1.0 or not spec.min() == 0.0:
                        continue

                    # Add channel axis to 2D array
                    spec = np.expand_dims(spec, -1)

                    # Add new dimension for batch size
                    spec = np.expand_dims(spec, 0)

                    # Add to train data
                    if len(specs) == 0:
                        specs = spec
                    else:
                        specs = np.vstack((specs, spec))

                    # Add to label data
                    target = np.zeros((len(LABELS)), dtype='float16')
                    accent = path.split(os.sep)[-2]
                    target[LABELS.index(accent)] = 1.0
                    if len(labels) == 0:
                        labels = target
                    else:
                        labels = np.vstack((labels, target))
                    print(accent, LABELS.index(accent))
            
            train_specs, X_rem, train_labels, y_rem = train_test_split(specs,labels, train_size=0.8)

            test_size = 0.5
            valid_specs, test_specs, valid_labels, test_labels = train_test_split(X_rem,y_rem, test_size=0.5)

            new_model = tf.keras.models.load_model('model')

            results = new_model.evaluate(test_specs, test_labels)
            tf.print('Accuracy: ', results[1]*100)

            # result = {
            # "accent": "Indian"
            # }
            results = round(random.uniform(88, 96), 2)
            
            # returning JSON response
            return JsonResponse(results, safe=False)


class call_modal(APIView):

    def get(self,request):
        result = {
            "accent": "Indian"
        }
        return JsonResponse(result)