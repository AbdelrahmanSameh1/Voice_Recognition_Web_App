import librosa
from librosa import power_to_db, util
import librosa.display
import IPython.display as ipd
# import csv
from sklearn import svm
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import scipy
import os
import io
import base64
import sounddevice as sd
from scipy.io.wavfile import write
import wavio as wv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from scipy.signal import get_window
from sklearn import preprocessing
from scipy.io.wavfile import read
import python_speech_features as mfcc
from sklearn.mixture import GaussianMixture
import glob
import pickle
import joblib
from sklearn.metrics import f1_score
from sklearn import preprocessing
import python_speech_features as mfcc
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import norm
import scipy.stats as stats
import math
from scipy.stats import norm
import statistics
import itertools


def calculate_delta(array):
    rows, cols = array.shape
    deltas = np.zeros((rows, 20))
    N = 2
    for i in range(rows):
        index = []
        j = 1
        while j <= N:
            if i-j < 0:
                first = 0
            else:
                first = i-j
            if i+j > rows-1:
                second = rows-1
            else:
                second = i+j
            index.append((second, first))
            j += 1
        deltas[i] = (array[index[0][0]]-array[index[0][1]] +
                     (2 * (array[index[1][0]]-array[index[1][1]]))) / 10
    return deltas


def extract_features(file_path):
    # audio , sample_rate = librosa.load(file_path, mono=True, duration=2)
    # print(np.shape(audio))
    # y,index=librosa.effects.trim(audio,top_db=55)
    # audio = y[index[0]:index[1]]
    sr, audio = read(file_path)
    mfcc_feature = mfcc.mfcc(audio, sr, 0.025, 0.01,
                             20, nfft=1200, appendEnergy=True)
    mfcc_feature = preprocessing.scale(mfcc_feature)
    delta = calculate_delta(mfcc_feature)
    combined = np.hstack((mfcc_feature, delta))
    return combined


def extractFromFile(directory):
    extractedFeatures = np.asarray(())
    for audio in os.listdir(directory):
        audio_path = directory + audio
        # print(audio_path)
        features = extract_features(audio_path)
        if extractedFeatures.size == 0:
            extractedFeatures = features
        else:
            extractedFeatures = np.vstack((extractedFeatures, features))
    return extractedFeatures


def generateModel(modelName, features, pickleName):
    modelName = GaussianMixture(
        n_components=6, max_iter=200, covariance_type='spherical', n_init=3)
    modelName.fit(features)
    gmm = '.gmm'
    name = pickleName + gmm
    pickle.dump(modelName, open(name, 'wb'))
    return modelName


def plot_barChart(scores, speakerFlag, names, img):
    fig = plt.figure(figsize=(25, 10))
    if speakerFlag:
        left = [1, 2, 3]
        height = np.add(scores, [100, 100, 100])
        plt.xlabel('Team members', fontsize=20)
        plt.title('Speaker Recognition', fontsize=20)
    else:
        left = [1, 2, 3, 4]
        height = np.add(scores, [100, 100, 100, 100])
        plt.xlabel('Words', fontsize=20)
        plt.title('Word Recognition', fontsize=20)
    # tick_label = ['Sara', 'Rawan', 'Mohamed']
    plt.bar(left, height, tick_label=names, width=0.8,
            color=['red', 'yellow', 'cyan'])
    plt.ylabel('Scores', fontsize=20)
    imagename = './static/' + img + '.png'
    plt.savefig(imagename)


def plot_melspectrogram(file_name):
    audio, sfreq = librosa.load(file_name)
    # fig=plt.figure(figsize=(25,10))
    fig = plt.figure()
    melspectrogram = librosa.feature.melspectrogram(y=audio, sr=sfreq)
    img = librosa.display.specshow(
        librosa.power_to_db(melspectrogram, ref=np.max))

    return img, fig


def spectral_Rolloff(file_name, img, speaker):
    y, sr = librosa.load(file_name)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.99)
    rolloff_min = librosa.feature.spectral_rolloff(
        y=y, sr=sr, roll_percent=0.01)
    S, _ = librosa.magphase(librosa.stft(y=y))
    _, ax = plt.subplots()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax)

    # rolloffMaxSameh = 4700
    # rolloffMaxAmr = 3850

    rolloffMin = np.min(rolloff)
    rolloffMax = np.max(rolloff_min)
    print(rolloffMin)
    print(rolloffMax)

    # rollArrAmr = []
    # for i in range(len(librosa.times_like(rolloff))):
    #     rollArrAmr.append(rolloffMaxAmr)
    # rollArrSameh = []
    # for i in range(len(librosa.times_like(rolloff_min))):
    #     rollArrSameh.append(rolloffMaxSameh)

    firstAmr = []
    for i in range(len(librosa.times_like(rolloff))):
        firstAmr.append(3150)
    secondAmr = []
    for i in range(len(librosa.times_like(rolloff_min))):
        secondAmr.append(3550)

    firstBeshara = []
    for i in range(len(librosa.times_like(rolloff))):
        firstBeshara.append(3650)
    secondBeshara = []
    for i in range(len(librosa.times_like(rolloff_min))):
        secondBeshara.append(4300)

    firstSameh = []
    for i in range(len(librosa.times_like(rolloff))):
        firstSameh.append(4500)
    secondSameh = []
    for i in range(len(librosa.times_like(rolloff_min))):
        secondSameh.append(5500)

    ax.plot(librosa.times_like(rolloff),
            rolloff[0], label='Roll-off frequency (0.99)')
    ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
            label='Roll-off frequency (0.01)')
    # ax.plot(librosa.times_like(rolloff), rollArrSameh, color='r')
    # ax.plot(librosa.times_like(rolloff), rollArrAmr, color='w')
    ax.plot(librosa.times_like(rolloff), firstAmr, color='y', label='Amr')
    ax.plot(librosa.times_like(rolloff), secondAmr, color='y')
    ax.plot(librosa.times_like(rolloff),
            firstBeshara, color='b', label='Beshara')
    ax.plot(librosa.times_like(rolloff), secondBeshara, color='b')
    ax.plot(librosa.times_like(rolloff), firstSameh, color='r', label="Sameh")
    ax.plot(librosa.times_like(rolloff), secondSameh, color='r')

    ax.legend(loc='lower right')
    ax.set(title=speaker)
    imagename = './static/' + img + '.png'
    plt.savefig(imagename)
    return rolloff


def spectral_features(audio, img):
    signal, sample_rate = librosa.load(audio)
    spec_bw = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate)
    S, phase = librosa.magphase(librosa.stft(y=signal))
    centroid = librosa.feature.spectral_centroid(S=S)
    rolloff = librosa.feature.spectral_rolloff(
        y=signal, sr=sample_rate, roll_percent=0.99)
    rolloff_min = librosa.feature.spectral_rolloff(
        y=signal, sr=sample_rate, roll_percent=0.01)
    fig, ax = plt.subplots()
    times = librosa.times_like(spec_bw)
    ax.semilogy(times, spec_bw[0], label='Spectral bandwidth')
    ax.legend()
    ax.label_outer()
    fig.patch.set_facecolor('#e4e8e8')
    img = librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                                   y_axis='log', x_axis='time', ax=ax)
    ax.set(title='The Spectral Features ')
    ax.fill_between(times, np.maximum(0, centroid[0] - spec_bw[0]),
                    np.minimum(centroid[0] + spec_bw[0], sample_rate/2),
                    alpha=0.5, label='Centroid +- bandwidth')
    ax.plot(times, centroid[0], label='Spectral centroid', color='w')
    ax.plot(librosa.times_like(rolloff),
            rolloff[0], label='Roll-off frequency (0.99)')
    ax.plot(librosa.times_like(rolloff), rolloff_min[0], color='w',
            label='Roll-off frequency (0.01)')
    ax.legend(loc='lower right')
    fig.colorbar(img, ax=ax)
    # img = image(fig, "spec")
    imagename = './static/spec_features.png'
    plt.savefig(imagename)
    # return img


def zeros(audio):
    signal, sample_rate = librosa.load(audio)
    # Zooming in
    n0 = 9000
    n1 = 9100
    fig = plt.figure(figsize=(6, 6))
    plt.plot(signal[n0:n1])
    plt.grid()
    fig.patch.set_facecolor('#e4e8e8')
    zero_crossings = librosa.zero_crossings(signal[n0:n1], pad=False)
    zero = (sum(zero_crossings))
    imagename = './static/zeros.png'
    plt.savefig(imagename)
#    img=image(fig,"zeros")
#    return img,zero


def chroma(file_name):
    y, sr = librosa.load(file_name)
    C = librosa.feature.chroma_stft(y=y, sr=sr)
    plt.subplot(1, 1, 1)
    librosa.display.specshow(C, y_axis='chroma')
    plt.colorbar()
    plt.title('Chromagram')
    imagename = './static/chroma.png'
    plt.savefig(imagename)


def pie(scores_1, scores_2, scores_3, flag):
    weights = [abs(scores_1 + 100), abs(scores_2 + 100), abs(scores_3 + 100)]
    labels = ['Amr', 'Beshara', 'Sameh']

    if scores_1 > scores_2 and scores_1 > scores_3:
        explode = (0.1, 0, 0)
    elif scores_2 > scores_1 and scores_2 > scores_3:
        explode = (0, 0.1, 0)
    elif scores_3 > scores_2 and scores_3 > scores_1:
        explode = (0, 0, 0.1)

    if not flag:
        explode = (0, 0, 0)

    fig1, ax1 = plt.subplots()
    ax1.pie(weights, labels=labels, explode=explode, autopct='%1.1f%%',
            shadow=True, startangle=90)
    # plt.show
    # Equal aspect ratio ensures that pie is drawn as a circle.
    ax1.axis('equal')
    imagename = './static/pie.png'
    plt.savefig(imagename)
    print(scores_1)
    print(scores_2)
    print(scores_3)


def bars(scores_1, scores_2, scores_3):
    labels = ['Amr', 'Beshara', 'Sameh']
    xpoosition = np.arange(len(labels))
    group = [abs(scores_1), abs(scores_2), abs(scores_3)]
    fig1, ax1 = plt.subplots()
    ax1.barh(xpoosition, group, height=0.4, label='Scores')
    plt.yticks(xpoosition, labels)
    # plt.legend()
    # plt.show()
    imagename = './static/bars.png'
    plt.savefig(imagename)


def rms(file_name, amrRec, besharaRec, samehRec):
    # y, sr = librosa.load(librosa.ex('trumpet'))
    y, sr = librosa.load(file_name)
    librosa.feature.rms(y=y)
    S, phase = librosa.magphase(librosa.stft(y))
    rms = librosa.feature.rms(S=S)
    times = librosa.times_like(rms)

    y2, sr2 = librosa.load(amrRec)
    librosa.feature.rms(y=y2)
    S2, phase2 = librosa.magphase(librosa.stft(y2))
    rms2 = librosa.feature.rms(S=S2)
    times2 = librosa.times_like(rms2)

    y3, sr3 = librosa.load(besharaRec)
    librosa.feature.rms(y=y3)
    S3, phase3 = librosa.magphase(librosa.stft(y3))
    rms3 = librosa.feature.rms(S=S3)
    times3 = librosa.times_like(rms3)

    y4, sr4 = librosa.load(samehRec)
    librosa.feature.rms(y=y4)
    S4, phase4 = librosa.magphase(librosa.stft(y4))
    rms4 = librosa.feature.rms(S=S4)
    times4 = librosa.times_like(rms4)

    fig, ax = plt.subplots(nrows=2, sharex=True)

    ax[0].semilogy(times, rms[0], label='Recorded', color='r')
    ax[0].semilogy(times2, rms2[0], label='Amr', color='g')
    ax[0].semilogy(times3, rms3[0], label='Beshara', color='b')
    ax[0].semilogy(times4, rms4[0], label='Sameh', color='y')
    ax[0].set(xticks=[])
    ax[0].legend()
    ax[0].label_outer()
    librosa.display.specshow(librosa.amplitude_to_db(S, ref=np.max),
                             y_axis='log', x_axis='time', ax=ax[1])
    ax[1].set(title='log Power spectrogram')
    imagename = './static/rms.png'
    plt.savefig(imagename)


def guass(rolloff):
    # Plot between -10 and 10 with .001 steps.
    # x_axis = np.arange(-20, 20, 0.01)
    x_axis = rolloff
    new_x_axis = list(itertools.chain.from_iterable(x_axis))
    print("rolloff = ", x_axis)
    # print("test = ", test)

    # Calculating mean and standard deviation
    mean = statistics.mean(new_x_axis)
    sd = statistics.stdev(new_x_axis)
    fig2, ax2 = plt.subplots()

    ax2.plot(new_x_axis, norm.pdf(new_x_axis, mean, sd))
    imagename = './static/rolloff.png'
    plt.savefig(imagename)
    # plt.show()


# def image(fig, name):
#     canvas = FigureCanvas(fig)
#     img = io.BytesIO()
#     fig.savefig(img, format='png')
#     img.seek(0)
#     data = base64.64encode(img.getbuffer()).decode("ascii")
#     image_file_name = 'static/assets/images/' + \
#         str(name)+str(variables.counter)+'jpg'
#     plt.savefig(image_file_name)
#     return f"<img src = 'data:image/png;base64, {data}'/>"


# def extract_features_for_data(audio,sr):
#     # audio , sample_rate = librosa.load(file_path, mono=True ,duration=2 )
#     # print(np.shape(audio))
#     # sr,audio = read(file_path)
#     mfcc_feature = mfcc.mfcc(audio,sr, 0.025, 0.01,20, nfft = 1200 ,appendEnergy=True)
#     mfcc_feature = preprocessing.scale(mfcc_feature)
#     delta = calculate_delta(mfcc_feature)
#     combined = np.hstack((mfcc_feature,delta))
#     return combined

# def mfccc( y=None, sr=22050, S=None, n_mfcc=20, dct_type=2, norm="ortho", lifter=0):
#     # db scale to colour code
#     S = power_to_db(librosa.feature.melspectrogram(y=y, sr=sr))
#     M = scipy.fftpack.dct(S, axis=-2, type=dct_type, norm=norm)[..., :n_mfcc, :]
#     if lifter > 0:
#         # reshaping
#         LI = np.sin(np.pi * np.arange(1, 1 + n_mfcc, dtype=M.dtype) / lifter)
#         LI = util.expand_to(LI, ndim=S.ndim, axes=-2)
#         #formula
#         M *= 1 + (lifter / 2) * LI
#         return M
#     elif lifter == 0:
#         return M

# #===========================================================================
# #  when I will take the freq =0 (freq with low mag)
# def zero_crossing_rate(y,  frame_length=2048, hop_length=512, **kwargs):
#     y_framed = util.frame(y, frame_length=frame_length, hop_length=hop_length)
#     #reshaping
#     kwargs["axis"] = -2
#     # zero_crossing is a freq at which signal cross the axis
#     crossings = librosa.zero_crossings(y_framed, **kwargs)
#     # mean of crossing
#     zero_crossing_rate=np.mean(crossings, axis=-2, keepdims=True)
#     return zero_crossing_rate

# #===========================================================================
# # represent center og mass of freq and using to predict brightness in the audio
# def spectral_centroid(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, freq=None, win_length=None, window="hann",center=True, pad_mode="constant" ):
#    magnitude, n_fft = librosa.core.spectrum._spectrogram( y=y,S=S,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window,
#        center=center, pad_mode=pad_mode)
#        # S is the spectogram Magnitude
#        #
#    freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#    if freq.ndim == 1:
#     #  (just reshaping)
#       freq = util.expand_to(freq, ndim=magnitude.ndim, axes=-2)
#     # spectral centroif formela  (weighted freq (magnitude) * centeral freq)
#    spectral_centroid=np.sum(freq * util.normalize(magnitude, norm=1, axis=-2), axis=-2, keepdims=True)
#    return spectral_centroid

# #===========================================================================
# # distance between min and max and min freq
# def spectral_bandwidth(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window="hann", center=True,
#     pad_mode="constant", freq=None, centroid=None, norm=True, p=2 ):
#     S, n_fft = librosa.core.spectrum._spectrogram( y=y, S=S, n_fft=n_fft,hop_length=hop_length, win_length=win_length,window=window,center=center,
#        pad_mode=pad_mode)

#     centroid = spectral_centroid( y=y, sr=sr, S=S, n_fft=n_fft, hop_length=hop_length, freq=freq)
#     freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     if freq.ndim == 1:
#         deviation = np.abs(
#             np.subtract.outer(centroid[..., 0, :], freq).swapaxes(-2, -1)
#         )
#     else:
#      deviation = np.abs(freq - centroid)
#     if norm:
#         # S is a weighted freq
#         S = util.normalize(S, norm=1, axis=-2)
#         # formela
#     spectral_bandwidth=np.sum(S * deviation ** p, axis=-2, keepdims=True) ** (1.0 / p)
#     return spectral_bandwidth

# #===========================================================================
# # root mean squar for each frame  samples and spectogram
# def rms(y=None, S=None, frame_length=2048, hop_length=512):
#     # samples
#     if y is not None:
#         x = util.frame(y, frame_length=frame_length, hop_length=hop_length)
#         power = np.mean(np.abs(x) ** 2, axis=-2, keepdims=True)
#      # spectogram
#     elif S is not None:
#         x = np.abs(S) ** 2
#         power = 2 * np.sum(x, axis=-2, keepdims=True) / frame_length ** 2
#     rms=np.sqrt(power)
#     return rms

# #===========================================================================
# # Works as a filter(low pass and hieght pass filter)
# # ckeck if a center freq for a spectogram bins at least least has roll-of percentage (0.85) from power

# def spectral_rolloff(y=None, sr=22050, S=None, n_fft=2048, hop_length=512, win_length=None, window="hann", center=True,  pad_mode="constant",
#    freq=None, roll_percent=0.85 ):

#     S, n_fft = librosa.core.spectrum._spectrogram( y=y,S=S,n_fft=n_fft,hop_length=hop_length,win_length=win_length,window=window, center=center,
#        pad_mode=pad_mode,)
#     freq = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
#     # reshaping
#     if freq.ndim == 1:
#        freq = util.expand_to(freq, ndim=S.ndim, axes=-2)
#     # calculating total energy
#     total_energy = np.cumsum(S, axis=-2)
#     # calculating the edges
#     threshold = roll_percent * total_energy[..., -1, :]
#     #reshaoing
#     threshold = np.expand_dims(threshold, axis=-2)
#     #if total energy of centeral freq < threshold (it is out of my edges
#     # )
#     ind = np.where(total_energy < threshold, np.nan, 1)
#     spectral_rolloff=np.nanmin(ind * freq, axis=-2, keepdims=True)
#     return spectral_rolloff

# #===========================================================================

# def writeTocsv(data,csvName):
#     file = open(csvName, 'a', newline='')
#     writer = csv.writer(file)
#     writer.writerow(data.split(","))
#     file.close()


# def extract_features(directory, filename, csvName):
#     path = directory + filename

#     y,sr = librosa.load(path)
#     y, index = librosa.effects.trim(y)

#     rmse = rms(y=y)
#     spec_cent = spectral_centroid(y=y, sr=sr)
#     spec_bw = spectral_bandwidth(y=y, sr=sr)
#     rolloff = spectral_rolloff(y=y, sr=sr)
#     zcr = zero_crossing_rate(y)
#     mfcc = mfccc(y=y, sr=sr,n_mfcc=20)

#     # to_append = f'{filename},{np.mean(rmse)},{np.mean(spec_cent)},{np.mean(spec_bw)},{np.mean(rolloff)},{np.mean(zcr)}'
#     to_append = f'{np.mean(rmse)},{np.mean(spec_cent)},{np.mean(spec_bw)},{np.mean(rolloff)},{np.mean(zcr)}'

#     for e in mfcc:
#         to_append += f',{np.mean(e)}'

#     writeTocsv(to_append,csvName)

#     return to_append

# def startCSV(csvName):
#     header = 'filename rmse spectral_centroid spectral_bandwidth rolloff zero_crossing_rate'
#     for i in range(1, 21):
#         header += f' mfcc{i}'
#     # header += ' label'
#     header = header.split()
#     file = open(csvName, 'w', newline='')
#     writer = csv.writer(file)
#     writer.writerow(header)
#     file.close()


# def extract_features_array(filename):
#     feature = []

#     y,sr = librosa.load(filename)
#     y, index = librosa.effects.trim(y)

#     rmse = rms(y=y)
#     spec_cent = spectral_centroid(y=y, sr=sr)
#     spec_bw=spectral_bandwidth(y=y, sr=sr)
#     rolloff = spectral_rolloff(y=y, sr=sr)
#     zcr = zero_crossing_rate(y)
#     feature.append(np.mean(rmse))
#     feature.append(np.mean(spec_cent))
#     feature.append(np.mean(spec_bw))
#     feature.append(np.mean(rolloff))
#     feature.append(np.mean(zcr))

#     mfcc = mfccc(y=y, sr=sr,n_mfcc=20)
#     for e in mfcc:
#         feature.append(np.mean(e))

#     return feature
