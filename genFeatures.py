# -*-coding:utf-8 -*-
'''
hardmstar@126.com
generate features
tensorflow 1.4
use librosa rather than pyaudioanalysis needing libmagic
reference
[1] pyaudioanalysis.FeatureExtraction
[2] GeMAPS for Voice Research and Affective Computing
[3] Jitter and Shimmer Measurements for Speaker Recognition
'''

from dataset import *
import librosa
import numpy as np

eps = 0.00000001


def genFeatures(wav_file, wav_feature_folder):
    # generate features including st temporal features and spectrogram features
    y, sr = librosa.load(wav_file)  # y: audio time series , sr: sampling rate
    win = 0.03  # window: 30ms
    step = 0.010  # step : 10ms
    features = stFeatureExtraction(y, sr, win * sr, step * sr)


def stFeatureExtraction(y, sr, win, step):
    win = int(win)
    step = int(step)
    # signal normalization to [-1,1]
    y = np.double(y)
    y = (y - np.min(y)) / (np.max(y) - np.min(y))
    N = len(y)  #

    cur_p = 0
    cnt_fr = 0
    features = []
    while (cur_p + win - 1 < N):
        cnt_fr += 1
        y_frame = y[cur_p:cur_p + win]
        cur_p += step

        features_frame = []
        # temporal features
        features_frame.append(stZCR(y_frame))
        features_frame.append(stEnergy)


def stZCR(frame):
    # computing zero crossing rate
    count = len(frame)
    count_z = np.sum(np.abs(np.diff(np.sign(frame)))) / 2
    return (np.float64(count_z) / np.float64(count - 1))


def stEnergy(frame):
    return (np.sum(frame ** 2) / np.float64(len(frame)))


def stShimmerDB(frame):
    '''
     amplitude shimmer 振幅扰动度
     expressed as variability of the peak-to-peak amplitude in decibels 分贝
     [3]
    '''
    count = len(frame)
    sigma = 0
    for i in range(count):
        if i == count - 1:
            break
        sigma += np.abs(20 * (np.log10(np.abs(frame[i + 1] / (frame[i] + eps)))))
    return np.float64(sigma) / np.float64(count - 1)

def stShimmerRelative(frame):
	'''
	shimmer relative is defined as average absolute difference between the amplitude
	of consecutive periods divided by the average amplitude, expressed as percentage
	[3]
	'''
	count = len(frame)
	sigma_diff = 0
	sigma_sum = 0
	for i in range(count):
		if i < count-1:
			sigma_diff += np.abs(np.abs(frame[i]) - np.abs(frame[i+1]))
		sigma_sum += np.abs(frame[i])
	return np.float64(sigma_diff / (count-1)) / np.float64(sigma_sum / count + eps)
		
def main():
    berlin_dataset = Dataset('berlin')
    for wav in os.listdir(berlin_dataset.wav_files):
        wav_file = '%s/%s' % (berlin_dataset.wav, wav)
        wav_feature_folder = '%s/%s' % (berlin_dataset.NN_inputs, wav)

        genFeatures(wav_file, wav_feature_folder)


x = [-0.9, 0.1, 0.5]
y = stShimmerRelative(x)
x = y
