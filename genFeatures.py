# -*-coding:utf-8 -*-
# hardmstar@126.com
# generate features
# tensorflow 1.4
# use librosa rather than pyaudioanalysis needing libmagic
import dataset
import librosa
import numpy as np
def genFeatures(wav_file, wav_feature_folder):
	# generate features including st temporal features and spectrogram features
	y,sr = librosa.load(wav_file) # y: audio time series , sr: sampling rate
	win = 0.03 # window: 30ms
	step = 0.010 # step : 10ms
	features = stFeatureExtraction(y, sr, win*sr, step*sr)

def stFeatureExtraction(y, sr, win, step):
	win = int(win)
	step = int(step)
	# signal normalization to [-1,1]
	y = np.double(y)
	y = (y-np.min(y))/(np.max(y)-np.min(y))
	N = len(y) # 
	
	cur_p = 0
	cnt_fr = 0
	features = []
	while (cur_p + win - 1 <N):
		cnt_fr += 1
		y_frame = y[cur_p:cur_p+win]
		cur_p += step
		
		features_frame = []
		# temporal features
		features_frame.append(stZCR(y_frame))
		features_frame.append(stEnergy)
		
		
		
def stZCR(frame):
	# computing zero crossing rate
	count = len(frame)
	count_z = np.sum(np.abs(np.diff(np.sign(frame))))/2
	return (np.float64(count_z)/np.float64(count-1))

def stEnergy(frame):
	return (np.sum(frame ** 2)/np.float64(len(frame)))

def main():
	berlin_dataset = Dataset('berlin')
	for wav in os.listdir(berlin_dataset.wav_files):
		wav_file = '%s/%s' %(berlin_dataset.wav, wav)
		wav_feature_folder = '%s/%s' %(berlin_dataset.NN_inputs,wav)
		
		genFeatures(wav_file, wav_feature_folder)
		