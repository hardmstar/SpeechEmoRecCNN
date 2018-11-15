# -*-coding:utf-8 -*-
'''
hardmstar@126.com
generate features
'''

from dataset import *
import numpy as np
import os
import csv


def genFeatures(wav, wav_file, wav_feature_folder):
    '''

    :param wav: wav file name, string
    :param wav_file: wav file path, string
    :param wav_feature_folder: the folder contains wav features, include frame features and overall features
    :return: none
    '''
    if not os.path.exists(wav_feature_folder):
        os.makedirs(wav_feature_folder)
    cmd = ('smilextract -C E:/useful/audio/opensmile-2.3.0/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -I ' 
          + wav_file + ' -lldcsvoutput ' + wav_feature_folder + wav[:-4] + '_frame.csv' + ' -instname ' + wav 
          + ' -timestampcsvlld 0 -headercsvlld 0')
    os.system(cmd)
    cmd = ('smilextract -C E:/useful/audio/opensmile-2.3.0/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -I ' 
          + wav_file + ' -csvoutput ' + wav_feature_folder + wav[:-4] + '_statistic.csv' 
          + ' -appendcsv 0 -timestampcsv 0 -headercsv 0')
    os.system(cmd)


def read_frame_csv(type, wav, wav_feature_folder):
    # read csv files and seprate them in 300 frames a file.
    # short than 300 frames using np.pad() to complete
    # longer than 300 frames, cut into blocks every 300 frames,
    # throw the last part < 150, otherwise keep.
    features_np_path = wav_feature_folder + 'np/'
    if not os.path.exists(features_np_path):
        os.makedirs(features_np_path)
    csvFile = open(wav_feature_folder + wav[:-4] + '_frame.csv', 'r')
    reader = csv.reader(csvFile)
    features_np = []
    for item in reader:
        features_np.append(item[0].split(';')[1:])
    features_np = np.array(features_np, dtype='float32')
    frames = len(features_np)
    blocks, last_block = divmod(frames, 300)
    if blocks == 0:
        features_np = np.pad(features_np, ((0, 300 - frames), (0, 0)), 'constant')
        np.save(features_np_path + str(return_class(type, wav)[1]) + "0.npy", features_np)
    else:
        for i in range(blocks):
            np.save(features_np_path + str(return_class(type, wav)[1]) + str(i) + '.npy', features_np[i * 300:(i + 1) * 300])
        if last_block >= 150:
            features_np = np.pad(features_np, ((0, 300 - last_block), (0, 0)), 'constant')
            np.save(features_np_path + str(return_class(type, wav)[1]) + str(blocks) + '.npy', features_np[blocks * 300:(blocks+1) * 300 ])


def main():
    berlin_dataset = Dataset('berlin')
    # delete all numpy files of wav feature
    berlin_dataset.delete_features_np()
    for wav in os.listdir(berlin_dataset.wav_files):
        wav_file = '%s/%s' % (berlin_dataset.wav_files, wav)
        wav_feature_folder = '%s/%s/' % (berlin_dataset.NN_inputs, wav)
        # generate wav frame and overall features with opensmile
        genFeatures(wav, wav_file, wav_feature_folder)
        # split frame features into 300 frames a block, 23 features per frame
        read_frame_csv(berlin_dataset.type, wav, wav_feature_folder, )
    # generate train_segments.txt and val_segments.txt
    berlin_dataset.record_train_val_segment_files()


main()
