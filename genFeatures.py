# -*-coding:utf-8 -*-
'''
hardmstar@126.com
generate features
'''

from dataset import *
import numpy as np
import os


def genFeatures(wav, wav_file, wav_feature_folder):
    if not os.path.exists(wav_feature_folder):
        os.makedirs(wav_feature_folder)
    cmd = 'smilextract -C F:/useful/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -I ' \
          + wav_file + ' -lldcsvoutput ' + wav_feature_folder + wav[:-4] + '_frame.csv' + ' -instname ' + wav \
          + ' -timestampcsvlld 0 -headercsvlld 0'
    os.system(cmd)
    cmd = 'smilextract -C F:/useful/opensmile-2.3.0/config/gemaps/eGeMAPSv01a.conf -I ' \
          + wav_file + ' -csvoutput ' + wav_feature_folder + wav[:-4] + '_statistic.csv' \
          + ' -appendcsv 0 -timestampcsv 0 -headercsv 0'
    os.system(cmd)


def main():
    berlin_dataset = Dataset('berlin')
    for wav in os.listdir(berlin_dataset.wav_files):
        wav_file = '%s/%s' % (berlin_dataset.wav_files, wav)
        wav_feature_folder = '%s/%s/' % (berlin_dataset.NN_inputs, wav)

        genFeatures(wav, wav_file, wav_feature_folder)


main()
