# -*-coding:utf-8 -*-
# hardmstar@126.com
# wavDataGenerator.py including class wavDataGenerator
import tensorflow as tf
import numpy as np
import os


class wavDataGenerator:
    def __init__(self, txt_file, batch_size, num_class, shuffle=True, buffer_size=1000):
        '''
        args:
            txt_file: txt file contains feature file paths
            batch_size: numbers of data per batch
            num_class: number of classes of the corpus
            shuffle: whether or not to shuffle the data in the dataset and the initial file list
            buffer_size: number of data used as buffer for tensorflow shuffling of the dataset
        '''
        self.txt_file = txt_file
        self.batch_size = batch_size
        self.num_class = num_class
        # get self.data, self.data is tensorflow dataset
        self._read_txt_file()
        if shuffle:
            self.data = self.data.shuffle(buffer_size=buffer_size)

        self.data = self.data.batch(batch_size)

    def _read_txt_file(self):
        '''
        read XX_segments.txt, extract numpy feature segment file paths,
        then accouring to the path, get the features
        '''
        np_file_paths = []
        labels = []
        with open(self.txt_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                item = line.split(' ')
                np_file_paths.append(item[0])
                labels.append(tf.one_hot(int(item[1]), self.num_class))

        self.data_size = len(labels)

        # features
        np_features = []
        for np_file_path in np_file_paths:
            np_features.append(np.load(np_file_path))

        # convert numpy array into tf tensor
        np_features = tf.convert_to_tensor(np.array(np_features))
        np_features = tf.reshape(np_features, [-1, 300, 23, 1])
        labels = tf.convert_to_tensor(labels)
        self.data = tf.data.Dataset.from_tensor_slices((np_features, labels))
