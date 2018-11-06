# -*-coding:utf-8-*-
# feature selection and svm
# hardmstar@126.com
import numpy as np
import get_fc1
import tensorflow as tf
from dataset import *

if __name__ == '__main__':
    berlin = Dataset('berlin')

    # generate residual cnn features and use numpy save
    berlin.generate_model_path('residual')
    berlin.generate_model_features_np()
    for i in range(len(berlin.speakers)):
        train_features, labels = get_fc1.get_fc1(berlin.model_path[i],
                berlin.train_path[i])
        np.save(berlin.train_segments_model_path[i],train_features)
        print('save speaker {} model train features segments in {}'.format(berlin.speakers[i], berlin.train_segments_model_path))
        val_features, labels = get_fc1.get_fc1(berlin.model_path[i],
                berlin.val_path[i])
        np.save(berlin.val_segments_model_path[i],val_features)
        print('save speaker {} model val features segments in {}'.format(berlin.speakers[i], berlin.val_segments_model_path))
