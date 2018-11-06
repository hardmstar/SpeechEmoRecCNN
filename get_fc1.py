# -*-coding:utf-8-*-
# hardmstar@126.com
# load tensorflow graph 
# use tf model calculating features of train and validation files

import os
import tensorflow as tf
import numpy as np
from dataset import *

def get_fc1(graph_filename, load_filename):
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))
    graph = load_graph(dir_name+'/'+graph_filename)

    input_x = graph.get_tensor_by_name('model/input/input_x:0')
    fc1 = graph.get_tensor_by_name('model/output/fully_connection_layers/fc1:0')

    paths, labels = load_inputs(dir_name+'/'+load_filename)
    features = []
    with tf.Session(graph=graph) as sess:
        for i in range(len(paths)):
            # origin_feature = np.load(dir_name+'/'+paths[i])
            # read feature numpy files per wav file
            f_np_per_wav = load_np_per_wav(paths[i])
            f_np_per_wav = sess.run(f_np_per_wav)
            #f_np_per_wav = np.asarray(f_np_per_wav, dtype=np.float32)
            #f_np_per_wav = tf.convert_to_tensor(f_np_per_wav,dtype=tf.float32)
            #f_np_per_wav = tf.reshape(f_np_per_wav,shape=[None,300,23,1])
            out = sess.run(fc1, feed_dict={input_x:f_np_per_wav})
            features.append(out)
    return features, labels

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                input_map=None,
                return_elements=None,
                name='model',
                op_dict=None,
                producer_op_list=None)
        return graph

def load_inputs(filename):
    paths = []
    labels = []
    with open(filename, 'r')  as f:
        for line in f.readlines():
            line_s = line.split(' ')
            paths.append(line_s[0])
            labels.append(int(line_s[1]))
    return paths, labels

def load_np_per_wav(np_path):
    feature_nps = []
    files = os.listdir(np_path+'/np')
    for file in files:
        filename = np_path + '/np/' + file
        if os.path.isfile(filename):
            feature_nps.append(np.load(filename))
    feature_nps = tf.convert_to_tensor(np.asarray(feature_nps))
    feature_nps = tf.reshape(feature_nps,[-1, 300, 23, 1])
    return feature_nps
