# -*-coding:utf-8-*-
# hardmstar@126.com
# load tensorflow graph 
# use tf model calculating features of train and validation files

import os
import tensorflow as tf
import numpy as np
import utils
from dataset import *

def get_fc1(graph_filename, load_filename):
    dir_name = os.path.dirname(os.path.abspath(get_fc1.py))
    graph = load_graph(dir_name+'/'+graph_filename)

    input_x = graph.get_tensor_by_name('input/input_x:0')
    fc1 = graph.get_tensor_by_name('fc1:0')

    paths, labels = load_inputs(dir_name+'/'+load_filename)
    features = []
    with tf.Session(graph=graph) as sess:
        for i in range(len(paths)):
            origin_feature = np.load(dir_name+'/'+path[i])
            out = sess.run(fc1, feed_dict={input_x:origin_feature})
            features.append(out)
    return features, labels

def load_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def,
                import_map=None,
                return_element=None,
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


