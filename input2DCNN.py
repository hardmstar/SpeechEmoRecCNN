# -*-coding:utf-8 -*-
# hardmstar@126.com
# use 2D cnn. input 300 frames.
from dataset import *
import tensorflow as tf
import numpy as np
import os


class wavDataGenerator:
    def __init___(self, txt_file, batch_size, num_class, shuffle=True, buffer_size=1000):
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

        # features
        np_features = []
        for np_file_path in np_file_paths:
            np_features.append(np.load(np_file_path))

        # convert numpy array into tf tensor
        np_features = tf.convert_to_tensor(np.array(np_features))
        labels = tf.convert_to_tensor(labels)
        self.data = tf.data.Dataset.from_tensor_slices((np_features, labels))


def train_session(ob_dataset, speaker, checkpoints_path, batch_size):
    '''
    train session of ser system
    '''

    # create checkpoint path
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)
    num_speakers = len(ob_dataset.speakers)
    train_txt_file = ob_dataset.root + speaker + '/train_segments.txt'
    # loading and preprocessing train and validation data on the cpu
    with tf.device('/cpu:0'):
        # def __init___(self, txt_file, batch_size, num_class, shuffle=True, buffer_size=1000):
        tr_data = wavDataGenerator(train_txt_file, batch_size, num_speakers)
        val_data = wavDataGenerator(ob_dataset.root + speaker + '/val_segments.txt',
                                    batch_size=batch_size,
                                    num_class=len(ob_dataset.speakers))

        # create an reuseble initializable iterator
        iterator = tf.data.Iterator.from_structure(tr_data.data.output_types,
                                                   tr_data.data.output_shapes)
        next_batch = iterator.get_next()

    # option of initializing train and validation iterator
    train_init_op = iterator.make_initializer(tr_data.data)
    val_init_op = iterator.make_initializer(val_data.data)

    # https://blog.csdn.net/sinat_33761963/article/details/57416517
    y_ = tf.placeholder("float", [None, len(ob_dataset.speakers)])
    # convolution layer and pool layer
    w_conv1 = weight_variable([5, 5, 1, 32])  # ? ? ? why 5
    b_conv1 = bias_variable([32])
    # conv output : 300 * 23 * 32
    h_conv1 = tf.nn.relu(conv2d(tr_data.data.output_shapes, w_conv1) + b_conv1)
    # pool output : 250 * 17 * 32
    h_pool1 = max_pool_2x2(h_conv1)

    # full connection layer
    w_fc1 = weight_variable([250 * 17 * 32, 1024])
    b_fc1 = bias_variable([1024])
    # 数据拉平为一行
    h_pool1_flat = tf.reshape(h_pool1, [-1, 250 * 17 * 32])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool1_flat, w_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, dropout_rate)
    # fc 2
    w_fc2 = weight_variable([1024, len(ob_dataset.speakers)])
    b_fc2 = bias_variable([len(ob_dataset.speakers)])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

    # optimization
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    # calculate accuracy
    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(2):
            sess.run(train_init_op)
            for step in range(2):
                data_batch, label_batch = sess.run(next_batch)
                train_accuracy = accuracy.eval(feed_dict={x: data_batch, y_: label_batch, keep_prob: 1.0})
                print("step %d, train_accuracy %g" % (step, train_accuracy))
                train_step.run(feed_dict: {x: data_batch, y_: label_batch, keep_prob: 0.5})

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.7)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(a, w):
    return tf.nn.conv2d(a, w, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(a):
    return tf.nn.max_pool(a, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    if __name__ == '__main__':
        berlin = Dataset('berlin')
        # if not exists train and validation file,
        # berlin.get_dataset('berlin')

        # learning params
        learning_rate = 1e-3
        num_epochs = 300
        batch_size = 30
        # Network params
        dropout_rate = 0.5

        # Path for tf.summary.FileWriter and to store model checkpoints
        filewriter_path = berlin.root + '/tmp/tensorboard'
        checkpoints_path = berlin.root + '/tmp/checkpoints'

        # x = tf.placeholder(tf.float32, [None, 300,], name='input')
        # y = tf.placeholder(tf.float32, [None, len(berlin.classes]))
        for speaker in berlin.speakers:
            train_session(berlin, speaker,checkpoints_path,batch_size)
