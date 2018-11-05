# -*-coding:utf-8-*-
# cnn with residual block
from dataset import *
import tensorflow as tf
import numpy as np
import os
from wavDataGenerator import *


def residual_block(x, num_input_filters, num_output_filters, block_num):
    # define weights and biases of this residual block
    w_conv_1 = tf.get_variable(name='rs_block_' + str(block_num) + '_w_conv_1',
                               shape=[3, 3, num_input_filters, num_output_filters], dtype=tf.float32)
    w_conv_2 = tf.get_variable(name='rs_block_' + str(block_num) + '_w_conv_2',
                               shape=[3, 3, num_input_filters, num_output_filters], dtype=tf.float32)
    b_conv_1 = tf.get_variable(name='rs_block_' + str(block_num) + '_b_conv_1', shape=[num_output_filters],
                               dtype=tf.float32)
    b_conv_2 = tf.get_variable(name='rs_block_' + str(block_num) + '_b_conv_2', shape=[num_output_filters],
                               dtype=tf.float32)

    # implementing residual block logic
    input_1 = tf.contrib.layers.batch_norm(x)
    input_1 = tf.nn.relu(input_1)
    weight_layer_1 = tf.nn.conv2d(input_1, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1
    intermediate = tf.contrib.layers.batch_norm(weight_layer_1)
    num_input_filters = tf.nn.relu(intermediate)
    weight_layer_2 = tf.nn.conv2d(intermediate, w_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2

    # elementwise addition of x and weight_layer_2
    if num_input_filters != num_output_filters:
        w_conv_increase = tf.get_variable('rs_block_' + str(block_num) + '_w_conv_increase',
                                          shape=[1, 1, num_input_filters], dtype=tf.float32)
        b_conv_increase = tf.get_variable('rs_block_' + str(block_num) + '_b_conv_increase', shape=[num_output_filters],
                                          dtype=tf.float32)
    output = tf.add(x, weight_layer_2)
    return output


def recurrent_neural_network(x, NUM_CLASSES):
    rs_block_1 = residual_block(x, 1, 32, 1)
    rs_block_2 = residual_block(rs_block_1, 32, 32, 2)
    rs_block_3 = tf.nn.max_pool(rs_block_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    rs_block_4 = residual_block(rs_block_3, 32, 64, 4)
    rs_block_5 = residual_block(rs_block_4, 64, 64, 5)
    rs_block_6 = tf.nn.max_pool(rs_block_5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    rs_block_7 = residual_block(rs_block_5, 64, 128, 7)
    rs_block_8 = residual_block(rs_block_7, 128, 128, 8)
    rs_block_9 = tf.nn.max_pool(rs_block_8, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    rs_block_10 = residual_block(rs_block_9, 128, 256, 10)
    rs_block_11 = residual_block(rs_block_10, 256, 256, 11)
    rs_block_12 = tf.nn.max_pool(rs_block_11, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    num_features = rs_block_12.shape[1] * rs_block_12.shape[2]
    flattened = tf.reshape(rs_block_12, [-1, num_features])

    # fully connected layers
    w_fc1 = tf.get_variable('w_fc1', shape=[num_features, 128], dtype=tf.float32)
    w_fc2 = tf.get_variable('w_fc2', shape=[128, NUM_CLASSES], dtype=tf.float32)
    b_fc1 = tf.get_variable('b_fc1', shape=[128], dtype=tf.float32)
    b_fc2 = tf.get_variable('b_fc2', shape=[NUM_CLASSES], dtype=tf.float32)

    fully_connected_1 = tf.matmul(flattened, w_fc1) + b_fc1
    fully_connected_2 = tf.matmul(fully_connected_1, w_fc2) + b_fc2

    return fully_connected_2


def train_seesion(ob_dataset, speaker, checkpoints_path, num_epochs, batch_size):
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    num_classes = len(ob_dataset.classes)
    # wavDataGenerator(txt_file, batch_size, num_class, shuffle=True, buffer_size=1000)
    tr_data = wavDataGenerator(ob_dataset.root + speaker + '/train_segments.txt',
                               batch_size=batch_size,
                               num_class=num_classes)
    val_data = wavDataGenerator(ob_dataset.root + speaker + '/val_segments.txt',
                                batch_size=batch_size,
                                num_class=num_classes)

    # 可重新初始化迭代器可以通过多个不同的 Dataset 对象进行初始化。
    # https://www.tensorflow.org/programmers_guide/datasets?hl=zh-cn
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
    next_batch = iterator.get_next()
    # option of initializing train and validation iterator
    train_init_op = iterator.make_initializer(tr_data.data)
    val_init_op = iterator.make_initializer(val_data.data)

    ########## test iterator
    sess = tf.Session()
    for _ in range(20):
        sess.run(train_init_op)
        for _ in range(100):
            print(sess.run(next_batch))

        sess.run(val_init_op)
        for _ in range(100):
            print(sess.run(next_batch))
    ##########
    x = tf.placeholder('float32',[None, 300, 23, 1], name='input_x')
    y = tf.placeholder('float32', [None, num_classes], name='label_y')

    logits = recurrent_neural_network(x,num_classes)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer()
    training = optimizer.minimize(loss)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # for epoch in range(num_epochs)
        for epoch in range(2):
            total_cost = 0

            # for i in range(NUM_EXAMPLES/BATCH_SIZE):
            for i in range(2):
                batch_x, batch_y = sess.run(next_batch)
                _, batch_cost = sess.run([training, loss], feed_dict={x: batch_x, y: batch_y})
                total_cost += batch_cost
                if i % 25 == 0:
                    print(i)
            print('Epoch ', epoch, ' Cost: total_cost')

            # predict validation accuray after every epoch
            sum_accuracy_validation = 0.0
            sum_i = 0
            # for i in range(int(dataset_validation_features.shape[0]/BATCH_SIZE))
            batch_x, batch_y = sess.run(next_batch)
            y_predicted = tf.nn.softmax(logits)
            correct = tf.equal(tf.argmax(y_predicted, 1), tf.argmax(y, 1))
            accuracy_function = tf.reduce_mean(tf.cast(correct, 'float'))
            accuracy_validation = accuracy_function.eval({x: batch_x, y: batch_y})
            sum_accuracy_validation += accuracy_validation
            sum_i += 1
            print('validation accuracy in epoch ', epoch, ': ', accuracy_validation, 'sum_i: ', sum_i,
                  'sum_accuracy_validation', sum_accuracy_validation)

if __name__ == '__main__':
    berlin = Dataset('berlin')

    learning_rate = 1e-3
    num_epochs = 300
    batch_size = 30
    dropout_rate = 0.5

    filewriter_path = berlin.root + '/residual/tensorboard'
    checkpoints_path = berlin.root + '/residual/checkpoints_path'

    for speaker in berlin.speakers:
        # train_seesion(ob_dataset, speaker, checkpoints_path, num_epochs, batch_size)
        train_seesion(berlin, speaker, checkpoints_path, num_epochs, batch_size)
