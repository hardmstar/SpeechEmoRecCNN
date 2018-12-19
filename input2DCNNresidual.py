# -*-coding:utf-8-*-
# cnn with residual block
from dataset import *
import tensorflow as tf
import numpy as np
import os
from wavDataGenerator import *

learning_rate = 1e-5


def residual_block(x, num_input_filters, num_output_filters, block_num):
    with tf.variable_scope('residual_block_'+str(block_num)):
        # define weights and biases of this residual block
        w_conv_1 = tf.get_variable(name='rs_block_' + str(block_num) + '_w_conv_1',
                                   shape=[3, 3, num_input_filters, num_output_filters], dtype=tf.float32)
        w_conv_2 = tf.get_variable(name='rs_block_' + str(block_num) + '_w_conv_2',
                                   shape=[3, 3, num_output_filters, num_output_filters], dtype=tf.float32)
        b_conv_1 = tf.get_variable(name='rs_block_' + str(block_num) + '_b_conv_1', shape=[num_output_filters],
                                   dtype=tf.float32)
        b_conv_2 = tf.get_variable(name='rs_block_' + str(block_num) + '_b_conv_2', shape=[num_output_filters],
                                   dtype=tf.float32)
    
        # implementing residual block logic
        input_1 = tf.contrib.layers.batch_norm(x)
        input_1 = tf.nn.relu(input_1)
        weight_layer_1 = tf.nn.conv2d(input_1, w_conv_1, strides=[1, 1, 1, 1], padding='SAME') + b_conv_1
        intermediate = tf.contrib.layers.batch_norm(weight_layer_1)
        intermediate = tf.nn.relu(intermediate)
        weight_layer_2 = tf.nn.conv2d(intermediate, w_conv_2, strides=[1, 1, 1, 1], padding='SAME') + b_conv_2
    
        # elementwise addition of x and weight_layer_2
        if num_input_filters != num_output_filters:
            w_conv_increase = tf.get_variable('rs_block_' + str(block_num) + '_w_conv_increase',
                                              shape=[1, 1, num_input_filters,num_output_filters],
                                              dtype=tf.float32)
            b_conv_increase = tf.get_variable('rs_block_' + str(block_num) + '_b_conv_increase',
                                              shape=[num_output_filters],
                                              dtype=tf.float32)
            x = tf.nn.conv2d(x, w_conv_increase, strides=[1,1,1,1], padding='SAME') + b_conv_increase
    with tf.name_scope('residual_add_output'+str(block_num)):
        output = tf.add(x, weight_layer_2,name='add')
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

    num_features = rs_block_12.shape[1] * rs_block_12.shape[2] * rs_block_12.shape[3]
    flattened = tf.reshape(rs_block_12, [-1, num_features])

    with tf.variable_scope('recurrent_neural_network_full_connection'):
        # fully connected layers
        w_fc1 = tf.get_variable('w_fc1', shape=[num_features, 128], dtype=tf.float32)
        w_fc2 = tf.get_variable('w_fc2', shape=[128, NUM_CLASSES], dtype=tf.float32)
        b_fc1 = tf.get_variable('b_fc1', shape=[128], dtype=tf.float32)
        b_fc2 = tf.get_variable('b_fc2', shape=[NUM_CLASSES], dtype=tf.float32)
    with tf.name_scope('fully_connection_layers'):
        # fully_connected_1 = tf.matmul(flattened, w_fc1) + b_fc1
        # fully_connected_2 = tf.matmul(fully_connected_1, w_fc2) + b_fc2
        fully_connected_1 = tf.nn.xw_plus_b(flattened, w_fc1, b_fc1,name='fc1')
        fully_connected_2 = tf.nn.xw_plus_b(fully_connected_1, w_fc2, b_fc2,name='fc2')

    return fully_connected_2


def train_seesion(ob_dataset, speaker, filewriter_path, checkpoints_path, num_epochs, batch_size):
    if not os.path.isdir(checkpoints_path):
        os.makedirs(checkpoints_path)

    num_classes = len(ob_dataset.classes)
    # wavDataGenerator(txt_file, batch_size, num_class, shuffle=True, buffer_size=1000)
    tr_data = wavDataGenerator(ob_dataset.root + speaker + '/train_segments.txt',
                               batch_size=batch_size,
                               num_class=num_classes)
    val_data = wavDataGenerator(ob_dataset.root + speaker + '/val_segments.txt',
                                batch_size=batch_size,
                                num_class=num_classes,
                                shuffle=False)

    # 可重新初始化迭代器可以通过多个不同的 Dataset 对象进行初始化。
    # https://www.tensorflow.org/programmers_guide/datasets?hl=zh-cn
    iterator = tf.data.Iterator.from_structure(tr_data.data.output_types, tr_data.data.output_shapes)
    next_batch = iterator.get_next()
    # option of initializing train and validation iterator
    train_init_op = iterator.make_initializer(tr_data.data)
    val_init_op = iterator.make_initializer(val_data.data)

    ########## test iterator
    '''
    sess = tf.Session()
    for _ in range(20):
        sess.run(train_init_op)
        for _ in range(1):
            print(sess.run(next_batch))

        sess.run(val_init_op)
        for _ in range(1):
            print(sess.run(next_batch))
    '''
    ##########
    with tf.name_scope('input'):
        x = tf.placeholder('float32', [None, 300, 23, 1], name='input_x')
        y = tf.placeholder('float32', [None, num_classes], name='label_y')
    with tf.device('/cpu:0'):
        with tf.name_scope('output'):
            logits = recurrent_neural_network(x, num_classes)
    with tf.name_scope('cross_entropy'):
        # loss function
        num = tf.convert_to_tensor(tf.constant(1e-10),dtype=tf.float32)
        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits+num, labels=y))
    # tf.summary.scalar('loss', loss)
    loss_summary = tf.summary.histogram("loss", loss)

    with tf.name_scope('train_optimize'):
        # optimizer

        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(loss)

    # accuracy
    with tf.name_scope('accuracy'):
        correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    # tf.summary.scalar('accuracy', accuracy)
    accuracy_summary = tf.summary.histogram('accuracy', accuracy)

    # merged_summary = tf.summary.merge_all()
    merged_summary = tf.summary.merge([accuracy_summary,loss_summary])

    # initialize filewriter
    writer = tf.summary.FileWriter(filewriter_path)

    # initialize an saver for store model checkpoints
    saver = tf.train.Saver()

    # cpu resources
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
    config = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=True)

    with tf.Session(config=config) as sess:
        if os.path.exists(checkpoints_path+'/model.ckpt'):
            saver.restore(sess, checkpoints_path+'/model.ckpt')
        else:
            sess.run(tf.global_variables_initializer())

        # add the model graph to tensorboard
        writer.add_graph(sess.graph)

        for epoch in range(num_epochs):
        # for epoch in range(1):
            total_cost = 0
    
            # train start
            print(speaker,' train start. epoch:', epoch)
            sess.run(train_init_op)
            train_acc = 0
            train_count = 0
            train_loss = 0
            train_batches_num = tr_data.data_size // batch_size
            for i in range(train_batches_num):
            # for i in range(1):
                batch_x, batch_y = sess.run(next_batch)
                # logits_return, y_prediction_return = sess.run((logits, y_prediction),feed_dict={x: batch_x, y: batch_y})
                train_op_return, train_acc_value, train_loss_value = sess.run((train_op, accuracy, loss),
                                                                              feed_dict={x: batch_x, y: batch_y})
                train_loss += train_loss_value
                train_acc += train_acc_value
                train_count += 1
                ###### summary#################
                summary_ = sess.run(merged_summary, feed_dict={x :batch_x, y: batch_y})
                writer.add_summary(summary_,epoch * train_batches_num + i)
                ###############################
                ###### every layer ############
                '''
                residual_add_outputlayer = sess.graph.get_tensor_by_name('output/residual_add_output2/add:0')
                print('rs2',sess.run(residual_add_outputlayer,feed_dict={x:batch_x}))
                residual_add_outputlayer = sess.graph.get_tensor_by_name('output/residual_add_output5/add:0')
                print('rs5',sess.run(residual_add_outputlayer,feed_dict={x:batch_x}))
                residual_add_outputlayer = sess.graph.get_tensor_by_name('output/residual_add_output8/add:0')
                print('rs8',sess.run(residual_add_outputlayer,feed_dict={x:batch_x}))
                residual_add_outputlayer = sess.graph.get_tensor_by_name('output/residual_add_output11/add:0')
                print('rs11',sess.run(residual_add_outputlayer,feed_dict={x:batch_x}))
                fc1_layer = sess.graph.get_tensor_by_name('output/fully_connection_layers/fc1:0')
                print('fc1',sess.run(fc1_layer,feed_dict={x:batch_x}))
                '''
                ################################
                print('speaker {} in epoch {}, batch {}'.format(speaker, epoch, i))
            train_loss /= train_count
            train_acc /= train_count
            print('speaker {} in epoch {}, train_acc={}, train_loss={}'.format(speaker, epoch, train_acc, train_loss))


            scope_now = tf.get_variable_scope()
            scope_now.reuse_variables()
            # validation start
            print(speaker,' start validation, epoch ',epoch)
            sess.run(val_init_op)
            test_acc = 0
            test_count = 0
            test_loss = 0
            for _ in range(val_data.data_size // batch_size):
            # for _ in range(1):
                batch_x, batch_y = sess.run(next_batch)
                acc, loss_value = sess.run((accuracy, loss), feed_dict={x: batch_x, y: batch_y})
                test_loss += loss_value
                test_acc += acc
                test_count += 1
            test_acc /= test_count
            test_loss /= test_count
            print('speaker {} in epoch {}, test_acc={}, test_loss={}'.format(speaker, epoch, test_acc, test_loss))
            # save checkpoint of the model
            saver.save(sess, checkpoints_path+'/model.ckpt')
        writer.close()
        graph = tf.graph_util.convert_variables_to_constants(sess,sess.graph_def, ['output/fully_connection_layers/fc2'])
        tf.train.write_graph(graph, '.',ob_dataset.root+speaker+'/residual_model.pb',as_text=False)
        print(speaker,' done train.')
            
if __name__ == '__main__':
    berlin = Dataset('berlin')

    num_epochs = 200
    batch_size = 30
    dropout_rate = 0.5

    filewriter_path = berlin.root + '/residual/tensorboard/'
    checkpoints_path = berlin.root + '/residual/checkpoints_path/'
	
    speaker = '03'
    train_seesion(berlin, speaker, filewriter_path+speaker, checkpoints_path+speaker, num_epochs, batch_size)
    '''for speaker in berlin.speakers:
        if not (speaker=='03' or speaker=='08' or speaker=='09'):
            
            train_seesion(berlin, speaker, filewriter_path+speaker, checkpoints_path+speaker, num_epochs, batch_size)
        #train_seesion(berlin, '08', filewriter_path + speaker, checkpoints_path + speaker, num_epochs, batch_size)
        #train_seesion(berlin, '09', filewriter_path + speaker, checkpoints_path + speaker, num_epochs, batch_size)
'''

