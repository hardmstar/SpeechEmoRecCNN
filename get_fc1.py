# -*-coding:utf-8-*-
# hardmstar@126.com
# load tensorflow graph 
# use tf model calculating features of train and validation files

import os
import tensorflow as tf
import numpy as np
from dataset import *
from sklearn.mixture import GaussianMixture
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import PredefinedSplit
from sklearn.externals import joblib
import pandas as pd
import csv
import matplotlib.pyplot as plt
import time


num_C = 9
Cs = 10 ** np.arange(num_C) * 1e-4

# gamma for rbf
gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
num_components = np.arange(2,32)
list_neighbors = np.arange(10,200,10)

def get_fc1(speakers, graph_filename='residual_model.pb'):
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))
    for speaker in speakers:
        print('speaker '+ speaker +' now')
        graph = load_graph(dir_name + '/EMODB/'+ speaker + '/' + graph_filename)
        speaker_dir = dir_name + '/EMODB/'+ speaker + '/'
        input_x = graph.get_tensor_by_name('model/input/input_x:0')
        fc1 = graph.get_tensor_by_name('model/output/fully_connection_layers/fc1:0')

        train_np_features_fc1, train_y_true = load_features(speaker_dir + 'train_segments.txt')
        train_np_features_fc1 = np.asarray(train_np_features_fc1)
        train_np_features_fc1 = train_np_features_fc1[:, :, :, np.newaxis]

        val_np_features_fc1, val_y_true = load_features(speaker_dir + 'val_segments.txt')
        val_np_features_fc1 = np.asarray(val_np_features_fc1)
        val_np_features_fc1 = val_np_features_fc1[:, :, :, np.newaxis]

        with tf.Session(graph=graph) as sess:
            out = sess.run(fc1, feed_dict={input_x:train_np_features_fc1})
            np.save(speaker_dir+'train_np_features_fc1.npy', out)
            out = sess.run(fc1, feed_dict={input_x:val_np_features_fc1})
            np.save(speaker_dir+'val_np_features_fc1.npy', out)

'''
    paths, labels = load_inputs(dir_name + '/' + load_filename)
    features = []
    with tf.Session(graph=graph) as sess:
        for i in range(len(paths)):
            # origin_feature = np.load(dir_name+'/'+paths[i])
            # read feature numpy files per wav file
            f_np_per_wav = load_np_per_wav(paths[i])
            f_np_per_wav = sess.run(f_np_per_wav)
            # f_np_per_wav = np.asarray(f_np_per_wav, dtype=np.float32)
            # f_np_per_wav = tf.convert_to_tensor(f_np_per_wav,dtype=tf.float32)
            # f_np_per_wav = tf.reshape(f_np_per_wav,shape=[None,300,23,1])
            out = sess.run(fc1, feed_dict={input_x: f_np_per_wav})
            # print(out)
            features.append(out)
    return features, labels
    '''

def fc_accuracy(speakers, graph_filename='residual_model.pb'):
    accuracies = np.zeros(len(speakers))
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))
    for (i,speaker) in zip(range(len(speakers)), speakers):
        print('speaker '+ speaker +' now')
        graph = load_graph(dir_name + '/EMODB/'+ speaker + '/' + graph_filename)
        input_x = graph.get_tensor_by_name('model/input/input_x:0')
        fc2 = graph.get_tensor_by_name('model/output/fully_connection_layers/fc2:0')
        # none sense input_y = graph.get_tensor_by_name('model/input/label_y:0')
        # none sense accuracy = graph.get_tensor_by_name('model/accuracy/Mean:0')

        np_features, y_true = load_features(dir_name + '/EMODB/'+ speaker + '/' + 'val_segments.txt')
        np_features = np.asarray(np_features)
        np_features = np_features[:,:,:,np.newaxis]
        with tf.Session(graph=graph) as sess:
            out = sess.run(fc2, feed_dict={input_x:np_features})
        y_predict = np.argmax(out, axis=1)
        accuracies[i] = accuracy_score(y_true, y_predict)


    with open(dir_name + '/EMODB/'+ 'log.txt', 'a') as f:
        f.write('\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        temp = '   fc:' + str(accuracies) + ',aver=' + str(np.mean(accuracies))+ ', max=' + str(np.max(accuracies)) + ', min=' + str(np.min(accuracies)) +'\n'
        f.write(temp)



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

def load_features(filename):
    np_file_paths = []
    labels = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        for line in lines:
            item = line.split(' ')
            np_file_paths.append(item[0])
            # labels.append(tf.one_hot(int(item[1]), 7))
            labels.append(int(item[1]))

    np_features = []
    for np_file_path in np_file_paths:
        np_features.append(np.load(np_file_path))

    return np_features, labels

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
    files = os.listdir(np_path + '/np')
    for file in files:
        filename = np_path + '/np/' + file
        if os.path.isfile(filename):
            feature_nps.append(np.load(filename))
    feature_nps = tf.convert_to_tensor(np.asarray(feature_nps))
    feature_nps = tf.reshape(feature_nps, [-1, 300, 23, 1])
    return feature_nps


def svm(speaker, X_train, y_train, X_test, y_test):
    '''
    change C, gamma
    '''

    ####svm model optimizing###
    # C from 0.01 to 16384 ( 0.01 * 2 ^14 ) ,
#    num_C = 9
#    Cs = 10 ** np.arange(num_C) * 1e-4
#
#    # gamma for rbf
#    gammas = [1e-4, 1e-3, 1e-2, 1e-1, 1]
    param_grid = {'estimator__C':Cs, 'estimator__gamma':gammas}

    train_val_features = np.concatenate((X_train,X_test), axis=0)
    train_val_labels = np.concatenate((y_train, y_test), axis=0)
    test_fold = np.zeros(train_val_features.shape[0])
    test_fold[:X_train.shape[0]] = -1 # train set indexs are -1
    ps = PredefinedSplit(test_fold=test_fold)

    model = OneVsRestClassifier(SVC(kernel='rbf'))
    clf = GridSearchCV(estimator=model, param_grid=param_grid, cv=ps)
    clf = clf.fit(train_val_features, train_val_labels)
#    train_score = clf.score(X_train, y_train)
#    test_score = clf.score(X_test, y_test)
#    clf_y_train = clf.predict(X_train)
#    clf_y_test = clf.predict(X_test)
#    print('speaker {} in svm classification, train accuracy: {}, test accuracy: {}'.format(speaker, train_score,test_score))
    # means = clf.cv_results_['mean_test_score']
    # stds = clf.cv_results_['std_test_score']
    # for mean, std, params in zip(means, stds, clf.cv_results_['params']):
    #    print('%0.3f (+/-%0.3f) for %r' % (mean, std * 2, params))
    #print('best params are {}'.format(clf.best_params_))
    #print(classification_report(y_test, clf_y_test))
    return clf #, train_score, test_score, clf_y_train, clf_y_test

def get_SVMs(dt):
    # dt means dataset
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))

    mean_test_scores = np.zeros(len(Cs) * len(gammas))
    for speaker in dt.speakers:
        print('svm speaker '+speaker )
        speaker_path = dir_name + '/' + dt.root + '/' + speaker
        X_train = np.load(speaker_path + '/train_np_features_fc1.npy')
        _, y_train = np.asarray(load_inputs(speaker_path+'/train_segments.txt'))
        X_test = np.load(speaker_path + '/val_np_features_fc1.npy')
        _, y_test = np.asarray(load_inputs(speaker_path+'/val_segments.txt'))
        # clf_svm = svm(speaker, X_train, y_train, X_test, y_test) 
        # joblib.dump(clf_svm,speaker_path+'/model_svm.m')
        clf_svm = joblib.load(speaker_path+'/model_svm.m')
        mean_test_scores += clf_svm.cv_results_['mean_test_score']
#        with open(speaker_path+'/log_svm.txt', 'a') as f:
#            f.write('\n')
#            f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
#            f.write('\n')
#            # f.write(str(pd.DataFrame.from_dict(clf_svm.cv_results_)))
#            # f.write(str(pd.DataFrame.from_dict(clf_svm.cv_results_, 'index')))
#            # f.write(str(clf_svm.cv_results_))
#            f.write('Cs: ' + str(clf_svm.cv_results_['param_estimator__C']))
#            f.write('gammas: ' + str(clf_svm.cv_results_['param_estimator__gamma']))

    mean_test_scores /= len(dt.speakers)
    mean_test_scores = mean_test_scores.reshape(len(Cs), len(gammas))

    # draw plot
#    _, ax = plt.subplots(1,1)
#    for ind,i in enumerate(gammas):
#        ax.plot(Cs, mean_test_scores[:, ind], '-o',label='gamma: ' + str(i))
#    ax.set_title('Grid Search Scores', fontsize=20, fontweight='bold')
#    ax.set_xlabel('Cs')
#    ax.set_ylabel('CV Mean Score', fontsize=16)
#    ax.set_xscale('log')
#    ax.legend(loc='best', fontsize=10)
#    ax.grid('on')
#    plt.show()
    index = mean_test_scores.argmax(axis=1)
    with open(dir_name + '/' + dt.root + '/log.txt', 'a') as f:
        f.write('\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        f.write('\n')
        # f.write()
        f.write('svm: best C=' + str(Cs[index]) + ', best gamma=' + str(gammas[index[1]]) + 'best average score=' + str(mean_test_scores[index]))
    
    

def gmm(speaker, X_train, y_train, X_test, y_test, num_classes, num_components=16):
    '''
    change components number
    '''
    clfs = []

    for i in range(num_classes):
        clf = GaussianMixture(n_components=num_components, max_iter=200)
        X_train_each = X_train[np.where(y_train == i)]
        clf.fit(X_train_each)
        clfs.append(clf)
    # gmm.score(X) input X shape (n_samples, n_features). return logprob shape (n_samples,)
    # log_likelihood shape (n_samples, n_classes/n_models)
    log_likelihood_train = np.zeros([X_train.shape[0], num_classes])
    log_likelihood_test = np.zeros([X_test.shape[0], num_classes])
    for i in range(num_classes):
        log_likelihood_train[:, i] = clfs[i].score_samples(X_train)
        log_likelihood_test[:, i] = clfs[i].score_samples(X_test)
    clf_y_train = np.argmax(log_likelihood_train, axis=1)
    clf_y_test = np.argmax(log_likelihood_test, axis=1)
    train_score = accuracy_score(y_train, clf_y_train)
    test_score = accuracy_score(y_test, clf_y_test)
    print('speaker {} in gmm classification, train accuracy: {}, test accuracy: {}'.format(speaker, train_score,
                                                                                           test_score))
    return train_score, test_score, clf_y_train, clf_y_test

def searchGMM(speaker, X_train, y_train, X_test, y_test, num_classes):

    gmm_accuracy = np.zeros(len(num_components))
    for (i,component) in zip(range(len(num_components)),num_components):
        _, gmm_accuracy[i], _, _ = gmm(speaker, X_train, y_train, X_test, y_test, num_classes, num_components=component)
    return gmm_accuracy


def knn(speaker, X_train, y_train, X_test, y_test, num_neighbors=80):
    '''
    change neighbors number
    '''
    clf = KNeighborsClassifier(num_neighbors)
    clf.fit(X_train, y_train)
    train_score = clf.score(X_train, y_train)
    test_score = clf.score(X_test, y_test)
    clf_y_train = clf.predict(X_train)
    clf_y_test = clf.predict(X_test)
    print('speaker {} in knn classification, train accuracy: {}, test accuracy: {}'.format(speaker, train_score,
                                                                                            test_score))
    return train_score, test_score, clf_y_train, clf_y_test

def searchKNN(speaker, X_train, y_train, X_test, y_test):
    knn_accuracy = np.zeros(len(list_neighbors))
    for (i, neighbors) in zip(range(len(list_neighbors)), list_neighbors):
        _, knn_accuracy[i], _, _ = knn(speaker, X_train, y_train, X_test, y_test, num_neighbors=neighbors)
    return knn_accuracy


def test_classifier():
    berlin = Dataset('berlin')

    gmm_accuracy_mean = np.zeros(len(num_components))

    knn_accuracy_mean = np.zeros(len(list_neighbors))

    for speaker in berlin.speakers:
        X_train_paths, y_train = load_inputs(berlin.root + speaker + '/train.txt')
        X_test_paths, y_test = load_inputs(berlin.root + speaker + '/val.txt')
        X_train = []
        for path in X_train_paths:
            file = find_file(path, 'statistic')
            with open(path + '/' + file, 'r') as f:
                reader = csv.reader(f)
                for item in reader:
                    X_train.append(item[0].split(';')[1:])
        X_test = []
        for path in X_test_paths:
            file = find_file(path, 'statistic')
            with open(path + '/' + file, 'r') as f:
                reader = csv.reader(f)
                for item in reader:
                    X_test.append(item[0].split(';')[1:])
        X_train = np.asarray(X_train, dtype='float32')
        X_test = np.asarray(X_test, dtype='float32')
        y_train = np.asarray(y_train, dtype='int')
        y_test = np.asarray(y_test, dtype='int')

#for C in Cs:
        '''
        for i in range(num_C):
            train_accuracys[i], test_accuracys[i], _, _ = svm(speaker, X_train, y_train, X_test, y_test, Cs[i])
        svm_train_accuracys_mean += train_accuracys
        svm_test_accuracys_mean += test_accuracys
        '''
        # gmm_accuracy_mean += searchGMM(speaker,X_train, y_train, X_test, y_test, len(berlin.classes))

        knn_accuracy_mean += searchKNN(speaker,X_train, y_train, X_test, y_test)

        #svm(speaker, X_train, y_train, X_test, y_test)
        #gmm(speaker, X_train, y_train, X_test, y_test, len(berlin.classes))
        knn(speaker, X_train, y_train, X_test, y_test)
    # svm_train_accuracys_mean /= len(berlin.speakers)
    # svm_test_accuracys_mean /= len(berlin.speakers)
    # draw_plt(Cs, 'C in svm', svm_test_accuracys_mean, svm_train_accuracys_mean, 'log')

    # gmm_accuracy_mean /= len(berlin.speakers)
    # draw_plt(num_components, 'components in gmm', gmm_accuracy_mean, np.ones(len(num_components)), 'linear')

    knn_accuracy_mean /= len(berlin.speakers)
    draw_plt(list_neighbors, 'neighbors number in knn', knn_accuracy_mean, np.ones(len(list_neighbors)), 'linear')


def find_file(path, keyword):
    files = os.listdir(path)
    for file in files:
        if os.path.isfile(path + '/' +file):
            if file.find(keyword) != -1:
                return file


def draw_plt(x_axis,x_description, precision, recall, x_type='linear'):
    '''
    x_axis: the x axis , for example C in svm, n_components in GMM
    x_description: the description of x axis
    precision: test accuracy
    recall: train accuracy
    '''
    plt.plot(x_axis, precision, 'o-', color='r', label='precision')
    plt.plot(x_axis, recall, 'o-', color='b', label='recall')
    plt.legend(loc='best')
    plt.xlabel(x_description)
    plt.ylabel('accuracy')
    plt.xscale(x_type)
    plt.show()
    if not os.path.exists('/savefig'):
        os.mkdir('/savefig/')
    plt.savefig('/savefig/' + x_description)

def main():
    berlin = Dataset('berlin')
    get_SVMs(berlin)
    # fc_accuracy(berlin.speakers)
#    get_fc1(berlin.speakers)
main()
# test_classifier()

