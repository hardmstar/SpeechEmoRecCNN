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
    confusion_matrix_fc = np.zeros([7,7])
#    for (i,speaker) in zip(range(len(speakers)), speakers):
#        print('speaker '+ speaker +' now')
#        graph = load_graph(dir_name + '/EMODB/'+ speaker + '/' + graph_filename)
#        input_x = graph.get_tensor_by_name('model/input/input_x:0')
#        fc2 = graph.get_tensor_by_name('model/output/fully_connection_layers/fc2:0')
#        # none sense input_y = graph.get_tensor_by_name('model/input/label_y:0')
#        # none sense accuracy = graph.get_tensor_by_name('model/accuracy/Mean:0')
#
#        np_features, y_true = load_features(dir_name + '/EMODB/'+ speaker + '/' + 'val_segments.txt')
#        np_features = np.asarray(np_features)
#        np_features = np_features[:,:,:,np.newaxis]
#        with tf.Session(graph=graph) as sess:
#            out = sess.run(fc2, feed_dict={input_x:np_features})
#        y_predict = np.argmax(out, axis=1)
#        accuracies[i] = accuracy_score(y_true, y_predict)
#        labels = np.arange(7)
#        cm_fc = confusion_matrix(y_true, y_predict, labels=labels)
#        # normalize confusion matrix
#        cm_fc = cm_fc.astype('float') / (cm_fc.sum(axis=1)[:, np.newaxis]+ 1e-7)
#        confusion_matrix_fc += cm_fc
#    confusion_matrix_fc /= len(speakers)
#    np.save(dir_name + '/EMODB/'+'confusion_matrix_fc.npy', confusion_matrix_fc)
    confusion_matrix_fc = np.load(dir_name + '/EMODB/'+'confusion_matrix_fc.npy')
    confusion_matrix_fc.astype('float')
    
    return confusion_matrix_fc


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

def searchSVMs(dt):
    # dt means dataset
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))

    mean_test_scores = np.zeros(len(Cs) * len(gammas))
    speaker_scores_svm = np.zeros(len(dt.speakers))
    for (i,speaker) in enumerate(dt.speakers):
        print('svm speaker '+speaker )
        speaker_path = dir_name + '/' + dt.root + '/' + speaker
        X_train = np.load(speaker_path + '/train_np_features_fc1.npy')
        _, y_train = np.asarray(load_inputs(speaker_path+'/train_segments.txt'))
        X_test = np.load(speaker_path + '/val_np_features_fc1.npy')
        _, y_test = np.asarray(load_inputs(speaker_path+'/val_segments.txt'))
#        clf_svm = svm(speaker, X_train, y_train, X_test, y_test) 
#        joblib.dump(clf_svm,speaker_path+'/model_svm_cv.m')
        clf_svm = joblib.load(speaker_path+'/model_svm_cv.m')
        mean_test_scores += clf_svm.cv_results_['mean_test_score']
        print(clf_svm.cv_results_['mean_test_score'])
        speaker_scores_svm[i] = clf_svm.score(X_test, y_test)
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
    index = np.where(mean_test_scores == np.max(mean_test_scores))
    with open(dir_name + '/' + dt.root + '/log.txt', 'a') as f:
        f.write('\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        f.write('\n')
        # f.write()
        f.write('svm cv: best C=' + str(Cs[index[0][0]]) + ', best gamma=' + str(gammas[index[1][0]]) + 'best average score=' + str(mean_test_scores[index])+ '\n') 
        temp = 'svm cv test socres: '+str(speaker_scores_svm)+', aver='+str(np.mean(speaker_scores_svm))+', max='+str(np.max(speaker_scores_svm))+', min='+str(np.min(speaker_scores_svm))+'\n'
        f.write(temp)
    
    

def gmm(speaker, X_train, y_train, X_test, y_test, num_classes, num_components=16):
    '''
    change components number
    '''
    clfs = []

    for i in range(num_classes):
        clf = GaussianMixture(n_components=num_components, max_iter=200)
        x = np.where(y_train==i)
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
    # print('speaker {} in gmm classification, train accuracy: {}, test accuracy: {}'.format(speaker, train_score, test_score))
#    return train_score, test_score, clf_y_train, clf_y_test
    return clfs

def searchGMM(dt):
    # dt means dataset
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))
    gmm_accuracies = np.zeros(len(num_components))

    for speaker in dt.speakers:
        print('gmm speaker '+speaker )
        speaker_path = dir_name + '/' + dt.root + '/' + speaker
        X_train = np.load(speaker_path + '/train_np_features_fc1.npy')
        _, y_train = np.asarray(load_inputs(speaker_path+'/train_segments.txt'))
        X_test = np.load(speaker_path + '/val_np_features_fc1.npy')
        _, y_test = np.asarray(load_inputs(speaker_path+'/val_segments.txt'))
        y_train = np.asarray(y_train,int)
        y_test = np.asarray(y_test,int)

        for (i,component) in zip(range(len(num_components)),num_components):
            _, gmm_accuracy, _, _ = gmm(speaker, X_train, y_train, X_test, y_test, len(dt.classes), num_components=component)
            gmm_accuracies[i] += gmm_accuracy
    gmm_accuracies /= len(dt.speakers)

    # draw plot
#    _, ax = plt.subplots(1,1)
#    max_index = np.argmax(gmm_accuracies)
#    ax.plot(num_components[max_index], gmm_accuracies[max_index], 'ks')
#    show_max = '(' + str(num_components[max_index]) + ', ' + str(round(gmm_accuracies[max_index],3)) + ')'
#    plt.annotate(show_max, xy=(num_components[max_index], gmm_accuracies[max_index]))
 
#    ax.plot(num_components, gmm_accuracies, '-o')
#    ax.set_title('GMM test Scores', fontsize=20, fontweight='bold')
#    ax.set_xlabel('Number of Gaussian Components')
#    ax.set_ylabel('Mean Test Score', fontsize=16)
##    ax.set_xscale('log')
#    ax.legend(loc='best', fontsize=10)
#    ax.grid('on')
#    plt.show()
 

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
#    print('speaker {} in knn classification, train accuracy: {}, test accuracy: {}'.format(speaker, train_score,test_score))
#    return train_score, test_score, clf_y_train, clf_y_test
    return clf

def searchKNN(dt):
    # dt means dataset
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))
    knn_accuracies = np.zeros(len(list_neighbors))

    for speaker in dt.speakers:
        print('knn speaker '+speaker )
        speaker_path = dir_name + '/' + dt.root + '/' + speaker
        X_train = np.load(speaker_path + '/train_np_features_fc1.npy')
        _, y_train = np.asarray(load_inputs(speaker_path+'/train_segments.txt'))
        X_test = np.load(speaker_path + '/val_np_features_fc1.npy')
        _, y_test = np.asarray(load_inputs(speaker_path+'/val_segments.txt'))
        y_train = np.asarray(y_train,int)
        y_test = np.asarray(y_test,int)

        for (i, neighbors) in zip(range(len(list_neighbors)), list_neighbors):
            _, knn_accuracy, _, _ = knn(speaker, X_train, y_train, X_test, y_test, num_neighbors=neighbors)
            knn_accuracies[i] += knn_accuracy

    knn_accuracies /= len(dt.speakers)

    # draw plot
    _, ax = plt.subplots(1,1)
    ax.plot(list_neighbors, knn_accuracies, '-o')

    max_index = np.argmax(knn_accuracies)
    ax.plot(list_neighbors[max_index], knn_accuracies[max_index], 'ks')
    show_max = '(' + str(list_neighbors[max_index]) + ', ' + str(round(knn_accuracies[max_index],3)) + ')'
    plt.annotate(show_max, xy=(list_neighbors[max_index], knn_accuracies[max_index]))
    ax.set_title('KNN test Scores', fontsize=20, fontweight='bold')
    ax.set_xlabel('Number of Neighbors')
    ax.set_ylabel('Mean Test Score', fontsize=16)
#    ax.set_xscale('log')
    ax.legend(loc='best', fontsize=10)
    ax.grid('on')
    plt.show()
 
def best_classifier(dt):
    # dt means dataset
    dir_name = os.path.dirname(os.path.abspath('get_fc1.py'))
    speaker_scores_svm = np.zeros(len(dt.speakers))
    speaker_scores_gmm = np.zeros(len(dt.speakers))
    speaker_scores_knn = np.zeros(len(dt.speakers))

    confusion_matrix_svm = np.zeros([len(dt.classes), len(dt.classes)])
    confusion_matrix_gmm = np.zeros([len(dt.classes), len(dt.classes)])
    confusion_matrix_knn = np.zeros([len(dt.classes), len(dt.classes)])

    for (index,speaker) in enumerate(dt.speakers):
        speaker_path = dir_name + '/' + dt.root + '/' + speaker
        X_train = np.load(speaker_path + '/train_np_features_fc1.npy')
        _, y_train = np.asarray(load_inputs(speaker_path+'/train_segments.txt'))
        X_test = np.load(speaker_path + '/val_np_features_fc1.npy')
        _, y_test = np.asarray(load_inputs(speaker_path+'/val_segments.txt'))
        y_train = np.asarray(y_train,int)
        y_test = np.asarray(y_test,int)

        
        # generate models
#        clf_svm = OneVsRestClassifier(SVC(kernel='rbf',C=10,gamma=0.001)) 
#        clf_svm = clf_svm.fit(X_train, y_train)
#        clf_gmms = gmm(speaker, X_train, y_train, X_test, y_test, len(dt.classes), num_components=17)
#        clf_knn = knn(speaker, X_train, y_train, X_test, y_test, num_neighbors=30)

        # save models
#        joblib.dump(clf_svm,speaker_path+'/model_svm_best.m')
#        for (ind,clf_gmm) in enumerate(clf_gmms):
#            if not os.path.exists(speaker_path+'/gmm'):
#                os.mkdir(speaker_path+'/gmm/')
#            joblib.dump(clf_gmm, speaker_path+'/gmm/model_gmm_best_'+ str(ind) +'.m')
#        joblib.dump(clf_knn, speaker_path+'/model_knn_best.m')
#
        # load models
        clf_svm = joblib.load(speaker_path+'/model_svm.m')
        clf_gmms = []
        for (ind,emotion) in enumerate(dt.classes):
            clf_gmms.append(joblib.load(speaker_path+'/gmm/model_gmm_best_'+ str(ind) +'.m'))
        clf_knn = joblib.load(speaker_path+'/model_knn_best.m')

        # calculate test scores
        svm_predict = clf_svm.predict(X_test)
        svm_predict = np.asarray(svm_predict, dtype=np.int8)
        speaker_scores_svm[index] = accuracy_score(y_test, svm_predict) 

        log_likelihood_test = np.zeros([X_test.shape[0], len(dt.classes)])
        for i in range(len(dt.classes)):
            log_likelihood_test[:, i] = clf_gmms[i].score_samples(X_test)
        clf_y_test = np.argmax(log_likelihood_test, axis=1)
        speaker_scores_gmm[index] = accuracy_score(y_test, clf_y_test)

        speaker_scores_knn[index] = clf_knn.score(X_test, y_test)
        knn_predict = clf_knn.predict(X_test)

        # calculate confusion matrix
        labels = np.arange(len(dt.classes))
        cm_svm = confusion_matrix(y_test, svm_predict, labels=labels)
        cm_gmm = confusion_matrix(y_test, clf_y_test , labels=labels)
        cm_knn = confusion_matrix(y_test, knn_predict, labels=labels)
        # normalize confusion matrix
        cm_svm = cm_svm.astype('float') / (cm_svm.sum(axis=1)[:, np.newaxis]+ 1e-7)
        cm_gmm = cm_gmm.astype('float') / (cm_gmm.sum(axis=1)[:, np.newaxis]+ 1e-7)
        cm_knn = cm_knn.astype('float') / (cm_knn.sum(axis=1)[:, np.newaxis]+ 1e-7)
        #
        confusion_matrix_svm += cm_svm
        confusion_matrix_gmm += cm_gmm
        confusion_matrix_knn += cm_knn 

    confusion_matrix_svm /= len(dt.speakers)
    confusion_matrix_gmm /= len(dt.speakers)
    confusion_matrix_knn /= len(dt.speakers)
    with open(dir_name + '/EMODB/'+ 'log.txt', 'a') as f:
        f.write('\n')
        f.write(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime()))
        f.write('\n')
        temp = 'svm test socres: '+str(speaker_scores_svm)+', aver='+str(np.mean(speaker_scores_svm))+', max='+str(np.max(speaker_scores_svm))+', min='+str(np.min(speaker_scores_svm))+'\n'
        f.write(temp)
        temp = 'gmm test socres: '+str(speaker_scores_gmm)+', aver='+str(np.mean(speaker_scores_gmm))+', max='+str(np.max(speaker_scores_gmm))+', min='+str(np.min(speaker_scores_gmm))+'\n'
        f.write(temp)
        temp = 'knn test socres: '+str(speaker_scores_knn)+', aver='+str(np.mean(speaker_scores_knn))+', max='+str(np.max(speaker_scores_knn))+', min='+str(np.min(speaker_scores_knn))+'\n'
        f.write(temp)
        f.write('\n')
        f.write('svm \n')
        f.write(str(confusion_matrix_svm))
        f.write('gmm \n')
        f.write(str(confusion_matrix_gmm))
        f.write('knn \n')
        f.write(str(confusion_matrix_knn))
        f.write('/n')
        return confusion_matrix_svm, confusion_matrix_gmm, confusion_matrix_knn


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

def plot_confusion_matrix(ax, cm, classes, title, cmap=plt.cm.Blues):
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
#    ax.set_xticks(tick_marks, classes, rotation=45)
    ax.tick_params(axis='x', rotation=45)
#    ax.set_xticks(tick_marks, classes)
    ax.set_xticklabels(classes)
#    ax.set_yticks(tick_marks, classes)
    ax.set_yticklabels(classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j,i, format(cm[i,j], fmt), horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')
    

def plot_confusion_matrix_single(cm, classes, title, cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
#    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i,j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j,i, format(cm[i,j], fmt), horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')
    plt.tight_layout()
    


def main():
    berlin = Dataset('berlin')
    # searchSVMs(berlin)
    # searchGMM(berlin)
    # searchKNN(berlin)
    # fc_accuracy(berlin.speakers)
#    get_fc1(berlin.speakers)
    cm_fc = fc_accuracy(berlin.speakers)
    cm_svm, cm_gmm, cm_knn = best_classifier(berlin)
    print(cm_fc)
    print(cm_svm)
    print(cm_gmm)
    print(cm_knn)
    EMODB_classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
    classes = ['anger', 'boredom', 'disgust', 'fear', 'happiness', 'sadness', 'neutral']

#    fig, (ax_fc, ax_svm, ax_gmm, ax_knn) = plt.subplots(1,4)

##    ax_fc = plt.subplot(1,4,1)
#    ax_fc = plt.subplot(2,2,1)
#    plot_confusion_matrix(ax_fc,   cm_fc, classes, title='a) FC Layer')
##    ax_svm = plt.subplot(1,4,2)
#    ax_svm = plt.subplot(2,2,2)
#    plot_confusion_matrix(ax_svm, cm_svm, classes, title='b) SVM')
##    ax_gmm = plt.subplot(1,4,3)
#    ax_gmm = plt.subplot(2,2,3)
#    plot_confusion_matrix(ax_gmm, cm_gmm, classes, title='c) GMM')
##    ax_knn = plt.subplot(1,4,4)
#    ax_knn = plt.subplot(2,2,4)
#    plot_confusion_matrix(ax_knn, cm_knn, classes, title='d) KNN')

    plot_confusion_matrix_single(cm_fc, classes, title='a) FC Layer')
#    plot_confusion_matrix_single(cm_svm, classes, title='b) SVM')
#    plot_confusion_matrix_single(cm_gmm, classes, title='c) GMM')
#    plot_confusion_matrix_single(cm_knn, classes, title='d) KNN')
    # plt.tight_layout()
    plt.show()
main()
# test_classifier()

