# -*-coding:utf-8-*-
# hardmstar@126.com
# train and test splits
# dataset
import os
import itertools

EMODB_speakers = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
EMODB_classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
EMODB_classes_num = {des: num for num, des in EMODB_classes.items()}
class Dataset:
    def __init__(self, type):
        self.train_path = []
        self.val_path = []
        self.type = type

        self.train_segment_files_path = []
        self.val_segment_files_path = []

        if self.type == 'berlin':
            self.root = 'EMODB/'
            self.speakers = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
            self.classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
            self.classes_num = {des: num for num, des in self.classes.items()}
            # python 3.x use items() , 2.x use iteritems()
            self.wav_files = self.root + 'wav'
            self.wav_files_list = []
            self.NN_inputs = self.root + 'NN_inputs/'
            for wav in os.listdir(self.wav_files):
                self.wav_files_list.append(wav)
            self.get_dataset()
            #self.generate_train_val_files()


    def get_dataset(self):
        for speaker in self.speakers:
            speakerhome = self.root + speaker + '/'

            self.train_path.append(speakerhome + 'train.txt')
            self.val_path.append(speakerhome + 'val.txt')


    def generate_train_val_files(self):
        # speaker independent training and validation
        # train files describe wav files and its class
        # validation files the same as train files
        self.delete_train_val_files()
        for speaker in self.speakers:
            speakerhome = self.root + speaker + '/'
            if not os.path.exists(speakerhome):
                makedirs = os.makedirs(speakerhome)
            train_file = open(speakerhome + 'train.txt', 'a')
            val_file = open(speakerhome + 'val.txt', 'a')

            for wav in os.listdir(self.wav_files):
                # generate lists of train files and validation files
                if self.return_speaker(wav) == speaker:
                    val_file.write('%s %s\n' % (self.NN_inputs + wav, self.return_class_num(wav)))
                else:
                    train_file.write('%s %s\n' % (self.NN_inputs + wav, self.return_class_num(wav)))

            train_file.close()
            val_file.close()

    def delete_features_np(self):
        for wav in self.wav_files_list:
            feature_np_path = self.NN_inputs + wav + '/np/'
            features_np = os.listdir(feature_np_path)
            for f in features_np:
                if os.path.isfile(feature_np_path+f):
                    os.remove(feature_np_path+f)
                    print(feature_np_path+f+' removed')

    def delete_train_val_files(self):
        for speaker in self.speakers:
            speakerhome = self.root + speaker + '/'
            files = os.listdir(speakerhome)
            for f in files:
                if os.path.isfile(speakerhome+f):
                    os.remove(speakerhome+f)
                    print(speakerhome+f+' removed')


    def return_speaker(self, wav):
        # return speaker of the wav file
        if self.type == 'berlin':
            return wav[:2]


    def return_class_num(self,wav):
        # return emotion of this wav file
        if self.type == 'berlin':
            return self.classes_num[wav[5]]

    def record_train_val_segment_files(self):
        '''
        after genFeatures() and read_frame_csv() in genFeatures.py,
        run record_train_val_segment_files() recording all blocks of features
        '''
        for speaker in self.speakers:
            speakerhome = self.root + speaker + '/'
            train_segment_file = open(speakerhome + 'train_segments.txt', 'a')
            val_segment_file = open(speakerhome + 'val_segments.txt', 'a')

            for wav in self.wav_files_list:
                wav_segment_files_root = self.NN_inputs + wav + '/np/'
                wav_segment_files = os.listdir(wav_segment_files_root)
                for f in wav_segment_files:
                     if os.path.isfile(wav_segment_files_root + f):
                         if self.return_speaker(wav) == speaker:
                             val_segment_file.write('%s %s\n' % (wav_segment_files_root+f, self.return_class_num(wav)))
                         else:
                             train_segment_file.write('%s %s\n' % (wav_segment_files_root+f, self.return_class_num(wav)))
            train_segment_file.close()
            val_segment_file.close()
            self.train_segment_files_path.append(speakerhome+'train_segments.txt')
            self.val_segment_files_path.append(speakerhome+'val_segments.txt')


def return_class(type, wav):
    if type == 'berlin':
        return EMODB_classes_num[wav[5]], wav[5]
dataset = Dataset('berlin')
