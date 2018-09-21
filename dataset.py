# -*-coding:utf-8-*-
# hardmstar@126.com
# train and test splits
# dataset
import os
import itertools


class Dataset:
    def __init__(self, type):
        self.train_path = []
        self.val_path = []
        if type == 'berlin':
            self.root = 'EMODB/'
            self.speakers = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']
            self.classes = {0: 'W', 1: 'L', 2: 'E', 3: 'A', 4: 'F', 5: 'T', 6: 'N'}
            self.classes_num = {des: num for num, des in self.classes.items()}
            # python 3.x use items() , 2.x use iteritems()
            self.wav_files = self.root + 'wav'
            self.NN_inputs = self.root + 'NN_inputs'
            #self.get_dataset('berlin')

    def get_dataset(self, type):
        for speaker in self.speakers:
            speakerhome = self.root + speaker + '/'
            if not os.path.exists(speakerhome):
                os.makedirs(speakerhome)

            self.train_path.append(speakerhome + 'train.txt')
            self.train_path.append(speakerhome + 'val.txt')

            train_file = open(speakerhome + 'train.txt', 'a')
            val_file = open(speakerhome + 'val.txt','a')

            for wav in os.listdir(self.wav_files):
                if self.return_speaker(type, wav) == speaker:
                    val_file.write('%s %s\n' %(speakerhome + wav, self.return_class(type, wav)))
                else:
                    train_file.write('%s %s\n' %( speakerhome + wav, self.return_class(type, wav)))

            train_file.close()
            val_file.close()

    def return_speaker(self, type, wav):
        # return speaker of the wav file
        if type == 'berlin':
            return wav[:2]

    def return_class(self, type, wav):
        # return emotion of this wav file
        if type == 'berlin':
            return self.classes_num[wav[5]]


#dataset = Dataset('berlin')
