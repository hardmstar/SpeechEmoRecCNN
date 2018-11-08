# -*-coding:utf-8-*-
import os
import datetime
import time
from dataset import *
def mdfilename(origin):
    berlin = Dataset('berlin')
    for speaker in berlin.speakers:
        nowStamp = int(time.time())
        timeTuple = time.localtime(nowStamp)
        otherTime = time.strftime("%Y%m%d_%H%M",timeTuple)
        origin_file_path = berlin.root+speaker+'/'+origin
        if os.path.isfile(origin_file_path):
            datefile = origin_file_path[:-3] + '_'+ otherTime + origin_file_path[-3:]
            os.rename(origin_file_path, datefile)

mdfilename('residual_model.pb')
