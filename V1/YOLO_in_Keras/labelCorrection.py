#
#
#
#
#
#
import glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

#
data_dir = '/home/ali/CLionProjects/object_detection/SqueezeDet/anchor/'
filelist = glob.glob(data_dir + 'labels/' + '*.txt')
imgdir = data_dir + 'images/'


# illustrating images and labels
for item in filelist:
    dname = item.split('/')[-1]
    f = open( item, 'r')
    # img = cv2.imread(imgdir + dname + 'jpg')
    # plotting bounding box and annotations on image
    # w, h, _ = img.shape
    fw = open(data_dir +'M_labels/' + dname, 'a')
    for line in f:
        infos = line.split(" ")
        # length = len(infos)
        # center_y = int(float(infos[1]) * w)
        # center_x = int(float(infos[0]) * h)
        # width = float(infos[2]) * w/2 # int(infos[2]) - int(infos[0])
        # height = float(infos[3][:-1]) * h  # int(i# nfos[3][:-1]) - int(infos[1])
        # xmin = int(center_x - height/2)
        # ymin = int(center_y - width/2)
        # xmax = int(center_x + height/2)
        # ymax = int(center_y + width/2)
        #
        infos[3] = str(float(infos[3])/2)
        # with open(data_dir +'M_labels/' + dname, 'a') as fw:
        fw.write(' '.join(infos))

    f.close()
    fw.close()


