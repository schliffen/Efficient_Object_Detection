#
#
#
import glob, os, sys
import numpy as np
import shutil

source_data_dir = "/home/ali/ProjLAB/DATA/NEWGUN/"

dest_train_dir = "/home/ali/ProjLAB/DATA/NEWGUN/train/"
dest_test_dir = "/home/ali/ProjLAB/DATA/NEWGUN/test/"
dest_val_dir = "/home/ali/ProjLAB/DATA/NEWGUN/val/"
if not os.path.exists(dest_train_dir):
    os.mkdir(dest_train_dir)
if not os.path.exists(dest_test_dir):
    os.mkdir(dest_test_dir)
if not os.path.exists(dest_val_dir):
    os.mkdir(dest_val_dir)

if __name__ == "__main__":

    imgList = glob.glob(source_data_dir + 'images/*.jpg')

    tr_dcounter = 0
    ts_dcounter = 0
    vl_dcounter = 0

    for item in imgList:
        rnd_ind = np.random.randint(0, len(imgList))
        splitrnd = np.random.randint(0,20)

        imgName = imgList[rnd_ind].split('/')[-1]
        lblName = imgName.split('/')[-1][:-3] + 'txt'

        if splitrnd in range(2,20):
            try:
                shutil.copy( source_data_dir + 'images/' +imgName , dest_train_dir + imgName)
                shutil.copy(source_data_dir + 'labels/' + lblName, dest_train_dir + lblName)
            except:
                print('this file does not exists: ', imgName)
                continue
            tr_dcounter +=1
        elif splitrnd in [0]:
            try:
                shutil.copy( source_data_dir + 'images/' +imgName , dest_test_dir + imgName)
                shutil.copy(source_data_dir + 'labels/' + lblName, dest_test_dir + lblName)
            except:
                print('this file does not exists: ', imgName)
                continue
            ts_dcounter +=1
        elif splitrnd in [1]:
            try:
                shutil.copy( source_data_dir + 'images/' +imgName , dest_val_dir + imgName)
                shutil.copy(source_data_dir + 'labels/' + lblName, dest_val_dir + lblName)
            except:
                print('this file does not exists: ', imgName)
                continue
            vl_dcounter+=1

        imgList.pop(rnd_ind)

        print('number of collected data train: {}, val: {}, test: {}'.format(tr_dcounter, vl_dcounter, ts_dcounter))

