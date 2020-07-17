#
"""
Preparing data for yolo v5 training
Note: yolo v5 training requires x_center, y_center, width, height in range [0, 1]

"""
import os
import glob
import cv2
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import shutil

if __name__ == '__main__':
    data_dir = "/home/ali/ProjLAB/data/gun/"  # dataset adresini images ve annotations  dosyası olacak
    save_dir = "./data/"  # kayıt dosyasını oluştur.
    save_new_imgs = "/home/ali/ProjLAB/data/NEWgun/"

    imgs_dir = data_dir + 'images/'
    lbl_dir = data_dir + 'annotations/'

    imgs_list = glob.glob(imgs_dir + '*.jpg')

    num_data = 0

    for imgdir in imgs_list:
        imgarr = cv2.imread(imgdir)
        h,w,c = imgarr.shape
        img_name = imgdir.split('/')[-1]
        lbl_name = img_name.split('.')[0] + '.txt'
        # reading the annotation
        # check if annotation is also exists:
        try:
            with open(lbl_dir + lbl_name, 'r') as f:
                lbl_lines = f.readlines()
        except:
            continue
        for line in lbl_lines:
            line = line.split(' ')
            # class_ = line[0]
            # bbox = [line[i].split('\n')[0] for i in range(1,5)]

            # calculating xc yc w h
            xc = float(line[2])
            yc = float(line[1])
            width = float(line[3])
            height = float(line[4])
            # calculating rectanble coordinates
            ymin =  max((float(line[2]) - float(line[4].split('\n')[0])/2 ),0)
            xmin = max((float(line[1]) - float(line[3])/2 ),0)
            xmax = min(xmin +  float(line[3]), 1)
            ymax = min(ymin + float(line[4].split('\n')[0]), 1)
            # checking the rectangle
            """
            cv2.rectangle( imgarr, (int(xmin), int(ymin)),(int(xmax), int(ymax)), (0,255,100), 2 )
            cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            cv2.imshow('img', imgarr)
            ckey = cv2.waitKey(0)
            # print(ckey)
            if ckey == 32:
                os.remove(imgdir)
                os.remove(lbl_dir + lbl_name)
                continue

            shutil.copy( imgdir, save_new_imgs + 'images/' + img_name   )
            shutil.copy(lbl_dir + lbl_name, save_new_imgs + 'annotations/' + lbl_name)
            os.remove( imgdir )
            os.remove( lbl_dir + lbl_name )
            """
            with open(save_dir + 'train_img_list.txt', 'a+') as f:
                f.write(line[0] + '_' + str(xc) + '-' + str(yc) + '-' + str(width) + '-' + str(height) + '_' + img_name + '\n')

            num_data += 1
            print("num data: ", num_data)
