#
#
#
#
import os
import glob
import cv2
import matplotlib

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import shutil

if __name__ == '__main__':
    imgs_dir = "/home/ali/ProjLAB/data/NEWGUN/ProcessedData/"  # dataset adresini images ve annotations  dosyası olacak
    # "./data/"  # kayıt dosyasını oluştur.
    imglsit = "/home/ali/ProjLAB/data/NEWGUN/train_img_list.txt"
    # save_dir = save_new_imgs

    with open( imglsit, 'r') as f:
        imgnames = f.readlines()


    for line in imgnames:
        imgarr = cv2.imread(imgs_dir + line.split('\n')[0])
        h, w, c = imgarr.shape

        # extracting labels
        lbl_cls  = line.split('/')[-1].split('_')[1].split('-')
        lblbes = [float(lbl_cls[i].split('\n')[0]) for i in range(4) ]
        # reading the annotation

        xc = lblbes[1]
        yc = lblbes[0]
        width = lblbes[2]
        height = lblbes[3]
        # calculating rectanble coordinates
        ymin = max((xc - width / 2), 0)
        xmin = max((yc - height/ 2), 0)
        xmax = min(xmin + width, 1)
        ymax = min(ymin + height, 1)
        # checking the area
        # print(width * height)
        # if width * height > .5 or width * height < .01:
        #     print(width * height)
        #     continue
        # check the boxes
        # if xmin >= xmax or ymin >= ymax:
        #     print('wrong bounding box!')
        #     continue

        cv2.rectangle( imgarr, (int(xmin * w), int(ymin*h)),(int(xmax*w), int(ymax*h)), (0,255,100), 2 )
        cv2.namedWindow('img', cv2.WINDOW_NORMAL)
        cv2.imshow('img', imgarr)
        ckey = cv2.waitKey(0)

        # print(ckey)
        # if ckey == 32:
        #     # os.remove(imgdir)
        #     # os.remove(lbl_dir + lbl_name)
        #     continue

        # shutil.copy( imgdir, save_new_imgs + 'newImgs/'+prefix + img_name   )
        # shutil.copy(lbl_dir + lbl_name, save_new_imgs + 'newlbls/'+prefix + lbl_name)




