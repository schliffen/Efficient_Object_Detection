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
    imgs_dir = "/home/ali/ProjLAB/data/NEWGUN/newImgs/"  # dataset adresini images ve annotations  dosyası olacak
     #"./data/"  # kayıt dosyasını oluştur.
    save_new_imgs = "/home/ali/ProjLAB/data/NEWGUN/"
    # save_dir = save_new_imgs
    # imgs_dir = data_dir #+ 'images/'
    lbl_dir = "/home/ali/ProjLAB/data/NEWGUN/newlbls/" #+ 'labels/'

    # prefix = '_WeaponS_'

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
            xc = float(line[1])
            yc = float(line[2])
            width = float(line[3])
            height = float(line[4])
            # calculating rectanble coordinates
            ymin =  max((float(line[2]) - float(line[4].split('\n')[0])/2 ),0)
            xmin = max((float(line[1]) - float(line[3])/2 ),0)
            xmax = min(xmin +  float(line[3]), 1)
            ymax = min(ymin + float(line[4].split('\n')[0]), 1)
            # checking the area
            # print(width * height)
            if width * height > .5 or width * height < .002:
                print( width * height )
                continue
            # check the boxes
            if xmin >= xmax or ymin>=ymax:
                print('wrong bounding box!')
                continue


            # cv2.rectangle( imgarr, (int(xmin * w), int(ymin*h)),(int(xmax*w), int(ymax*h)), (0,255,100), 2 )
            # cv2.namedWindow('img', cv2.WINDOW_NORMAL)
            # cv2.imshow('img', imgarr)
            # ckey = cv2.waitKey(0)

            # print(ckey)
            # if ckey == 32:
            #     # os.remove(imgdir)
            #     # os.remove(lbl_dir + lbl_name)
            #     continue


            # shutil.copy( imgdir, save_new_imgs + 'newImgs/'+prefix + img_name   )
            # shutil.copy(lbl_dir + lbl_name, save_new_imgs + 'newlbls/'+prefix + lbl_name)

            # os.remove( imgdir )
            # os.remove( lbl_dir + lbl_name )

            # creating

            img_name = line[0] + '_' + str(xc) + '-' + str(yc) + '-' + str(width) + '-' + str(height)  + '_' + img_name
            with open(save_new_imgs + '/train_img_list.txt', 'a+') as f:
                    f.write( img_name  +  '\n')

            cv2.imwrite(save_new_imgs + 'ProcessedData/' + img_name, imgarr)
            # shutil.copy( imgdir, save_new_imgs + 'newImgs/' + img_name   )
            # shutil.copy(lbl_dir + lbl_name, save_new_imgs + 'newlbls/'+prefix + lbl_name)

            num_data += 1
            print("num data: ", num_data)


