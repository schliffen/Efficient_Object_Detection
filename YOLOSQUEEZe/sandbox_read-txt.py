#
#
#
import os
import numpy as np
from PIL import Image
import shutil

import matplotlib.pyplot as plt
import cv2

ann_dir = '/home/ali/data/GBP_data/GBBall_data_01/labels/'
img_dir = '/home/ali/data/GBP_data/GBBall_data_01/images/'

destination = ''
#
# ---------------------< visualization >-------------------------
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = .5
fontColor              = (10,10,255)
lineType               = 2
fontColor              = (255, 0, 0)
#
def visualize(frame, point):
    cv2.rectangle(frame, (point[0], point[1]), (point[2], point[3]), fontColor, 2)
    return frame



def parse_annotation_txt(ann_dir, img_dir, labels=[]):
    #
    all_imgs = []
    seen_labels = {}
    #
    num_lbl = 0
    #
    for ann in sorted(os.listdir(ann_dir)):
        # reading txt lines
        with open(ann_dir + ann , 'r') as f:
            imglbl = f.readlines()[0].split(',')

        if int(imglbl[0]) == int(imglbl[1]) == -1:
            continue

        # get image name
        img_name = ann.split('.')[0] + '.bmp'
        image = Image.open( img_dir + img_name )
        #
        width, height = image.size

        # compute bounding box
        xmax = float(imglbl[0]) + float(imglbl[2])
        ymax = float(imglbl[1]) + float(imglbl[2])
        xmin = float(imglbl[0]) - float(imglbl[2])
        ymin = float(imglbl[1]) - float(imglbl[2])
        # data structure
        # {'height': 1920, 'filename': '/home/ali/data/images/001f737c-1d25-4520-a942-92ef7cf4e988.jpg',
        # 'width': 1080,
        # 'object': [{'ymax': 1415, 'xmin': 719, 'ymin': 1386, 'name': 'gball', 'xmax': 746}]}
        #
        single_img = {'height': height, 'width':width, 'filename': img_dir + img_name,
                      'object': [{'ymax': ymax, 'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'name': 'grball'}]
                           }
        num_lbl += 1
        print('num valid data ... ', num_lbl )
        #
        img = np.array(image)
        img = cv2.rectangle(np.array(img), (int(xmin), int(ymin)), (int(xmax), int(ymax)), fontColor, 2)
        #
        # put text on the detector
        cv2.putText(img, '.',
                    (int(imglbl[0]), int(imglbl[1])),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        plt.imshow(img)

        # checking the data
        cv2.imshow('image')
        key = cv2.waitKey(0)

        if key == ord('s'):
            #
            filename =  img_dir + img_name
            label_name = ann_dir + ann
            # copy image and data as valid data
            shutil.copy( img_dir + img_name )
            shutil.copy( ann_dir + ann )




    return all_imgs, seen_labels


if __name__ == '__main__':
    #
    parse_annotation_txt(ann_dir, img_dir, labels=[])

    print('finished! ... ')