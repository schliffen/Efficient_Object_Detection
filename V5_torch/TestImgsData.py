#
#
#
import glob, os, sys
import cv2
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt


img_dir = '/home/ali/ProjLAB/data/BCCD.v1-resize-416x416.yolov5pytorch/test/'

if __name__ == '__main__':

    imgList = glob.glob( img_dir + 'images/*.jpg')


    for item in imgList:
        img = cv2.imread( item )
        w,h,c = img.shape
        labelName = item.split('/')[-1][:-3] + 'txt'

        with open( img_dir + 'labels/' + labelName, 'r') as f:
            lines = f.readlines()

        objects = []
        for line in lines:
            rect = line.split(' ')
            cls = int( rect[0] )
            xc = float( rect[1] ) * w
            yc = float(rect[2]) * h
            width = float(rect[3]) * w
            height = float(rect[4].split('\n')[0]) * h
            xmin = xc - width/2
            ymin = yc - height/2
            xmax = xc + width/2
            ymax = yc + height / 2
            #
            objects.append([xmin, ymin, xmax, ymax])


        # visualizing the bounding boxes

        for i in range(len(objects)):
            cv2.rectangle(img,(int(objects[i][0]), int(objects[i][1])), (int(objects[i][2]), int(objects[i][3])) , (0,1,255),1)
            cv2.putText(img, '.', (int((objects[i][0] + objects[i][2])/2), int((objects[i][1] + objects[i][3])/2)), cv2.FONT_HERSHEY_COMPLEX, 1, (1,1,255), 1 )
            cv2.imshow('img', img)
            cv2.waitKey(0)


