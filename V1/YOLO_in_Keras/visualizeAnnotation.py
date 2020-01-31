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

#visualization
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,500)
fontScale              = .5
fontColor              = (10,10,255)
lineType               = 2
fontColor = (255, 0, 0)
def visualize(frame, point):

    cv2.rectangle(frame, (point[0], point[1]), (point[2], point[3]), fontColor, 2)
    return frame

# illustrating images and labels
for item in filelist:
    dname = item[:-3].split('/')[-1]
    f = open( item, 'r')
    img = cv2.imread(imgdir + dname + 'jpg')

    # plotting bounding box and annotations on image
    w, h, _ = img.shape
    for line in f:
        infos = line.split(" ")[1:]
        length = len(infos)
        center_y = int(float(infos[1]) * w)
        center_x = int(float(infos[0]) * h)
        width = float(infos[2]) * w/2 # int(infos[2]) - int(infos[0])
        height = float(infos[3][:-1]) * h  # int(i# nfos[3][:-1]) - int(infos[1])
        xmin = int(center_x - height/2)
        ymin = int(center_y - width/2)
        xmax = int(center_x + height/2)
        ymax = int(center_y + width/2)
        #
        img = cv2.rectangle(img, (xmin, ymin), (xmax, ymax), fontColor, 2)
            #
            # put text on the detector
        cv2.putText(img, '.',
                    (center_x, center_y),
                    font,
                    fontScale,
                    fontColor,
                    lineType)
        # cv2.namedWindow('image', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('image', img)
        # cv2.waitKey(0)
        plt.imshow(img)




    f.close()
