#! /usr/bin/env python

import argparse
import os, glob
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json
import matplotlib.pyplot as plt

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')
argparser.add_argument(
    '-c',
    '--conf',
    default='/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/YOLOSQUEEZe/experimental/config_sq.json',
    help='path to configuration file')
argparser.add_argument(
    '-w',
    '--weights',
    default= '/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/YOLOSQUEEZe/results/weights/gball_squeezenet_backend_01.h5',
    help='path to pretrained weights')
argparser.add_argument(
    '-i',
    '--input',
    default='/home/ali/data/Golf_Ball_Data/p6/',
    help='path to an image or an video (mp4 format)')
argparser.add_argument(
    '-s',
    '--save',
    default='/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/pretrained_models/k_model/compiled_full_keras_model_03.h5',
    help=' save working and compiled keras model')


def _main_(args):
    config_path  = args.conf
    weights_path = args.weights
    image_path   = args.input

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    ###############################
    #   Make the model 
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'],
                training = False)

    ###############################
    #   Load trained weights
    ###############################    

    yolo.load_weights(weights_path)

    # saving yolo model as keras
    yolo.model.save(args.save)



    ###############################
    #   Predict bounding boxes 
    ###############################

    if image_path[-4:] == '.mp4':
        video_out = image_path[:-4] + '_detected' + image_path[-4:]
        video_reader = cv2.VideoCapture(image_path)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               50.0, 
                               (frame_w, frame_h))

        for i in tqdm(range(nb_frames)):
            _, image = video_reader.read()
            
            boxes = yolo.predict(image)
            image = draw_boxes(image, boxes, config['model']['labels'])

            video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  

    else:

        image_list= glob.glob(image_path + '*.jpg')

        for item in image_list:
            # rnd = np.random.randint(0, len(image_list))
            image = cv2.imread( item )

            boxes = yolo.predict(image)

            if len(boxes) >= 1:
                image = draw_boxes(image, boxes, config['model']['labels'])

                plt.imshow(image)
                plt.show()
            print(len(boxes), 'boxes are found')

        # cv2.imwrite(image_path[:-4] + '_detected' + image_path[-4:], image)

if __name__ == '__main__':
    args = argparser.parse_args()
    _main_(args)
