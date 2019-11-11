#
# converting to coreml
#
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
from preprocessing import parse_annotation
from utils import draw_boxes
from frontend import YOLO
import json

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import coremltools

model_path = '/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/pretrained_models/raccoon_squeezenet_backend.h5'
# ------------------------------------------------------
argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')
argparser.add_argument(
    '-c',
    '--conf',
    default='config.json',
    help='path to configuration file')
argparser.add_argument(
    '-w',
    '--weights',
    default=model_path,
    help='path to pretrained weights')
argparser.add_argument(
    '-i',
    '--input',
    default='',
    help='path to an image or an video (mp4 format)')
# -----------------------------------------------------
ap = argparser.parse_args()



if __name__ == '__main__':
    #
    config_path  = ap.conf
    weights_path = ap.weights
    #
    with open(config_path) as config_buffer:
        config = json.load(config_buffer)
    #
    ###############################
    #   Make the model
    ###############################

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'],
                labels              = config['model']['labels'],
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    ###############################
    #   Load trained weights
    ###############################

    yolo.load_weights(weights_path)
    #
    # model is ready --> working on transforming the model
    #


    # dividing model to the submodels
    #


    #
    coreml_model = coremltools.converters.keras.convert(
        model_path,
        input_names='image',
        image_input_names='image',
        output_names='output',
        image_scale=1./255.)

    coreml_model.author = 'A.N'
    coreml_model.license = 'BSD'
    coreml_model.short_description = 'Keras port of YOLOTiny VOC2007 by Joseph Redmon and Ali Farhadi'
    coreml_model.input_description['image'] = '416x416 RGB Image'
    coreml_model.output_description['output'] = '13x13 Grid made up of: [confidence, cx, cy, w, h, 20 x classes] * 5 bounding boxes'

    coreml_model.save('output/tinyyolo_squeezedet.mlmodel')

