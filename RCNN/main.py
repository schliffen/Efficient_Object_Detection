# This is a sample Python script.

import numpy as np
import torch
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, random
import cv2
import glob

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

# --------------------------
img_path  = "/home/ali/ProjLAB/DATA/cars_train"

# --------------------------


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    assert torch.__version__.startswith("1.6")
    #
    # check wether it is working
    img_list = glob.glob(img_path + '/*.jpg')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
