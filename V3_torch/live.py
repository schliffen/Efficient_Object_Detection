"""
Process video from file or from camera using trained model
"""
from __future__ import division
import torch 
import torch.nn as nn
from util import de_letter_box, write_results, load_classes, process_output
from darknet import Darknet
from preprocess import prep_image, inp_to_image, letterbox_image

import time
import numpy as np
import cv2
import pandas as pd
import random 
import pickle as pkl
import argparse
import copy

# Choose backend device for tensor operations - GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def arg_parse():
    """Parse arguements to the detect module"""
    parser = argparse.ArgumentParser(description='YOLO v3 Video Detection Module')
   
    parser.add_argument("--video", dest='video', 
                        help="Video to run detection upon", type=str)
    parser.add_argument("--source", dest='source', 
                        help="Video source used by OpenCV VideoCapture", 
                        type=int, default=0)
    parser.add_argument("--confidence", dest="confidence", 
                        help="Object confidence to filter predictions", default=0.6)
    parser.add_argument("--nms_thresh", dest="nms_thresh", 
                        help="NMS Threshhold", default=0.4)
    parser.add_argument("--cfg", dest='cfgfile', 
                        help="Config file",
                        default="cfg/yolov3.cfg", type=str)
    parser.add_argument("--weights", dest='weightsfile', 
                        help="weightsfile",
                        default="yolov3.weights", type=str)
    parser.add_argument("--datacfg", dest="datafile", 
                        help="Config file containing the configuration for the dataset",
                        type=str, default="cfg/coco.data")
    parser.add_argument("--reso", dest='reso', 
                        help="Input resolution of the network. Increase to increase \
                            accuracy. Decrease to increase speed",
                        default="416", type=str)
    parser.add_argument("--plot-conf", dest="plot_conf", type=float,
                        help="Bounding box plotting confidence", default=0.8)
    return parser.parse_args()

def get_test_input(input_dim):
    """A single test image"""
    img = cv2.imread("dog-cycle-car.png")
    img = cv2.resize(img, (input_dim, input_dim)) 
    img_ =  img[:,:,::-1].transpose((2,0,1))
    img_ = img_[np.newaxis,:,:,:]/255.0
    img_ = torch.from_numpy(img_).float()
    
    img_ = img_.to(device)
    
    return img_

def prep_image(img, model_dim):
    """
    Prepare image for input to the neural network. 
    """
    orig_im = img
    orig_dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (model_dim, model_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, orig_dim

def write(x, img):
    """
    Arguments
    ---------
    x : array of float
        [batch_index, x1, y1, x2, y2, objectness, label, probability]
        where x1, y1 is left, bottom corner
    img : numpy array
        original image

    Returns
    -------
    img : numpy array
        Image with bounding box drawn


    Note
    ----
    OpenCV draws and calls x1, y1, x2, y2 differently:

    cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 2)

    x1,y1 ------
    |          |
    |          |
    |          |
    --------x2,y2

    Such that new_y1 = y2, new_y2 = y1

    """

    # if no box, just line
    if x[0] == x[2] or x[1] == x[3]:
        return img

    # Scale up thickness of lines to match size of original image
    scale_up = int(img.shape[0]/416)

    if x[-1] is not None:

        x = [int(n) for n in x]
        x_copy = copy.deepcopy(x)
        # print('old x ', x)
        x[1] = x_copy[3]
        x[3] = x_copy[1]
        ## top, left corner of rectangle
        c1 = (x[0],x[1])
        ## bottom, right corner of rectangle 
        c2 = (x[2],x[3])
        label = int(x[-2])
        label = "{0}".format(classes[label])
        color = random.choice(colors)
        print(c1, c2)
        cv2.rectangle(img, c1, c2, color, thickness=scale_up)
        t_size = cv2.getTextSize(text=label,
                                 fontFace=cv2.FONT_HERSHEY_PLAIN, 
                                 fontScale=1*scale_up//2, 
                                 thickness=1*scale_up)[0]
        c3 = (x[0], x[3])
        c4 = c3[0] + t_size[0] + 3, c3[1] + t_size[1] + 4
        cv2.rectangle(img, c3, c4, color, thickness=-1)
        cv2.putText(img, label, (x[0], x[3] + t_size[1] + 4), 
                    fontFace=cv2.FONT_HERSHEY_PLAIN,
                    fontScale=1*scale_up//2,
                    color=[225,255,255],
                    thickness=1*scale_up)
    return img

def remove_empty_boxes(output):
    """In processed output prediction, if a box is emtpy (height
    or width is zero), then it is removed from the final output array"""
    ary = []
    output_ = output.clone()
    for i in range(output_.size(0)):
        inner_ary = output_[i].numpy()
        if inner_ary[0] == inner_ary[2] and inner_ary[1] == inner_ary[3]:
            continue
        else:
            ary.append(inner_ary)
    return torch.tensor(ary, dtype=torch.int32).view(-1, 7)

if __name__ == '__main__':
    args = arg_parse()
    confidence = float(args.confidence)
    nms_thesh = float(args.nms_thresh)
    start = 0
        
    print("Loading network.....")
    model = Darknet(cfgfile=args.cfgfile, train=False)
    model.load_state_dict(torch.load(args.weightsfile))
    print("Network successfully loaded")

    model.net_info["height"] = args.reso
    model_dim = int(model.net_info["height"])
    assert model_dim % 32 == 0 
    assert model_dim > 32
    num_classes = int(model.net_info["classes"])
    bbox_attrs = 5 + num_classes

    model = model.to(device)
    model.eval()
    
    if args.video: # video file
        videofile = args.video
        cap = cv2.VideoCapture(videofile)
    else:
        # On mac, 0 is bulit-in camera and 1 is USB webcam on Mac
        # On linux, 0 is video0, 1 is video1 and so on
        cap = cv2.VideoCapture(args.source)
    assert cap.isOpened(), 'Cannot capture source'
    
    frames = 0
    start = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            
            image, orig_im, orig_dim = prep_image(frame, model_dim)
            orig_dim = torch.FloatTensor(orig_dim).repeat(1,2)
            orig_h, orig_w = image.shape[0], image.shape[1]

        # Predict on input test image
        image = image.to(device)
        with torch.no_grad():        
            output = model(image)

        # NB, output is:
        # [batch, image_id, [x_center, y_center, width, height, objectness_score, class_score1, class_score2, ...]]
        
        if output.shape[0] > 0:

            output = output.squeeze(0)
            output = process_output(output, num_classes)

            # Center to corner
            output_ = copy.deepcopy(output)
            output[:,0] = output_[:,0] - output_[:,2]/2
            output[:,1] = output_[:,1] - output_[:,3]/2
            output[:,2] = output_[:,0] + output_[:,2]/2
            output[:,3] = output_[:,1] + output_[:,3]/2

            # # NMS
            # keep = nms(output[:,:4], output[:,-1])
            # output = torch.index_select(output, 0, keep[0])

            # Scale
            output = output[output[:,-1] >= float(args.plot_conf), :]
            outputs = []

            if output.size(0) > 0:
                # Reshape the bboxes to reflect original h, w
                scale = min(model_dim/orig_w, model_dim/orig_h)
                output[:,:4] /= scale
                new_w = scale*orig_w
                new_h = scale*orig_h
                del_h = (model_dim - new_h)/2
                del_w = (model_dim - new_w)/2
                add_matrix = torch.tensor(np.array([[del_w, del_h, del_w, del_h]]), dtype=torch.float32)
                output[:,:4] -= add_matrix
                # If bboxes have negative values, make them zero (end up on image border instead)
                output[:,[0,2]] = torch.clamp(output[:,[0,2]], 0.0, orig_dim[0,0])
                output[:,[1,3]] = torch.clamp(output[:,[1,3]], 0.0, orig_dim[0,1])
                outputs = list(np.asarray(output[:,:8]))

            classes = load_classes('data/obj.names')
            colors = pkl.load(open("pallete", "rb"))
            
            # # Test resizing to model dim
            # img_ = Image.fromarray(np.uint8(img_))
            # img_ = F.resize(img_, (model_dim, model_dim))
            # img_ = np.asarray(img_)
            list(map(lambda x: write(x, orig_im), outputs))
            
            cv2.imshow("frame", orig_im)
            cv2.imwrite('detection.png', orig_im)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break
            frames += 1
            print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
            
        else:
            break
