#
#
#
import os, glob, sys
import numpy as np
import math
import time
# import required packages
import cv2
import torch
import onnx
import onnxruntime
from onnx import optimizer, shape_inference
import onnx.utils

import torchvision
import matplotlib
matplotlib.use('qt5agg')
import matplotlib.pyplot as plt
import uuid

import argparse

from demoModelYolov5 import Model


# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--source',  default="video" ,
                help = 'test data source: image or video')
ap.add_argument('-i', '--sourceDir',  default="/home/ali/ProjLAB/data/videos/" ,
                help = 'path to input image')
ap.add_argument('-cg', '--cfg', default='models/yolov5m.yaml')
ap.add_argument('-ms', '--msdict', default='weights/std_train_71.pt')
# ap.add_argument('-ox', '--onx', default='weights/std_train_0.onnx')
ap.add_argument('-v', '--video',  default="test_01.mp4",
                help = 'path to input image')
ap.add_argument('-c', '--trlogs',  default="./runs/exp1_yolov5m_train/",
                help = 'path to yolo config file')
ap.add_argument('-w', '--best',  default= "weights/best.pt",
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes',  default="./weights/classes.txt",
                help = 'path to text file containing class names')
ap.add_argument('-si', '--cimgs', default='')

ap.add_argument('-dv', '--device', default='0', help='device type to be used: cpu or 0, 1, 2, ...')
args = ap.parse_args()

# init configs
imgsz = 416
conf_thres = 0.001
iou_thres = 0.6


def attempt_load(weights, map_location):

    model = torch.load(weights, map_location=map_location)['model'].float().fuse().eval()

    return model

def select_device(device='', apex=False, batch_size=None):
    # device = 'cpu' or '0' or '0,1,2,3'
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:  # if device requested other than 'cpu'
        os.environ['CUDA_VISIBLE_DEVICES'] = device  # set environment variable
        assert torch.cuda.is_available(), 'CUDA unavailable, invalid device %s requested' % device  # check availablity

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2  # bytes to MB
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:  # check that batch_size is compatible with device_count
            assert batch_size % ng == 0, 'batch-size %g not multiple of GPU count %g' % (batch_size, ng)
        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA ' + ('Apex ' if apex else '')  # apex for mixed precision https://github.com/NVIDIA/apex
        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            print("%sdevice%g _CudaDeviceProperties(name='%s', total_memory=%dMB)" %
                  (s, i, x[i].name, x[i].total_memory / c))
    else:
        print('Using CPU')

    print('')  # skip a line
    return torch.device('cuda:0' if cuda else 'cpu')
def make_divisible(x, divisor):
    # Returns x evenly divisble by divisor
    return math.ceil(x / divisor) * divisor

def check_img_size(img_size, s=32):
    # Verify img_size is a multiple of stride s
    new_size = make_divisible(img_size, int(s))  # ceil gs-multiple
    if new_size != img_size:
        print('WARNING: --img-size %g must be multiple of max stride %g, updating to %g' % (img_size, s, new_size))
    return new_size
def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
#

def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)
#
def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

#
def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, merge=False, classes=None, agnostic=False):
    """Performs Non-Maximum Suppression (NMS) on inference results

    Returns:
         detections with shape: nx6 (x1, y1, x2, y2, conf, cls)
    """
    if prediction.dtype is torch.float16:
        prediction = prediction.float()  # to FP32

    nc = prediction[0].shape[1] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    # Settings
    min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
    max_det = 300  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        x = x[xc[xi]]  # confidence

        # If none remain process next image
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])

        # Detections matrix nx6 (xyxy, conf, cls)
        if multi_label:
            i, j = (x[:, 5:] > conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)
        else:  # best class only
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]

        # Filter by class
        if classes:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        # Apply finite constraint
        # if not torch.isfinite(x).all():
        #     x = x[torch.isfinite(x).all(1)]

        # If none remain process next image
        n = x.shape[0]  # number of boxes
        if not n:
            continue

        # Sort by confidence
        # x = x[x[:, 4].argsort(descending=True)]

        # Batched NMS
        c = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
        boxes, scores = x[:, :4] + c, x[:, 4]  # boxes (offset by class), scores
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)
        if i.shape[0] > max_det:  # limit detections
            i = i[:max_det]
        if merge and (1 < n < 3E3):  # Merge NMS (boxes merged using weighted mean)
            try:  # update boxes as boxes(i,4) = weights(i,n) * boxes(n,4)
                iou = box_iou(boxes[i], boxes) > iou_thres  # iou matrix
                weights = iou * scores[None]  # box weights
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)  # merged boxes
                if redundant:
                    i = i[iou.sum(1) > 1]  # require redundancy
            except:  # possible CUDA error https://github.com/ultralytics/yolov3/issues/1139
                print(x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) > time_limit:
            break  # time limit exceeded

    return output

def clip_coords(boxes, img_shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    boxes[ 0] = max(0, boxes[0])  # x1
    boxes[ 1] = max(0, boxes[1])  # y1
    boxes[ 2] = min(img_shape[1], boxes[2])  # x2
    boxes[ 3] = min(img_shape[0], boxes[3])  # y2
    return boxes

def export_onnx(model, path, device):
    dummy_input = torch.randn(1, 3, 416, 416, device=device)
    model_dir = path[:-3] + '.onnx'
    if args.device == 'cpu':
        torch.onnx.export(model,
                          dummy_input,
                          model_dir,
                          input_names='inp',
                          output_names='yout'
                          )
        # Test load model back in with onnx
        print("Test loading model with ONNX")
        original_model = onnx.load( model_dir )
        # optimizing the model
        # A full list of supported optimization passes can be found using get_available_passes()
        all_passes = optimizer.get_available_passes()
        print("Available optimization passes:")
        passes = []
        for p in all_passes:
            passes.append(p)
            print(p)
        print()

        # # Pick one pass as example
        # passes = ['fuse_consecutive_transposes']
        #
        # # Apply the optimization on the original model
        # optimized_model = optimizer.optimize(original_model, passes)
        # polished_model = onnx.utils.polish_model(original_model)
        #
        # # save the model
        # onnx.save_model(optimized_model, path[:-3] + '_optimized.onnx' )


    else:
        torch.onnx.export(model,
                          dummy_input,
                          model_dir,
                          input_names='inp',
                          output_names='yout'
                          )
        # Test load model back in with onnx
        print("Test loading model with ONNX")
        original_model = onnx.load( model_dir )
        # optimizing the model
        # A full list of supported optimization passes can be found using get_available_passes()
        all_passes = optimizer.get_available_passes()
        print("Available optimization passes:")
        for p in all_passes:
            print(p)
        print()

        # Pick one pass as example
        passes = ['fuse_consecutive_transposes']

        # Apply the optimization on the original model
        optimized_model = optimizer.optimize(original_model, passes)
        polished_model = onnx.utils.polish_model(model)

        # save the model
        onnx.save_model(optimized_model, path[:-3] + '_optimized.onnx' )

    # # Check that the IR is well formed
    # onnx.checker.check_model(onnx_model)
    # # Print a human readable representation of the graph
    # onnx.helper.printable_graph(onnx_model.graph)

    print("Test successful!")
    return onnx_model

def save_torch_model(model, path):
    torch.save( model.state_dict(), './runs/exp0_yolov5s_train/statedict_yolov5s.pth')


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()


if __name__ == '__main__':
    #
    augment = False
    merge = False

    device = select_device(args.device, batch_size=1)

    # Load model
    # model = attempt_load(args.trlogs + args.weights, map_location=device)  # load FP32 model
    # imgsz = check_img_size(imgsz, s=model.stride.max())  # check img_size
    # torch.save(model.state_dict(), args.trlogs + args.msdict)
    # second way of loading the model
    model = Model(args.cfg, nc=1).to(device)
    #
    # torch.save(model.state_dict(), args.trlogs + args.msdict)

    # loading model weights (state dict)
    model.load_state_dict(torch.load(args.trlogs + args.msdict)) # './runs/exp0_yolov5s_train/statedict_yolov5s_2.pth'
    # saving as onnx model
    onnx_model = export_onnx(model, args.trlogs + args.msdict, device)

    # loading onnx inference
    ort_session = onnxruntime.InferenceSession( (args.trlogs + args.msdict)[:-3] + '.onnx' )
    # onx_model =    onnx.load_model((args.trlogs + args.msdict)[:-3] + '.onnx')



    # Half
    half = device.type != 'cpu'  # half precision only supported on CUDA
    if half:
        model.half()
    # config
    model.eval()

    # define video processor
    cap = cv2.VideoCapture(args.sourceDir + args.video)
    # Disable gradients
    t0, t1 = 0., 0.

    cap.set(1, 9000)

    cv2.namedWindow('img', cv2.WINDOW_AUTOSIZE)

    rat = True
    while rat:
        rat, img = cap.read()
        img = cv2.resize(img, (imgsz, imgsz))
        primg = img.copy()
        # current frame
        currFrameNo = cap.get(cv2.CAP_PROP_POS_FRAMES)
        with torch.no_grad():
            # image to tensor
            imgt = torch.tensor(np.expand_dims(img.transpose(2,0,1)/255.,0)).half()
            # img to gpu
            imgtc = imgt.to(device, non_blocking=True)
            # if device != 'cpu':
            #     imgtc = imgt.cuda()
            # Run model

            # onnx model
            # compute ONNX Runtime output prediction
            # ort_inputs = {ort_session.get_inputs()[0].name: to_numpy( imgtc ) }
            # ort_outs = ort_session.run(None, ort_inputs)

            t = time_synchronized()
            torch_out, train_out = model(imgtc, augment=augment)  # inference and training outputs
            t0 += time_synchronized() - t

            # comparing onnx and pytorch results
            # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

            # Run NMS
            t = time_synchronized()
            output = non_max_suppression(torch_out, conf_thres=conf_thres, iou_thres=iou_thres, merge=merge)
            t1 += time_synchronized()

            # Statistics per image
            for si in range(output[0].shape[0]):

                if output[0][si,4].cpu().numpy() < .4:
                    continue
                # Clip boxes to image bounds
                pred = output[0][si,:].cpu().numpy()
                pred = clip_coords(pred, (imgsz, imgsz) )
                # pred = output[0][si, :].cpu().numpy()
                #
                # box = pred[:, :4].clone()  # xyxy
                # print( box)

                cv2.rectangle(primg, (pred[0], pred[1]), (pred[2], pred[3]), (1,1,255), 1)
                cv2.putText(primg, str(pred[4]), (pred[0], pred[3]), cv2.FONT_HERSHEY_COMPLEX, 1, (1,255,1))



            # wether to save the image
            cv2.imshow('img', primg)
            key = cv2.waitKey(0)
            if  key == ord('s'):
                cv2.imwrite( img, args.cimgs + str(uuid.uuid4()) + '.jpg' )




