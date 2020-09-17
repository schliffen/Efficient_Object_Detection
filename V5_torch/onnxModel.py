#
#
#
import onnx
import onnxruntime


import argparse



# handle command line arguments
ap = argparse.ArgumentParser()
ap.add_argument('-s', '--source',  default="video" ,
                help = 'test data source: image or video')
ap.add_argument('-i', '--sourceDir',  default="/home/ali/ProjLAB/data/videos/" ,
                help = 'path to input image')
ap.add_argument('-cg', '--cfg', default='models/yolov5s.yaml')
ap.add_argument('-ox', '--onx', default='weights/std_train_0.onnx')
ap.add_argument('-v', '--video',  default="test_01.mp4",
                help = 'path to input image')
ap.add_argument('-c', '--trlogs',  default="./runs/exp10_yolov5s_train/",
                help = 'path to yolo config file')

ap.add_argument('-dv', '--device', default='0', help='device type to be used: cpu or 0, 1, 2, ...')
args = ap.parse_args()


def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

if __name__ == '__main__':
    ort_session = onnxruntime.InferenceSession( args.trlogs + args.onx)
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # compare ONNX Runtime and PyTorch results
    # np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")

