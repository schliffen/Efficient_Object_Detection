#
#
#
# Conversion script for tiny-yolo-voc to Metal.
# Needs Python 3 and Keras 1.2.2

import os
import numpy as np
import keras
from keras.models import Sequential, load_model

root_path = '/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/pretrained_models/'

mtype = 'squeezenet'

model_path = root_path + "raccoon_squeezenet_backend.h5" # "tinyyolo_voc2007_modelweights.h5"
dest_path = root_path

# Load the model that was exported by YAD2K.
model = load_model(model_path)
model.summary()

print("\nConverting parameters...")

def export_conv_and_batch_norm(conv_layer, bn_layer, name):
    print(name)

    # Keras stores the learnable weights for a BatchNormalization layer
    # as four separate arrays:
    #   0 = gamma (if scale == True)
    #   1 = beta (if center == True)
    #   2 = moving mean
    #   3 = moving variance
    bn_weights = bn_layer.get_weights()
    gamma = bn_weights[0]
    beta = bn_weights[1]
    mean = bn_weights[2]
    variance = bn_weights[3]

    # Get the weights for the convolution layer and transpose from
    # Keras order to Metal order.
    conv_weights = conv_layer.get_weights()[0]
    conv_weights = conv_weights.transpose(3, 0, 1, 2).flatten()

    # We're going to save the conv_weights and the BN parameters
    # as a single binary file.
    combined = np.concatenate([conv_weights, mean, variance, gamma, beta])
    combined.tofile(os.path.join(dest_path, name + ".bin"))

def export_conv(conv_layer, name):
    print(name)
    conv_weights = conv_layer.get_weights()[0]
    conv_weights = conv_weights.transpose(3, 0, 1, 2)
    conv_weights.tofile(os.path.join(dest_path, name + ".bin"))

if __name__ == '__main__':
    #
    if mtype == 'tinyyolo':
        export_conv_and_batch_norm(model.layers[1], model.layers[2], "tconv1")
        export_conv_and_batch_norm(model.layers[5], model.layers[6], "tconv2")
        export_conv_and_batch_norm(model.layers[9], model.layers[10], "tconv3")
        export_conv_and_batch_norm(model.layers[13], model.layers[14], "tconv4")
        export_conv_and_batch_norm(model.layers[17], model.layers[18], "tconv5")
        export_conv_and_batch_norm(model.layers[21], model.layers[22], "tconv6")
        export_conv_and_batch_norm(model.layers[25], model.layers[26], "tconv7")
        export_conv_and_batch_norm(model.layers[28], model.layers[29], "tconv8")
        export_conv(model.layers[31], "tconv9")
        print('tiny yolo weights converted to metal!')

    elif mtype == 'squeezenet':
        export_conv_and_batch_norm(model.layers[1], model.layers[2], "tconv1")


    else:
        print('select model type')



print("Done!")