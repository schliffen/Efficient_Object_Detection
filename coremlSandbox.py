#
# testing coreml conversion
#
import numpy as np
from keras.models import Model, load_model
import tensorflow as tf
from keras.layers import Reshape, Activation, Conv2D, Input, MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import concatenate
from coremltools.proto import NeuralNetwork_pb2

import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    import coremltools

# ---------------------------< THE PARAMETERS >---------------------------
labels = ['raccoon']
anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
#
input_size = 416
input_size = 416
labels   = list(labels)
nb_class = len(labels)
nb_box   = len(anchors)//2
class_wt = np.ones(nb_class, dtype='float32')
anchors  = anchors
max_box_per_image = 10
# loading the model and converting it step by step
#

# Step 1 building up the model

def fire_module(x, fire_id, squeeze=16, expand=64):
    s_id = 'fire' + str(fire_id) + '/'

    x     = Conv2D(squeeze, (1, 1), padding='valid', name=s_id + sq1x1)(x)
    x     = Activation('relu', name=s_id + relu + sq1x1)(x)

    left  = Conv2D(expand,  (1, 1), padding='valid', name=s_id + exp1x1)(x)
    left  = Activation('relu', name=s_id + relu + exp1x1)(left)

    right = Conv2D(expand,  (3, 3), padding='same',  name=s_id + exp3x3)(x)
    right = Activation('relu', name=s_id + relu + exp3x3)(right)

    x = concatenate([left, right], axis=3, name=s_id + 'concat')

    return x


def arg_select(args):
    return args[0]




def create_model():
    # inp = Input(shape=(25, 25, 512))

    # define some auxiliary variables and the fire module
    sq1x1  = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu   = "relu_"

    # define the model of SqueezeNet

    input_image = Input(shape=(input_size , input_size, 3))
    # true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))

    x_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
    x_1 = Activation('relu', name='relu_conv1')(x_0)
    x_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x_1)

    # model_01 = Model(input_image, x_2)

    x_3 = fire_module(x_2, fire_id=2, squeeze=16, expand=64)
    x_4 = fire_module(x_3, fire_id=3, squeeze=16, expand=64)
    x_5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x_4)

    x_6 = fire_module(x_5, fire_id=4, squeeze=32, expand=128)
    x_7 = fire_module(x_6, fire_id=5, squeeze=32, expand=128)
    x_8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x_7)

    x_9  = fire_module(x_8, fire_id=6, squeeze=48, expand=192)
    x_10 = fire_module(x_9, fire_id=7, squeeze=48, expand=192)
    x_11 = fire_module(x_10, fire_id=8, squeeze=64, expand=256)
    x_12 = fire_module(x_11, fire_id=9, squeeze=64, expand=256)

    feature_extractor = Model(input_image, x_12)
    # self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)

    # feature_extractor.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])

    # Creating the object detection model
    print(feature_extractor.layers[-1].output_shape)
    grid_h, grid_w = feature_extractor.layers[-1].output_shape[1:3]
    features = feature_extractor(input_image)


    ox_1 = Conv2D(nb_box * (4 + 1 + nb_class),
               (1,1), strides=(1,1),
               padding='same',
               name='DetectionLayer',
               kernel_initializer='lecun_normal')(features)
    # ox_2 = Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))( ox_1 )
    # ox_3 = Lambda(arg_select)( [ox_2, true_boxes] )                       # look here!
    return Model( input_image, ox_1 )


# converting lambda layer
def cnvrt_lmbd(layer):
    if layer.function == arg_select:
        params = NeuralNetwork_pb2.CustomLayerParams()
        # The name of the Swift or Obj-C class that implements this layer.
        params.className = "ArgSelect"
        # The desciption is shown in Xcode's mlmodel viewer.
        params.description = "selecting the first argument"
        return params
    else:
        return None




if __name__ == '__main__':
    # define some auxiliary variables and the fire module
    sq1x1  = "squeeze1x1"
    exp1x1 = "expand1x1"
    exp3x3 = "expand3x3"
    relu   = "relu_"
    #
    # # define the model of SqueezeNet
    #
    # input_image = Input(shape=(input_size , input_size, 3))
    # true_boxes = Input(shape=(1, 1, 1, max_box_per_image , 4))
    #
    # x_0 = Conv2D(64, (3, 3), strides=(2, 2), padding='valid', name='conv1')(input_image)
    # x_1 = Activation('relu', name='relu_conv1')(x_0)
    # x_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x_1)
    #
    # model_01 = Model(input_image, x_2)
    #
    # x_3 = fire_module(x_2, fire_id=2, squeeze=16, expand=64)
    # x_4 = fire_module(x_3, fire_id=3, squeeze=16, expand=64)
    # x_5 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool3')(x_4)
    #
    # x_6 = fire_module(x_5, fire_id=4, squeeze=32, expand=128)
    # x_7 = fire_module(x_6, fire_id=5, squeeze=32, expand=128)
    # x_8 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool5')(x_7)
    #
    # x_9  = fire_module(x_8, fire_id=6, squeeze=48, expand=192)
    # x_10 = fire_module(x_9, fire_id=7, squeeze=48, expand=192)
    # x_11 = fire_module(x_10, fire_id=8, squeeze=64, expand=256)
    # x_12 = fire_module(x_11, fire_id=9, squeeze=64, expand=256)
    #
    # feature_extractor = Model(input_image, x_12)
    # # self.feature_extractor.load_weights(SQUEEZENET_BACKEND_PATH)
    #
    # # feature_extractor.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])
    #
    # # Creating the object detection model
    # print(feature_extractor.layers[-1].output_shape)
    # grid_h, grid_w = feature_extractor.layers[-1].output_shape[1:3]
    # features = feature_extractor(input_image)
    # #
    # # make the object detection layer
    # #
    # post_feature = Conv2D(nb_box * (4 + 1 + nb_class),
    #                 (1,1), strides=(1,1),
    #                 padding='same',
    #                 name='DetectionLayer',
    #                 kernel_initializer='lecun_normal')(features)
    #
    # # FextPlusConvModel - DirectConversion is successful until here
    # fextconv = Model( input_image, post_feature )
    #
    # fext_reshape = Reshape((grid_h, grid_w, nb_box, 4 + 1 + nb_class))( post_feature )
    #
    #
    # output = Lambda(lambda args: args[0])([fext_reshape, true_boxes])
    #
    #
    # model = Model([input_image, true_boxes], output)


    # Compiling the model --> There is a problem with model compile
    # fmodel = model.compile(optimizer = 'adam', loss='mean_squared_error', metrics=['accuracy'])

    # see the model summary

    squeeznet_weight = '/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/pretrained_models/raccoon_squeezenet_backend.h5' #,
                        # custom_objects={"weighted_loss" : weighted_loss} )
    model = create_model()

    model.summary()

    model.load_weights(squeeznet_weight)
    # Transforming Layer by Layer
    # coreml_model = coremltools.converters.keras.convert(
    #     model,
    #     input_names='image',
    #     image_input_names='image',
    #     output_names='output',
    #     image_scale=1./255.)

    coreml_model = coremltools.converters.keras.convert(
        model,
        input_names="image",
        image_input_names="image",
        output_names="output",
        # add_custom_layers=True,
        # custom_conversion_functions={ "Lambda": cnvrt_lmbd }
    )

    coreml_model.save('featureExtractor_squeezedet.mlmodel')










    #







