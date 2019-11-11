#
# merging pipeline toghether
#
import numpy as np
import keras
from keras.models import *
from keras.layers import *
#
from keras import backend as K
import coremltools as mlt
from coremltools.models import datatypes
from coremltools.proto import NeuralNetwork_pb2
from coremltools.models.pipeline import *
from coremltools.models import neural_network



# Path to the feature extractor of mlmodel
featExt_ml = '/home/ali/CLionProjects/object_detection/SqueezeDet/squeezedet/YOLOSQUEEZe/featureExtractor_squeezedet.mlmodel'

# model parameters
labels = ['raccoon']
anchors = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
input_size = 416
input_size = 416
#
labels   = list(labels)
nb_class = len(labels)
nb_box   = len(anchors)//2
#

# loading feature extractor
#
fext_model  = mlt.models.MLModel(featExt_ml)

# to write reshape part here plus Lambda function --  on coreml
#
# preparing the previous layers

spec = fext_model.get_spec()

# spec.neuralNetwork.preprocessing[0].featureName = "image"

# For some reason the output shape of the "scores" output is not filled in.
# spec.description.output[0].type.multiArrayType.shape.append(num_classes + 1)
# spec.description.output[0].type.multiArrayType.shape.append(num_anchors)


_, grid_h, grid_w = spec.description.output[0].type.multiArrayType.shape
# inputs to the Network
input_features = [ ("features", datatypes.Array(25, 25, nb_box * (4 + 1 + nb_class)))
                   # ("boxes", datatypes.Array(4, nb_box, 1))
                   ]
# Networks output
output_features = [ ("reshaped", datatypes.Array(grid_h, grid_w, nb_box, 4 + 1 + nb_class))
                    # ("raw_coordinates", datatypes.Array(num_anchors, 4))
                    ]
# bulding Neuralnetworks
builder_02 = neural_network.NeuralNetworkBuilder(input_features, output_features)

# adding reshape layer
builder_02.add_reshape( name='reshape',
    input_name='features',
                    output_name='reshaped',
                    target_shape=(grid_h, grid_w, nb_box, 4 + 1 + nb_class),
                    mode=0
                    )

reshape_submodel = mlt.models.MLModel(builder_02.spec)
reshape_submodel.save("reshaper_pip_02.mlmodel")

# adding the lambda function

def swish(x):
    return K.sigmoid(x) * x

def create_model():
    inp = Input(shape=(256, 256, 3))
    x = Conv2D(6, (3, 3), padding="same")(inp)
    x = Lambda(swish)(x)                       # look here!
    x = GlobalAveragePooling2D()(x)
    x = Dense(10, activation="softmax")(x)
    return Model(inp, x)

# The conversion function for Lambda layers.
def convert_lambda(layer):
    # Only convert this Lambda layer if it is for our swish function.
    if layer.function == swish:
        params = NeuralNetwork_pb2.CustomLayerParams()

        # The name of the Swift or Obj-C class that implements this layer.
        params.className = "Swish"

        # The desciption is shown in Xcode's mlmodel viewer.
        params.description = "A fancy new activation function"

        # Set configuration parameters
        # params.parameters["someNumber"].intValue = 100
        # params.parameters["someString"].stringValue = "Hello, world!"

        # Add some random weights
        # my_weights = params.weights.add()
        # my_weights.floatValue.extend(np.random.randn(10).astype(float))

        return params
    else:
        return None


# --------
def arg_select(args):
    return args[0]

#output = Lambda(lambda args: args[0])([fext_reshape, true_boxes])


# inputs to the Network
reshaped_fext = [ ("reshaped", datatypes.Array( grid_h, grid_w, nb_box, 4 + 1 + nb_class ))
                   # ("boxes", datatypes.Array(4, nb_box, 1))
                   ]
# Networks output
lambda_out = [ ("lambda_out", datatypes.Array(grid_h, grid_w, nb_box, 4 + 1 + nb_class))
                    # ("raw_coordinates", datatypes.Array(num_anchors, 4))
                    ]
#
#
# # bulding Neuralnetworks
builder_03 = neural_network.NeuralNetworkBuilder(input_features, output_features)
#
# # adding reshape layer
builder_03.add_reshape( name= 'lambda',
                        input_name='features',
                        output_name='reshaped',
                        target_shape=(grid_h, grid_w, nb_box, 4 + 1 + nb_class),
                        mode=0
                        )
#
reshape_submodel = mlt.models.MLModel(builder_03.spec)
reshape_submodel.save("reshaper_pip_03.mlmodel")
#



print("\nConverting the reshape and lambda model:")

# Convert the model to Core ML.

# This is the alternative method of filling in the CustomLayerParams:
# grab the layer and change its properties directly.











#
# --------------------------------------------<< Extra codes >>--------------------------------
#
# converting lambda layer
# def cnvrt_lmbd(layer):
#     if layer.function == arg_select:
#         params = NeuralNetwork_pb2.CustomLayerParams()
#         # The name of the Swift or Obj-C class that implements this layer.
#         params.className = "ArgSelect"
#         # The desciption is shown in Xcode's mlmodel viewer.
#         params.description = "selecting the first argument"
#         return params
#     else:
#         return None
#
# model = create_model()
#
# coreml_model = mlt.converters.keras.convert(
#     model,
#     input_names="image",
#     image_input_names="image",
#     output_names="output",
#     add_custom_layers=True,
#     custom_conversion_functions={ "Lambda": cnvrt_lmbd })
# # Look at the layers in the converted Core ML model.
# print("\nLayers in the converted model:")
# for i, layer in enumerate(coreml_model._spec.neuralNetwork.layers):
#     if layer.HasField("custom"):
#         print("Layer %d = %s --> custom layer = %s" % (i, layer.name, layer.custom.className))
#     else:
#         print("Layer %d = %s" % (i, layer.name))
#
#
# # Fill in the metadata and save the model.
# coreml_model.author = "AuthorMcAuthorName"
# coreml_model.license = "Public Domain"
# coreml_model.short_description = "Playing with custom Core ML layers"
# coreml_model.input_description["image"] = "Input image"
# coreml_model.output_description["output"] = "The predictions"
# coreml_model.save("PipelineModel.mlmodel")
