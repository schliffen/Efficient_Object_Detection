#
#
#
import keras
import coremltools
import tensorflow as tf
# from keras.layers import DepthwiseConv2D
from pathlib import Path
from keras import backend as K
from keras.models import model_from_json, load_model
# from keras.applications.mobilenet import MobileNet
# from keras.applications.mobilenet import MobileNet
from keras.activations import relu
from keras.utils.generic_utils import CustomObjectScope
# from keras_applications.mobilenet_v2  import MobileNetV2
# from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector
# from coremltools.models.neural_network import quantization_utils
from keras import backend
# from keras.applications.mobilenet_v2 import MobileNetV2
# from coremltools.models.neural_network.quantization_utils import *




if __name__ == '__main__':
    # loading the keras model

    root_path = '/home/ali/Downloads/'
    model_1_name = 'grid_club_tracking_unet_epochs_6_bach_size_32_n_filters_9_heatmap_variance_2.0.h5'
    model_2_name = '1_11_2019_club_tracking_unet_epochs_10_bach_size_36_n_filters_7_heatmap_variance_2.0.h5'
    #
    #
    # with CustomObjectScope({'customLoss_L2': customLoss_L2}):
    #     kmodel = load_model(root_path + model_2_name)
    #
    #     # model = model_from_json(model_structure)
    #     # model.load_weights(model_weights)
    #     # output_labels = ['0', '1', '2', '3', '4', '5', '6']
    #
    #     coreml_model = coremltools.converters.keras.convert(
    #         kmodel,
    #         # class_labels=output_labels,
    #         add_custom_layers=True,
    #         custom_conversion_functions={'relu6': relu6,},
    #         image_scale=1/255,
    #         image_input_names='image')


    # quantizing the coreml model
    bit = 16
    function = "kmeans"

    # model = coremltools.models.MLModel(root_path + "mesutmodel_1.mlmodel")
    # lin_quant_model = quantize_weights(model, bit, function)
    # lin_quant_model.short_description = str(bit)+" bit per quantized weight, using "+ function+"."
    # lin_quant_model.save(model+"_"+function+"_"+str(bit)+".mlmodel")
    # compare_models(model, lin_quant_model, 'testing_data/pizza')
    # lin_quant_model.save(root_path + 'mesutmodel_2.mlmodel')


    #
    #
    #



    # Print out layer attributes for debugging
    # Sometimes we want to print out weights of a particular layer for debugging purposes.
    # Following is an example showing how we can utilize the protobuf APIs to access any attributes include weight parameters.
    # This code snippet uses the model we created in the previous example.

    import coremltools # coreml conversion were done by coremltools v 0.8
    import numpy as np

    model = coremltools.models.MLModel('mobilenetEeatExt.mlmodel')

    spec = model.get_spec()
    print(spec)

    layer = spec.neuralNetwork.layers[0]
    weight_params = layer.convolution.weights

    print('Weights of {} layer: {}.'.format(layer.WhichOneof('layer'), layer.name))
    print(np.reshape(np.asarray(weight_params.floatValue), (1, 1, 3, 3)))






    from coremltools.models.neural_network.quantization_utils import quantize_weights

    # model = coremltools.models.MLModel('model.mlmodel')
    # Example 1: 8-bit linear
    # quantized_model = quantize_weights(model, nbits=8, quantization_mode="linear")


    # maybe quantizing in macos!!!
    # Example 2: Quantize to FP-16 weights
    # quantized_model = quantize_weights(model, nbits=16)

    # Example 3: 4-bit k-means generated look-up table
    # quantized_model = quantize_weights(model, nbits=4, quantization_mode="kmeans")

    # Example 4: 8-bit symmetric linear quantization skipping bias,
    # batchnorm, depthwise-convolution, and convolution layers
    # with less than 4 channels or 4096 elements
    from coremltools.models.neural_network.quantization_utils import AdvancedQuantizedLayerSelector

    # selector = AdvancedQuantizedLayerSelector(
    #     skip_layer_types=['batchnorm', 'bias', 'depthwiseConv'],
    #     minimum_conv_kernel_channels=4,
    #     minimum_conv_weight_count=4096)
    # quantized_model = quantize_weights(model, 8, quantization_mode='linear_symmetric',
    #                                    selector=selector)

    # Example 5: 8-bit linear quantization skipping the layer with name 'dense_2'
    from coremltools.models.neural_network.quantization_utils import QuantizedLayerSelector


    class MyLayerSelector(QuantizedLayerSelector):

        def __init__(self):
            super(MyLayerSelector, self).__init__()

        def do_quantize(self, layer, **kwargs):
            ret = super(MyLayerSelector, self).do_quantize(layer)
            if not ret or layer.name == 'dense_2':
                return False
                return True


    selector = MyLayerSelector()
    quantized_model = quantize_weights(
        model, 8, quantization_mode='linear', selector=selector)