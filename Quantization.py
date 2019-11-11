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
from coremltools.models.neural_network import quantization_utils
from keras import backend
# from keras.applications.mobilenet_v2 import MobileNetV2
from coremltools.models.neural_network.quantization_utils import *



# loading moilenet v2

#base_model = MobileNetV2(input_shape=(224,224, 3),
#                                               include_top=True,
#                                               weights='imagenet')
#base_model.summary()
# model_json = base_model.to_json()
# open('mobilenetv2_arc.json', 'w').write(model_json)
# base_model.save_weights('mobilenetv2_weigt.h5', overwrite=True)

# mlmodel =  coremltools.converters.keras.convert(base_model)
# import coremltools.converters.keras as k

# def save_model():
#     model = MobileNet(input_shape=(128,128,3), include_top=False)
#     model.save('temp.h5')
#
# def convert():
#     model = k.convert('temp.h5',
#                       input_names=['input'],
#                       output_names=['output'],
#                       model_precision='float16',
#                       custom_conversion_functions={'relu6': relu6, 'DepthwiseConv2D': DepthwiseConv2D})
#     model.save('temp.model')
#
# save_model()
# convert()




# from tensorflow.python.keras.utils.generic_utils import CustomObjectScope

# def relu6(x):
#     res = 0 if x < 0 else x
#     res = 6 if x > 6 else x
#     return x



# mobnet = MobileNet(weights='imagenet', include_top=True, pooling='avg')

# mobnet.summary()
# mobnet.save('mobilenetFeatExt.hdf5')
# model_json = mobnet.to_json()
# open('mobilenetv2_arc.json', 'w').write(model_json)
# mobnet.save_weights('mobilenetv2_weigt.h5', overwrite=True)
#
# model_architecture = './mobilenetv2_arc.json'
# model_weights = './mobilenetv2_weigt.h5'

# model_structure = Path(model_architecture).read_text()
#
# with CustomObjectScope({'relu6': relu6 ,'DepthwiseConv2D': DepthwiseConv2D }):
#     model = model_from_json(model_structure)
#     model.load_weights(model_weights)
    # output_labels = ['0', '1', '2', '3', '4', '5', '6']


#
# selector = AdvancedQuantizedLayerSelector(
#     skip_layer_types=['batchnorm', 'bias', 'depthwiseConv'],
#     minimum_conv_kernel_channels=4,
#     minimum_conv_weight_count=4096)
#
# # quantized_model = quantize_weights(model, 8, selector=selector)
#
#
#
# coreml_model = coremltools.converters.keras.convert(
#     model,
#     # class_labels=output_labels,
#     add_custom_layers=True,
#     image_scale=1/255,
#     image_input_names='image')
#
#
#
#
#
# coreml_model.save('mobilenetEeatExt.mlmodel')
#




# mobilenet to pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph



# Create, compile and train model...
#
# frozen_graph = freeze_session(K.get_session(),
#                               output_names=[out.op.name for out in mobnet.outputs])
# tf.train.write_graph(frozen_graph, './', 'mobilenetfeatExt.pbtxt', as_text=True)
# tf.train.write_graph(frozen_graph, './', 'mobilenetfeatExt.pb', as_text=False)

# Define L2 norm
def customLoss_L2(y_true, y_pred):
    sum_of_channel_loss = K.sum(K.square(y_pred - y_true), axis=-1)
    final_loss = K.mean(sum_of_channel_loss, axis=-1)
    return final_loss

def relu6(x):
    return relu(x, max_value=6)



# functions = ["linear", "linear_lut", "kmeans"]
#
# for function in functions :
#     for bit in [16,8,7,6,5,4,3,2,1]:
#         print("processing ",function," on ",bit,".")
#



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

    model = coremltools.models.MLModel(root_path + "mesutmodel_1.mlmodel")
    # lin_quant_model = quantize_weights(model, bit, function)
    # lin_quant_model.short_description = str(bit)+" bit per quantized weight, using "+ function+"."
    # lin_quant_model.save(model+"_"+function+"_"+str(bit)+".mlmodel")
    # compare_models(model, lin_quant_model, 'testing_data/pizza')
    # lin_quant_model.save(root_path + 'mesutmodel_2.mlmodel')

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