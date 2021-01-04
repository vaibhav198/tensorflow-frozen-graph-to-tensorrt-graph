import tensorflow as tf
import tensorflow.contrib.tensorrt as trt
from tensorflow.python.framework import graph_io


output_names = ['yolov2-tiny-voc/convolutional9/BiasAdd']

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())

    # Then, we import the graph_def into a new Graph and returns it 
    #with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        #tf.import_graph_def(graph_def, name="prefix")
    return graph_def

frozen_graph = load_graph('/home/nano/tf_to_tftrt/frozen_model.pb') #path frozen



trt_graph = trt.create_inference_graph(
    input_graph_def=frozen_graph,
    outputs=output_names,
    max_batch_size=1,
    max_workspace_size_bytes=1 << 25,
    precision_mode='FP32',
    minimum_segment_size=50
)

graph_io.write_graph(trt_graph, "/home/nano/scripts/",
                     "trt_graph_fp32_nano02.pb", as_text=False)
