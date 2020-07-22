import tensorflow as tf
from tensorflow.python.framework import graph_io
from tensorflow.keras.models import load_model
import argparse

# Clear any previous session.
tf.keras.backend.clear_session()

parser = argparse.ArgumentParser()

parser.add_argument('--s', help="save_pb_dir", dest="spd", default='../Training/Results/Tensorflow_pb/')
parser.add_argument('--m', help="model_fname", dest="mfn", default='../Training/Results/Keras_h5/MFN.h5')

args = parser.parse_args()

def freeze_graph(graph, session, output, save_pb_dir='.', save_pb_name='MFN.pb', save_pb_as_text=False):
    with graph.as_default():
        graphdef_inf = tf.graph_util.remove_training_nodes(graph.as_graph_def())
        graphdef_frozen = tf.graph_util.convert_variables_to_constants(session, graphdef_inf, output)
        graph_io.write_graph(graphdef_frozen, args.spd, save_pb_name, as_text=save_pb_as_text)
        return graphdef_frozen

# This line must be executed before loading Keras model.
tf.keras.backend.set_learning_phase(0)

model = load_model(args.mfn)

session = tf.keras.backend.get_session()

INPUT_NODE = [t.op.name for t in model.inputs]
OUTPUT_NODE = [t.op.name for t in model.outputs]
print(INPUT_NODE, OUTPUT_NODE)
frozen_graph = freeze_graph(session.graph, session, [out.op.name for out in model.outputs], save_pb_dir=args.spd)