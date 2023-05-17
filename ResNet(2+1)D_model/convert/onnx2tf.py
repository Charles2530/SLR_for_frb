from onnx_tf.backend import prepare
import onnx

TF_PATH = "tf_model" # where the representation of tensorflow model will be stored
ONNX_PATH = "F:/中国手语数据集科大/SLR/convert/slr_cnn3d_epoch005.onnx" # path to my existing ONNX model
onnx_model = onnx.load(ONNX_PATH)  # load onnx model
tf_rep = prepare(onnx_model)  # creating TensorflowRep object
tf_rep.export_graph(TF_PATH)