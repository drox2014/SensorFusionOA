import os

import numpy as np
import tensorflow as tf

from utils import label_map_util
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()
PATH_TO_FASTERRCNN_CKPT = os.path.join(CWD_PATH, 'data', 'object_detection_models', 'frozen_inference_graph.pb')
PATH_TO_LABELS_TFOD_API = os.path.join(CWD_PATH, 'data', 'object_detection_models', 'labels.pbtxt')
# Number of classes the object detector can identify
NUM_CLASSES = 10
INPUT_SIZE = 608

# Load the label map.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_TFOD_API,
                                                                    use_display_name=True)
# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
graph_def = tf.GraphDef()
with detection_graph.as_default():
    with tf.gfile.GFile(PATH_TO_FASTERRCNN_CKPT, 'rb') as fid:
        od_graph_def = tf.GraphDef()
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

sess = tf.Session(graph=detection_graph)

# Define tensors for TF Object Detection API
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
sess = tf.Session(config=gpu_options, graph=detection_graph)


def detect_objects(frame):
    image_expanded = np.expand_dims(frame, axis=0)

    (boxes, scores, classes, num) = sess.run([
        detection_boxes,
        detection_scores,
        detection_classes,
        num_detections
    ], feed_dict={image_tensor: image_expanded})

    vis_util.visualize_boxes_and_labels_on_image_array(
        frame,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8,
        min_score_thresh=0.60)

    return frame
