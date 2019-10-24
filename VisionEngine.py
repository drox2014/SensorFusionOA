import os

import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils
from utils import label_map_util
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()
PATH_TO_FASTERRCNN_CKPT = os.path.join(CWD_PATH, 'data', 'object_detection_models', 'faster_rcnn_inception_v2.pb')
PATH_TO_YOLO_CKPT = os.path.join(CWD_PATH, 'data', 'object_detection_models', 'yolov3_coco_v1.1.pb')
PATH_TO_LABELS_TFOD_API = os.path.join(CWD_PATH, 'data', 'labels.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 6
num_classes = 10
input_size = 608
# Load the label map.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS_TFOD_API, use_display_name=True)

# tensors of graph
return_elements_yolo = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
return_elements_frcnn = ["image_tensor:0", "detection_boxes:0", "detection_scores:0", "detection_classes:0",
                         "num_detections:0"]

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
graph_def = tf.GraphDef()
# with tf.gfile.FastGFile(PATH_TO_FASTERRCNN_CKPT, "rb") as f:
#     serialized_graph = f.read()
#     graph_def.ParseFromString(serialized_graph)
#
# with tf.gfile.FastGFile(PATH_TO_YOLO_CKPT, "rb") as f:
#     serialized_graph = f.read()
#     graph_def.ParseFromString(serialized_graph)
#
# with detection_graph.as_default():
#     return_elements_frcnn = tf.import_graph_def(graph_def, return_elements=return_elements_frcnn)
#     return_elements_yolo = tf.import_graph_def(graph_def, return_elements=return_elements_yolo)

with detection_graph.as_default():
    with tf.gfile.GFile(PATH_TO_FASTERRCNN_CKPT, 'rb') as fid:
        od_graph_def = tf.GraphDef()
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    with tf.gfile.GFile(PATH_TO_YOLO_CKPT, 'rb') as fid:
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
# Define tensors for YOLO V3
input_tensor = detection_graph.get_tensor_by_name('input/input_data:0')
sb_boxes = detection_graph.get_tensor_by_name('pred_sbbox/concat_2:0')
mb_boxes = detection_graph.get_tensor_by_name('pred_mbbox/concat_2:0')
lb_boxes = detection_graph.get_tensor_by_name('pred_lbbox/concat_2:0')

gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.5))
sess = tf.Session(config=gpu_options, graph=detection_graph)

image = cv2.imread("20190508_163439.jpg")
image_expanded = np.expand_dims(image, axis=0)
frame_size = image.shape[:2]
image_data = utils.image_preporcess(np.copy(image), [input_size, input_size])
image_data = np.expand_dims(image_data, axis=0)

(boxes, scores, classes, num) = sess.run([
    detection_boxes,
    detection_scores,
    detection_classes,
    num_detections
], feed_dict={image_tensor: image_expanded})

pred_sbbox, pred_mbbox, pred_lbbox = sess.run([
    sb_boxes,
    mb_boxes,
    lb_boxes
], feed_dict={input_tensor: image_data})

pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                            np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
bboxes = utils.nms(bboxes, 0.45, method='nms')
result = np.asarray(utils.draw_bbox(image, bboxes))

vis_util.visualize_boxes_and_labels_on_image_array(
    image,
    np.squeeze(boxes),
    np.squeeze(classes).astype(np.int32),
    np.squeeze(scores),
    category_index,
    use_normalized_coordinates=True,
    line_thickness=8,
    min_score_thresh=0.60)

cv2.imwrite("predicted_frcnn.jpg", image)
cv2.imwrite("predicted_yolo.jpg", result)

# Press any key to close the image
# cv2.waitKey(0)
#
# # Clean up
# cv2.destroyAllWindows()
