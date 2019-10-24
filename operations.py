import os

import PIL.Image as Image
import cv2
import numpy as np
import tensorflow as tf
from PIL import ImageDraw
from sklearn.externals import joblib

from utils import label_map_util
from utils import visualization_utils as vis_util

CWD_PATH = os.getcwd()
PATH_TO_CKPT = os.path.join(CWD_PATH, 'data', 'object_detection_models', 'faster_rcnn_inception_v2.pb')
PATH_TO_LABELS = os.path.join(CWD_PATH, 'data', 'labels.pbtxt')

# Number of classes the object detector can identify
NUM_CLASSES = 6

# Load the label map.
category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

# Load the Tensorflow model into memory.
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

    sess = tf.Session(graph=detection_graph)

# Input tensor is the image
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
num_detections = detection_graph.get_tensor_by_name('num_detections:0')

sf_model = joblib.load('data/sf_svm.pkl')


def find_op(input_vector):
    return sf_model.predict([input_vector])


def count(frame, object_id):
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    bounding_box(frame, object_id, scores, classes, boxes, num, 0.75, True)


def color(frame, object_id):
    frame_expanded = np.expand_dims(frame, axis=0)
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    bounding_box(frame, object_id, scores, classes, boxes, num, 0.75, False, True)


def focus(frame, object_id):
    frame_expanded = np.expand_dims(frame, axis=0)
    # Perform the actual detection by running the model with the image as input
    (boxes, scores, classes, num) = sess.run(
        [detection_boxes, detection_scores, detection_classes, num_detections],
        feed_dict={image_tensor: frame_expanded})

    bounding_box(frame, object_id, scores, classes, boxes, num, 0.75)


def no_operation():
    return 0


def bounding_box(image, object_id, scores, classes, boxes, num, min_score_thresh, print_count=False,
                 display_colour=False):
    object_count = 0
    scores_arr = np.squeeze(scores)
    classes_arr = np.squeeze(classes).astype(np.int32)
    boxes_arr = np.squeeze(boxes)
    image_pil = Image.fromarray(image)
    im_width, im_height = image_pil.size
    avg_color = (0, 0, 0)
    for i in range(int(num[0])):
        if scores_arr[i] > min_score_thresh:
            if category_index[classes_arr[i]]['id'] == object_id:
                object_count += 1
                vis_util.draw_bounding_box_on_image(image_pil,
                                                    boxes_arr[i, 0],
                                                    boxes_arr[i, 1],
                                                    boxes_arr[i, 2],
                                                    boxes_arr[i, 3],
                                                    display_str_list=[category_index[classes_arr[i]]['name']])
                if display_colour:
                    (ymin, xmin, ymax, xmax) = (boxes_arr[i, 0], boxes_arr[i, 1], boxes_arr[i, 2], boxes_arr[i, 3])
                    (left, right, top, bottom) = (
                        int(xmin * im_width), int(xmax * im_width), int(ymin * im_height), int(ymax * im_height))
                    # returns [blue, green, red] i
                    avg_color = np.average(np.average(image[top:bottom, left:right], axis=0), axis=0)

    if print_count:
        draw = ImageDraw.Draw(image_pil)
        draw.text((0, 0), "# Objects : " + str(object_count))

    np.copyto(image, np.array(image_pil))

    if display_colour:
        cv2.rectangle(image, (0, 0), (50, 50), avg_color, -1)
