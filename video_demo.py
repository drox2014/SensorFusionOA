#! /usr/bin/env python
import time

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image
from core.config import cfg
import core.utils as utils

return_elements = ["input/input_data:0", "pred_sbbox/concat_2:0", "pred_mbbox/concat_2:0", "pred_lbbox/concat_2:0"]
pb_file = "data/models/yolov3_coco_v1.1.pb"
video_path = 0
num_classes = 10
input_size = 608
graph = tf.Graph()
return_tensors = utils.read_pb_return_tensors(graph, pb_file, return_elements)
# gpu_options = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
sess = tf.Session(graph=graph)
vid = cv2.VideoCapture(video_path)
vid.set(3, 608)
vid.set(4, 608)
class_names = utils.read_class_names(cfg.YOLO.CLASSES)

while True:
    ret, frame = vid.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # image = Image.fromarray(frame)
    frame_size = frame.shape[:2]
    image_data = utils.image_preporcess(np.copy(frame), [input_size, input_size])
    # image_data = image_data[np.newaxis, ...]
    image_data = np.expand_dims(image_data, axis=0)

    prev_time = time.time()

    pred_sbbox, pred_mbbox, pred_lbbox = sess.run(
        [return_tensors[1], return_tensors[2], return_tensors[3]],
        feed_dict={return_tensors[0]: image_data})

    pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_mbbox, (-1, 5 + num_classes)),
                                np.reshape(pred_lbbox, (-1, 5 + num_classes))], axis=0)

    bboxes = utils.postprocess_boxes(pred_bbox, frame_size, input_size, 0.3)
    bboxes = utils.nms(bboxes, 0.45, method='nms')
    image = utils.draw_bbox(frame, bboxes, classes=class_names)

    curr_time = time.time()
    exec_time = curr_time - prev_time
    result = np.asarray(image)
    print("FPS: %.2f" % (1 / exec_time))
    cv2.namedWindow("result", cv2.WINDOW_AUTOSIZE)
    result = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow("result", result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
