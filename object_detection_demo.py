import os

import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg


def get_keywords():
    keywords = open("data/keywords")
    return keywords.read().splitlines()


class VisionEngine:
    def __init__(self):
        from config import config
        # define paths to load the models
        self.PATH_TO_FRCNN_CKPT = os.path.join('data', 'models', 'ssd_inception_v7.pb')
        self.PATH_TO_YOLO_CKPT = os.path.join('data', 'models', 'yolo_v3.pb')
        self.PATH_TO_LABELS_TFOD_API = os.path.join('data', 'classes', 'labels.pbtxt')
        # define constants
        self.NUM_CLASSES = 10
        self.INPUT_SIZE = 608
        self.VH = config.VH
        # load the label map
        # self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS_TFOD_API,
        #                                                                          use_display_name=True)
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)
        self.keywords = get_keywords()
        # should be removed later by changing the classes order in yolo
        self.yolo_mapping = {1: 6, 2: 4, 3: 0, 4: 3, 5: 7, 6: 9, 7: 5, 8: 8, 9: 1, 10: 2}

        self.background = cv2.imread("data/overlay-ar.png")
        self.background = cv2.resize(self.background, (672, 504))
        self.primary_color = (60, 76, 231)

        # Load the models into session
        self.detection_graph = tf.Graph()
        self.graph_def = tf.GraphDef()

        with self.detection_graph.as_default():
            if self.VH == 1:
                with tf.gfile.GFile(self.PATH_TO_YOLO_CKPT, 'rb') as fid:
                    od_graph_def = tf.GraphDef()
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')
            else:
                with tf.gfile.GFile(self.PATH_TO_FRCNN_CKPT, 'rb') as fid:
                    od_graph_def = tf.GraphDef()
                    serialized_graph = fid.read()
                    od_graph_def.ParseFromString(serialized_graph)
                    tf.import_graph_def(od_graph_def, name='')

        config = tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(config=config, graph=self.detection_graph)
        if self.VH == 1:
            self.yolo_tensors = self.get_tensors(tensor_names=["input/input_data:0",
                                                               "pred_sbbox/concat_2:0",
                                                               "pred_mbbox/concat_2:0",
                                                               "pred_lbbox/concat_2:0"])
        else:
            self.frcnn_tensors = self.get_tensors(tensor_names=["image_tensor:0",
                                                                "detection_boxes:0",
                                                                "detection_scores:0",
                                                                "detection_classes:0",
                                                                "num_detections:0"])

    def get_tensors(self, tensor_names):
        return [self.detection_graph.get_tensor_by_name(n) for n in tensor_names]

    def get_yolo_prediction(self, image, object_id=None, pointing=False):
        image_data = self.yolo_preporcess(image)
        image_data = np.expand_dims(image_data, axis=0)
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run([
            self.yolo_tensors[1],
            self.yolo_tensors[2],
            self.yolo_tensors[3]
        ], feed_dict={self.yolo_tensors[0]: image_data})
        return self.yolo_bboxes(pred_sbbox, pred_mbbox, pred_lbbox, image.shape[:2], object_id, pointing)

    def get_frcnn_prediction(self, image, object_id=None):
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run([
            self.frcnn_tensors[1],
            self.frcnn_tensors[2],
            self.frcnn_tensors[3],
            self.frcnn_tensors[4]
        ], feed_dict={self.frcnn_tensors[0]: image_expanded})
        if object_id:
            return self.frcnn_bboxes_filter(image, scores, classes, boxes, num, 0.45, object_id)
        return self.frcnn_bboxes(image, scores, classes, boxes, num, 0.45)

    def yolo_preporcess(self, image):
        h, w, _ = image.shape

        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[self.INPUT_SIZE, self.INPUT_SIZE, 3], fill_value=128.0)
        dw, dh = (self.INPUT_SIZE - nw) // 2, (self.INPUT_SIZE - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        return image_paded / 255.

    def yolo_bboxes(self, pred_sbbox, pred_mbbox, pred_lbbox, frame_size, object_id, pointing):
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.NUM_CLASSES)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.NUM_CLASSES)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.NUM_CLASSES))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self.INPUT_SIZE, 0.3)
        if pointing:
            return utils.nms_pointing(bboxes, 0.45, method='nms', object_id=object_id)
        elif object_id:
            return utils.nms_filter(bboxes, 0.45, method='nms', object_id=object_id)
        return utils.nms(bboxes, 0.45, method='nms')

    def frcnn_bboxes(self, image, scores, classes, boxes, num, min_score_thresh):
        image_h, image_w, _ = image.shape
        scores_arr = np.squeeze(scores)
        classes_arr = np.squeeze(classes).astype(np.int32)
        boxes_arr = np.squeeze(boxes)
        bboxes = []
        for i in range(int(num[0])):
            if scores_arr[i] > min_score_thresh:
                b = [boxes_arr[i, 1] * image_w, boxes_arr[i, 0] * image_h, boxes_arr[i, 3] * image_w,
                     boxes_arr[i, 2] * image_h]
                bboxes.append(np.concatenate((b, scores_arr[i], self.yolo_mapping[classes_arr[i]]), axis=None))
        return bboxes

    def frcnn_bboxes_filter(self, image, scores, classes, boxes, num, min_score_thresh, object_id):
        image_h, image_w, _ = image.shape
        scores_arr = np.squeeze(scores)
        classes_arr = np.squeeze(classes).astype(np.int32)
        boxes_arr = np.squeeze(boxes)
        bboxes = []
        for i in range(int(num[0])):
            if scores_arr[i] > min_score_thresh and self.yolo_mapping[classes_arr[i]] == object_id:
                b = [boxes_arr[i, 1] * image_w, boxes_arr[i, 0] * image_h, boxes_arr[i, 3] * image_w,
                     boxes_arr[i, 2] * image_h]
                bboxes.append(np.concatenate((b, scores_arr[i], self.yolo_mapping[classes_arr[i]]), axis=None))
        return bboxes

    def draw_bounding_box(self, image, ymin, xmin, ymax, xmax):
        image_h, image_w, _ = image.shape
        c1, c2 = (int(xmin * image_w), int(ymin * image_h)), (int(xmax * image_w), int(ymax * image_h))
        cv2.rectangle(image, c1, c2, (255, 255, 0), 2)

    def draw_rect(self, frame, pt1, pt2):
        cv2.rectangle(frame, pt1, pt2, (185, 128, 41), 2)
        cv2.line(frame, (pt1[0] + 20, pt1[1]), (pt2[0] - 20, pt1[1]), (80, 62, 44), 1)
        cv2.line(frame, (pt1[0] + 20, pt2[1]), (pt2[0] - 20, pt2[1]), (80, 62, 44), 1)
        cv2.line(frame, (pt1[0], pt1[1] + 20), (pt1[0], pt2[1] - 20), (80, 62, 44), 1)
        cv2.line(frame, (pt2[0], pt1[1] + 20), (pt2[0], pt2[1] - 20), (80, 62, 44), 1)

    def overlay(self, frame, object_id):
        frame = cv2.addWeighted(frame, 1, self.background, 0.6, 0)
        cv2.putText(frame, "Object: %s" % self.class_names[object_id], (20, 395), cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    self.primary_color, 1)
        cv2.putText(frame, "Key words: %s" % self.keywords[object_id], (45, 420),
                    cv2.FONT_HERSHEY_DUPLEX,
                    0.5, self.primary_color, 1)
        return frame

    def draw_bbox(self, image, bboxes, show_label=True):
        """
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        """

        for i, bbox in enumerate(bboxes):
            coordinates = np.array(bbox[:4], dtype=np.int32)
            c1, c2 = (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])
            self.draw_rect(image, c1, c2)
