import os
import time
import cv2
import numpy as np
import tensorflow as tf

import core.utils as utils
from core.config import cfg
from utils import label_map_util


class VisionEngine:
    def __init__(self):
        # define paths to load the models
        self.PATH_TO_FRCNN_CKPT = os.path.join('data', 'models', 'ssd_inception.pb')
        self.PATH_TO_YOLO_CKPT = os.path.join('data', 'models', 'yolo_v3.pb')
        self.PATH_TO_LABELS_TFOD_API = os.path.join('data', 'classes', 'labels.pbtxt')
        # define constants
        self.NUM_CLASSES = 10
        self.INPUT_SIZE = 608
        # load the label map
        self.category_index = label_map_util.create_category_index_from_labelmap(self.PATH_TO_LABELS_TFOD_API,
                                                                                 use_display_name=True)
        self.class_names = utils.read_class_names(cfg.YOLO.CLASSES)

        # should be removed later by changing the classes order in yolo
        self.yolo_mapping = {1: 6, 2: 4, 3: 0, 4: 3, 5: 7, 6: 9, 7: 5, 8: 8, 9: 1, 10: 2}

        # Load the models into session
        self.detection_graph = tf.Graph()
        self.graph_def = tf.GraphDef()

        with self.detection_graph.as_default():
            with tf.gfile.GFile(self.PATH_TO_FRCNN_CKPT, 'rb') as fid:
                od_graph_def = tf.GraphDef()
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

            with tf.gfile.GFile(self.PATH_TO_YOLO_CKPT, 'rb') as fid:
                od_graph_def = tf.GraphDef()
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(per_process_gpu_memory_fraction=0.9))
                               , graph=self.detection_graph)
        self.yolo_tensors = self.get_tensors(tensor_names=["input/input_data:0",
                                                           "pred_sbbox/concat_2:0",
                                                           "pred_mbbox/concat_2:0",
                                                           "pred_lbbox/concat_2:0"])
        self.frcnn_tensors = self.get_tensors(tensor_names=["image_tensor:0",
                                                            "detection_boxes:0",
                                                            "detection_scores:0",
                                                            "detection_classes:0",
                                                            "num_detections:0"])

    def get_tensors(self, tensor_names):
        return [self.detection_graph.get_tensor_by_name(n) for n in tensor_names]

    def get_yolo_prediction(self, image, object_id=None):
        image_data = self.yolo_preporcess(image)
        image_data = np.expand_dims(image_data, axis=0)
        pred_sbbox, pred_mbbox, pred_lbbox = self.sess.run([
            self.yolo_tensors[1],
            self.yolo_tensors[2],
            self.yolo_tensors[3]
        ], feed_dict={self.yolo_tensors[0]: image_data})
        return self.yolo_bboxes(pred_sbbox, pred_mbbox, pred_lbbox, image.shape[:2], object_id)

    def get_frcnn_prediction(self, image, object_id=None):
        image_expanded = np.expand_dims(image, axis=0)
        (boxes, scores, classes, num) = self.sess.run([
            self.frcnn_tensors[1],
            self.frcnn_tensors[2],
            self.frcnn_tensors[3],
            self.frcnn_tensors[4]
        ], feed_dict={self.frcnn_tensors[0]: image_expanded})
        if object_id:
            return self.frcnn_bboxes_filter(image, scores, classes, boxes, num, 0.75, object_id)
        return self.frcnn_bboxes(image, scores, classes, boxes, num, 0.75)

    def yolo_preporcess(self, image):
        h, w, _ = image.shape

        scale = min(self.INPUT_SIZE / w, self.INPUT_SIZE / h)
        nw, nh = int(scale * w), int(scale * h)
        image_resized = cv2.resize(image, (nw, nh))

        image_paded = np.full(shape=[self.INPUT_SIZE, self.INPUT_SIZE, 3], fill_value=128.0)
        dw, dh = (self.INPUT_SIZE - nw) // 2, (self.INPUT_SIZE - nh) // 2
        image_paded[dh:nh + dh, dw:nw + dw, :] = image_resized
        return image_paded / 255.

    def yolo_bboxes(self, pred_sbbox, pred_mbbox, pred_lbbox, frame_size, object_id):
        pred_bbox = np.concatenate([np.reshape(pred_sbbox, (-1, 5 + self.NUM_CLASSES)),
                                    np.reshape(pred_mbbox, (-1, 5 + self.NUM_CLASSES)),
                                    np.reshape(pred_lbbox, (-1, 5 + self.NUM_CLASSES))], axis=0)

        bboxes = utils.postprocess_boxes(pred_bbox, frame_size, self.INPUT_SIZE, 0.3)
        if object_id:
            return utils.nms_pointing(bboxes, 0.45, method='nms', object_id=object_id)
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

    def draw_bbox(self, image, bboxes, show_label=True):
        """
        bboxes: [x_min, y_min, x_max, y_max, probability, cls_id] format coordinates.
        """

        image_h, image_w, _ = image.shape

        for i, bbox in enumerate(bboxes):
            coordinates = np.array(bbox[:4], dtype=np.int32)
            fontScale = 0.5
            score = bbox[4]
            class_ind = int(bbox[5])
            bbox_thick = int(0.6 * (image_h + image_w) / 600)
            c1, c2 = (coordinates[0], coordinates[1]), (coordinates[2], coordinates[3])
            cv2.rectangle(image, c1, c2, (255, 0, 0), bbox_thick)

            if show_label:
                bbox_mess = '%s: %.2f' % (self.class_names[class_ind], score)
                t_size = cv2.getTextSize(bbox_mess, 0, fontScale, thickness=bbox_thick // 2)[0]
                cv2.rectangle(image, c1, (c1[0] + t_size[0], c1[1] - t_size[1] - 3), (255, 0, 0), -1)  # filled

                cv2.putText(image, bbox_mess, (c1[0], c1[1] - 2), cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale, (0, 0, 0), bbox_thick // 2, lineType=cv2.LINE_AA)
        return image


def main():
    ve = VisionEngine()
    # image = cv2.imread("test/2019-11-06 18_39_33.905298.jpg")
    #
    # yolo_bboxes = ve.get_yolo_prediction(image)
    # rcnn_bboxes = ve.get_frcnn_prediction(image)
    #
    # yolo_result = ve.draw_bbox(np.copy(image), yolo_bboxes)
    # rcnn_result = ve.draw_bbox(np.copy(image), rcnn_bboxes)
    # cv2.imwrite("predicted_yolo.jpg", yolo_result)
    # cv2.imwrite("predicted_rcnn.jpg", rcnn_result)

    cap = cv2.VideoCapture(0)
    cap.set(3, 608)
    cap.set(4, 608)
    # pos_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    while True:
        flag, frame = cap.read()
        if flag:
            prev_time = time.time()

            bboxes = ve.get_yolo_prediction(frame, object_id=3)
            index = None
            d_prev = 1000000
            hand, hand_coor = None, None
            for i, bbox in enumerate(bboxes):
                if bbox[5] == 1:
                    hand = bbox[:4]
                    hand_coor = (int(0.125 * hand[2] + 0.875 * hand[0]), int(0.125 * hand[3] + 0.875 * hand[1]))
                    continue
                if hand is not None:
                    obj = 0.5 * bbox[:4]
                    obj_coor = (int(obj[2] + obj[0]), int(obj[3] + obj[1]))
                    cv2.line(frame, hand_coor, obj_coor, (44, 62, 80), 2)
                    d = np.square(hand_coor[0] - obj_coor[0]) + np.square(hand_coor[1] - obj_coor[1])
                    if d_prev > d:
                        d_prev = d
                        index = i
            if index:
                ve.draw_bbox(frame, [bboxes[index]])

            curr_time = time.time()
            exec_time = curr_time - prev_time
            print("time: %.2f FPS" % (1 / exec_time))
            cv2.imshow('video', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        # if cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(cv2.CAP_PROP_FRAME_COUNT):
        #     If the number of captured frames is equal to the total number of frames,
        #     we stop
            # break

    # while True:
    #     ret, frame = vid.read()
    #     # prev_time = time.time()
    #     #
    #     # bboxes = ve.get_frcnn_prediction(frame)
    #     # ve.draw_bbox(frame, bboxes)
    #     #
    #     # curr_time = time.time()
    #     # exec_time = curr_time - prev_time
    #     # print("time: %.2f FPS" % (1 / exec_time))
    #     if vid.get(cv2.CAP_PROP_POS_FRAMES) == vid.get(cv2.CAP_PROP_FRAME_COUNT):
    #         # If the number of captured frames is equal to the total number of frames,
    #         # we stop
    #         print("stop")
    #         break
    #
    #     cv2.imshow("Object Detector", frame)
    #
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break
    #
    # # Clean up
    # vid.release()
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
