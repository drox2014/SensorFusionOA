import _thread
import queue
import visualizer
import cv2
import numpy as np
from multiprocessing import Queue
from camera_feed import CameraFeed


class FusionEngine:
    def __init__(self, _queue: Queue):
        from object_detection import VisionEngine
        self.__image_iqueue = queue.Queue(3)
        self.__image_oqueue = queue.Queue(5)
        self.__last_operation = None
        self.__queue = _queue
        self.__vision_engine = VisionEngine()
        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
        self.__camera_feed = CameraFeed(2)
        self.__selection_frames = 75
        self.__operation_runtime_frames = 150

        _thread.start_new_thread(self.__camera_feed.start_feed, (self,))
        _thread.start_new_thread(visualizer.stream, (self,))

        while True:
            try:
                while self.__image_iqueue.qsize() > 3:
                    self.__image_iqueue.get()

                image = self.__image_iqueue.get()
                if not self.__queue.empty():
                    self.__last_operation = self.__queue.get()

                if self.__last_operation is None:
                    self.__image_oqueue.put(image)
                    continue
                elif self.__last_operation["operation"] == "Locate":
                    if self.__last_operation["pointing"]:
                        self.__image_oqueue.put(image)
                        while self.__selection_frames > 0:
                            print("Pointing function %d" % self.__selection_frames)
                            image = self.__image_iqueue.get()
                            self.point_out(image, self.__last_operation["object_id"])
                            self.__image_oqueue.put(image)
                            self.__selection_frames -= 1
                        self.__selection_frames = 75
                        self.__last_operation = None
                        continue
                    else:
                        bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                        if len(bboxes) == 0:
                            ''' No objects identified with current detector'''
                            pass
                        elif len(bboxes) == 1:
                            ''' More than one object identified '''
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                print("Ambiguous operation")
                            else:
                                ''' Speech command was given to identify only one object'''
                                self.__vision_engine.draw_bbox(image, bboxes)
                        else:
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                self.__vision_engine.draw_bbox(image, bboxes)
                            else:
                                ''' Speech command was given to identify only one object, pointing is required'''
                                print("Ambiguous operation, please point out the object...")

                self.__image_oqueue.put(image)
            except KeyboardInterrupt:
                break

    def point_out(self, image, object_id):
        bboxes = self.__vision_engine.get_yolo_prediction(image, object_id=object_id, pointing=True)
        index = None
        d_prev = 10000
        hand, hand_coor = None, None
        for i, bbox in enumerate(bboxes):
            if bbox[5] == 1:
                hand = bbox[:4]
                hand_coor = (int(0.125 * hand[2] + 0.875 * hand[0]),
                             int(0.125 * hand[3] + 0.875 * hand[1]))
                continue
            if hand is not None:
                obj = 0.5 * bbox[:4]
                obj_coor = (int(obj[2] + obj[0]), int(obj[3] + obj[1]))
                cv2.line(image, hand_coor, obj_coor, (44, 62, 80), 2)
                d = np.square(hand_coor[0] - obj_coor[0]) + np.square(hand_coor[1] - obj_coor[1])
                if d_prev > d:
                    d_prev = d
                    index = i
        if index:
            self.__vision_engine.draw_bbox(image, [bboxes[index]])

    def image_enqueue(self, image):
        self.__image_iqueue.put(image)

    def image_dequeue(self):
        return self.__image_oqueue.get()

    def image_is_none(self):
        return self.__image_oqueue.empty()
