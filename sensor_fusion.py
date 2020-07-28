import _thread
import queue
import visualizer
import cv2
import numpy as np
from multiprocessing import Queue
from utils.logger import Logger
from config import config


class Timer:
    def __init__(self, counter=100):
        self.__counter = counter
        self.__memory = counter

    def count(self):
        self.__counter -= 1
        return self.__counter

    def is_running(self):
        return self.__counter > 0

    def reset(self):
        self.__counter = self.__memory


class FusionEngine:
    def __init__(self, _queue: Queue):
        from object_detection_demo import VisionEngine
        self.__image_oqueue = queue.Queue(5)
        self.__last_operation = None
        self.__queue = _queue
        self.__vision_engine = VisionEngine()
        if config.VH == 1:
            self.__default_object_detector = self.__vision_engine.get_yolo_prediction
        else:
            self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
        self.__is_zoomed = False
        self.__selection_timer = Timer(5)
        self.__logger = Logger("frame")

        # Initialize webcam feed
        # self.capture = cv2.VideoCapture(0)
        # self.capture.set(3, 608)
        # self.capture.set(4, 608)

        # _thread.start_new_thread(self.__camera_feed.start_feed, (self,))
        _thread.start_new_thread(visualizer.stream, (self,))

        while True:
            try:
                image = self.get_image()
                if not self.__queue.empty():
                    self.__last_operation = self.__queue.get()

                if self.__last_operation is None:
                    self.__image_oqueue.put(image)
                    continue
                elif self.__last_operation["operation"] == "Locate":
                    '''Performing locating object - no mixing with gestures'''

                    # Find the objects for given object id with SSD
                    self.__image_oqueue.put(image)
                    bboxes = self.search_objects(self.__last_operation["object_id"])

                    # if len(bboxes) == 0:
                    #     ''' No objects identified with SSD. Change the detecion algorithm to yolo'''
                    #     self.__default_object_detector = self.__vision_engine.get_yolo_prediction
                    #     bboxes = self.search_objects(self.__last_operation["object_id"])
                    #     self.__default_object_detector = self.__vision_engine.get_frcnn_prediction

                    # Compare the sizes of found objects and given speech command
                    if len(bboxes) == 0:
                        ''' No objects identified with both detectors'''
                        self.show_message("No such object found...")
                    elif len(bboxes) == 1:
                        ''' More than one object identified '''
                        if self.__last_operation["multiple"]:
                            ''' Speech command was given to identify multiple objects'''
                            self.track_objects(bboxes, image, self.__last_operation["object_id"], "Only one object found...")
                        else:
                            ''' Speech command was given to identify only one object'''
                            self.track_objects(bboxes, image, self.__last_operation["object_id"], "we found your object...")
                    else:
                        if self.__last_operation["multiple"]:
                            ''' Speech command was given to identify multiple objects'''
                            self.track_objects(bboxes, image, self.__last_operation["object_id"], "Objects found...")
                        else:
                            ''' Speech command was given to identify only one object'''
                            self.track_objects(bboxes, image, self.__last_operation["object_id"], "More than one object found...")

                elif self.__last_operation["operation"] == "Describe":
                    self.__image_oqueue.put(image)

                    if self.__last_operation["pointing"]:
                        '''Pointing should be done to identify the object'''
                        # object_bbox = self.get_selection(self.__last_operation["object_id"])
                        #
                        # '''Tracking the object'''
                        # if object_bbox is not None:
                        #     self.track_objects([object_bbox], image, self.__last_operation["object_id"], "Object has been selected...", True)

                        # self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
                        self.__last_operation = None
                    else:
                        # Find the objects for given object id with SSD
                        bboxes = self.search_objects(self.__last_operation["object_id"])

                        # if len(bboxes) == 0:
                        #     ''' No objects identified with SSD. Change the detecion algorithm to yolo'''
                        #     self.__default_object_detector = self.__vision_engine.get_yolo_prediction
                        #     bboxes = self.search_objects(self.__last_operation["object_id"])
                        #     self.__default_object_detector = self.__vision_engine.get_frcnn_prediction

                        if len(bboxes) == 0:
                            ''' No objects identified with current detector. Change the object detection algorithm to yolo
                            and verify the existence of objects'''
                            self.show_message("No such object found...")
                        elif len(bboxes) == 1:
                            ''' More than one object identified '''
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                self.track_objects(bboxes, image, self.__last_operation["object_id"], "Only one object found...", True)
                            else:
                                ''' Speech command was given to identify only one object'''
                                self.track_objects(bboxes, image, self.__last_operation["object_id"], "Object found...", True)
                        else:
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                self.track_objects(bboxes, image, self.__last_operation["object_id"], "Objects found...", True)
                            else:
                                ''' Speech command was given to identify only one object, pointing is required'''
                                '''Pointing should be done to identify the object'''
                                # object_bbox = self.get_selection(self.__last_operation["object_id"])
                                #
                                # '''Tracking the object'''
                                # if object_bbox is not None:
                                #     self.track_objects([object_bbox], image, self.__last_operation["object_id"], "Object has been selected...", True)

                elif self.__last_operation["operation"] == "ZoomIn":
                    self.__is_zoomed = True
                elif self.__last_operation["operation"] == "ZoomOut":
                    self.__is_zoomed = False
                self.__image_oqueue.put(image)
                self.__last_operation = None

            except KeyboardInterrupt:
                self.__logger.close()
                break
        # self.capture.release()

    def point_out(self, image, object_id):
        bboxes = self.__vision_engine.get_yolo_prediction(image, object_id=object_id, pointing=True)
        index = None
        d_prev = 1000000
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
                # cv2.line(image, hand_coor, obj_coor, (44, 62, 80), 2)
                d = np.square(hand_coor[0] - obj_coor[0]) + np.square(hand_coor[1] - obj_coor[1])
                if d_prev > d:
                    d_prev = d
                    index = i
        if index:
            self.__vision_engine.draw_bbox(image, [bboxes[index]])
            return bboxes[index]
        return None

    def image_dequeue(self):
        return self.__image_oqueue.get()

    def image_is_none(self):
        return self.__image_oqueue.empty()

    def enqueue_command(self, command):
        self.__queue.put(command)

    def get_image(self, object_id=10):
        # ret, frame = self.capture.read()
        frame = cv2.imread("images/%d.jpg" % object_id)
        if self.__is_zoomed:
            # get the webcam size
            height, width, channels = frame.shape

            frame = frame[50:height - 50, 50:width - 50]
            frame = cv2.resize(frame, (width, height))
        return frame

    def track_objects(self, bboxes, image, object_id, message, overlay=False):
        self.__logger.add_flog("object_tracking")
        trackers = cv2.MultiTracker_create()
        for bbox in bboxes:
            bbox = [int(r) for r in bbox[:4]]
            rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            tracker = cv2.TrackerKCF_create()
            trackers.add(tracker, image, rect)

        while self.__selection_timer.is_running():
            self.__logger.start()
            image = self.get_image()
            success, bboxes = trackers.update(image)
            if overlay:
                image = self.__vision_engine.overlay(image, self.__last_operation["object_id"])
            if success:
                for bbox in bboxes:
                    p1 = (int(bbox[0]), int(bbox[1]))
                    p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                    cv2.putText(image, message, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
                    self.__vision_engine.draw_rect(image, p1, p2)
            else:
                cv2.putText(image,
                            "Tracking Failed",
                            (20, 20),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2)
            self.__image_oqueue.put(image)
            self.__logger.checkpoint("track for %d objects" % len(bboxes))
            self.__selection_timer.count()

        self.__selection_timer.reset()
        self.__logger.save()

    def search_objects(self, object_id):
        bboxes = None
        self.__logger.add_flog("object_detection")
        while self.__selection_timer.is_running():
            self.__selection_timer.count()
            self.__logger.start()
            image = self.get_image()
            bboxes = self.__default_object_detector(image, object_id)
            cv2.putText(image, "Searching...", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            self.__vision_engine.draw_bbox(image, bboxes)
            self.__image_oqueue.put(image)
            self.__logger.checkpoint("search for %d objects" % len(bboxes))
        self.__selection_timer.reset()
        self.__logger.save()
        return bboxes

    def get_selection(self, object_id):
        object_bbox = None
        while self.__selection_timer.is_running():
            self.__selection_timer.count()
            image = self.get_image()
            bbox = self.point_out(image, object_id)
            cv2.putText(image, "Point out the object...", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if bbox is not None:
                object_bbox = bbox
            self.__image_oqueue.put(image)
        self.__selection_timer.reset()
        return object_bbox

    def show_message(self, message):
        while self.__selection_timer.is_running():
            self.__selection_timer.count()
            image = self.get_image()
            cv2.putText(image, message, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            self.__image_oqueue.put(image)
        self.__selection_timer.reset()


if __name__ == '__main__':
    com_queue = Queue()
    FusionEngine(com_queue)
