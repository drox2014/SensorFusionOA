import _thread
import queue
import visualizer
import cv2
import numpy as np
from multiprocessing import Queue

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
        from object_detection import VisionEngine
        self.__image_oqueue = queue.Queue(5)
        self.__last_operation = None
        self.__queue = _queue
        self.__vision_engine = VisionEngine()
        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
        self.__is_zoomed = False
        self.__selection_timer = Timer(50)

        # Initialize webcam feed
        self.capture = cv2.VideoCapture(2)
        self.capture.set(3, 608)
        self.capture.set(4, 608)

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

                    if len(bboxes) == 0:
                        ''' No objects identified with SSD. Change the detecion algorithm to yolo'''
                        self.__default_object_detector = self.__vision_engine.get_yolo_prediction
                        bboxes = self.search_objects(self.__last_operation["object_id"])
                        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction

                    # Compare the sizes of found objects and given speech command
                    if len(bboxes) == 0:
                        ''' No objects identified with both detectors'''
                        print("Couldn't find any object...")
                    elif len(bboxes) == 1:
                        ''' More than one object identified '''
                        if self.__last_operation["multiple"]:
                            ''' Speech command was given to identify multiple objects'''
                            self.track_objects(bboxes, image, "Only one object found...")
                        else:
                            ''' Speech command was given to identify only one object'''
                            self.track_objects(bboxes, image, "we found your object...")
                    else:
                        if self.__last_operation["multiple"]:
                            ''' Speech command was given to identify multiple objects'''
                            self.track_objects(bboxes, image, "Objects found...")
                        else:
                            ''' Speech command was given to identify only one object'''
                            self.track_objects(bboxes, image, "More than one object found...")

                elif self.__last_operation["operation"] == "Describe":
                    self.__image_oqueue.put(image)

                    # Find the objects for given object id with SSD
                    bboxes = self.search_objects(self.__last_operation["object_id"])

                    if len(bboxes) == 0:
                        ''' No objects identified with SSD. Change the detecion algorithm to yolo'''
                        self.__default_object_detector = self.__vision_engine.get_yolo_prediction
                        bboxes = self.search_objects(self.__last_operation["object_id"])
                        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction

                    if self.__last_operation["pointing"]:
                        ''' Waiting for object selection'''

                        if len(bboxes) == 0:
                            ''' No objects identified with current detector. Change the object detection algorithm to yolo
                              and verify the existence of objects'''
                            print("Couldn't find any object...")
                        elif len(bboxes) == 1:
                            '''No need of pointing since only one object was found'''
                            self.track_objects(bboxes, image, "Object found...", True)
                        else:
                            '''Pointing should be done to identify the object'''
                            object_bbox = self.get_selection(self.__last_operation["object_id"])

                            '''Tracking the object'''
                            if object_bbox is not None:
                                self.track_objects([object_bbox], image, "Object has been selected...", True)

                        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
                        self.__last_operation = None
                    else:
                        if len(bboxes) == 0:
                            ''' No objects identified with current detector. Change the object detection algorithm to yolo
                            and verify the existence of objects'''
                            pass
                        elif len(bboxes) == 1:
                            ''' More than one object identified '''
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                self.track_objects(bboxes, image, "Only one object found...", True)
                            else:
                                ''' Speech command was given to identify only one object'''
                                self.track_objects(bboxes, image, "Object found...", True)
                        else:
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                self.track_objects(bboxes, image, "Objects found...", True)
                            else:
                                ''' Speech command was given to identify only one object, pointing is required'''
                                '''Pointing should be done to identify the object'''
                                object_bbox = self.get_selection(self.__last_operation["object_id"])

                                '''Tracking the object'''
                                if object_bbox is not None:
                                    self.track_objects([object_bbox], image, "Object has been selected...", True)

                elif self.__last_operation["operation"] == "ZoomIn":
                    self.__is_zoomed = True
                elif self.__last_operation["operation"] == "ZoomOut":
                    self.__is_zoomed = False
                self.__image_oqueue.put(image)
                self.__last_operation = None

            except KeyboardInterrupt:
                break
        self.capture.release()

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

    def get_image(self):
        ret, frame = self.capture.read()
        if self.__is_zoomed:
            # get the webcam size
            height, width, channels = frame.shape

            frame = frame[50:height - 50, 50:width - 50]
            frame = cv2.resize(frame, (width, height))
        return frame

    def track_objects(self, bboxes, image, message, overlay=False):
        trackers = cv2.MultiTracker_create()
        for bbox in bboxes:
            bbox = [int(r) for r in bbox[:4]]
            rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
            tracker = cv2.TrackerKCF_create()
            trackers.add(tracker, image, rect)

        while self.__selection_timer.is_running():
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
            self.__selection_timer.count()

        self.__selection_timer.reset()

    def search_objects(self, object_id):
        bboxes = None
        while self.__selection_timer.is_running():
            print("Searching for objects %d" % self.__selection_timer.count())
            image = self.get_image()
            bboxes = self.__default_object_detector(image, object_id)
            cv2.putText(image, "Searching...", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            self.__vision_engine.draw_bbox(image, bboxes)
            self.__image_oqueue.put(image)
        self.__selection_timer.reset()
        return bboxes

    def get_selection(self, object_id):
        object_bbox = None
        while self.__selection_timer.is_running():
            print("Pointing function %d" % self.__selection_timer.count())
            image = self.get_image()
            bbox = self.point_out(image, object_id)
            cv2.putText(image, "Point out the object...", (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)
            if bbox is not None:
                object_bbox = bbox
            self.__image_oqueue.put(image)
        self.__selection_timer.reset()
        return object_bbox


if __name__ == '__main__':
    com_queue = Queue()
    FusionEngine(com_queue)
