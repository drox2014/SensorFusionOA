import _thread
import queue
import visualizer
import cv2
import numpy as np
from multiprocessing import Queue
from camera_feed import CameraFeed


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
        self.__image_iqueue = queue.Queue(5)
        self.__image_oqueue = queue.Queue(5)
        self.__last_operation = None
        self.__queue = _queue
        self.__vision_engine = VisionEngine()
        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
        # self.__camera_feed = CameraFeed(0)
        self.__selection_timer = Timer(20)

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
                    bboxes = None
                    self.__image_oqueue.put(image)
                    while self.__selection_timer.is_running():
                        print("Searching for objects %d" % self.__selection_timer.count())
                        image = self.get_image()
                        bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                        self.__vision_engine.draw_bbox(image, bboxes)
                        self.__image_oqueue.put(image)
                    self.__selection_timer.reset()

                    if len(bboxes) == 0:
                        ''' No objects identified with SSD. Change the detecion algorithm to yolo'''
                        self.__default_object_detector = self.__vision_engine.get_yolo_prediction
                        while self.__selection_timer.is_running():
                            print("Searching for objects %d" % self.__selection_timer.count())
                            image = self.get_image()
                            bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                            self.__vision_engine.draw_bbox(image, bboxes)
                            self.__image_oqueue.put(image)
                        self.__selection_timer.reset()

                    # Compare the sizes of found objects and given speech command
                    if len(bboxes) == 0:
                        ''' No objects identified with both detectors'''
                        print("Couldn't find any object...")
                    elif len(bboxes) == 1:
                        ''' More than one object identified '''
                        if self.__last_operation["multiple"]:
                            ''' Speech command was given to identify multiple objects'''
                            print("Only one object found...")
                        else:
                            ''' Speech command was given to identify only one object'''
                            print("we found your object...")

                        while self.__selection_timer.is_running():
                            print("Tracking objects %d" % self.__selection_timer.count())
                            image = self.get_image()
                            bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                            self.__vision_engine.draw_bbox(image, bboxes)
                            self.__image_oqueue.put(image)
                        self.__selection_timer.reset()

                    else:
                        if self.__last_operation["multiple"]:
                            ''' Speech command was given to identify multiple objects'''
                            print("We found your objects...")
                        else:
                            ''' Speech command was given to identify only one object'''
                            print("More than one object found...")

                        while self.__selection_timer.is_running():
                            print("Tracking objects %d" % self.__selection_timer.count())
                            image = self.get_image()
                            bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                            self.__vision_engine.draw_bbox(image, bboxes)
                            self.__image_oqueue.put(image)
                        self.__selection_timer.reset()

                    self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
                    self.__last_operation = None
                elif self.__last_operation["operation"] == "Describe":
                    if self.__last_operation["pointing"]:
                        ''' Waiting for object selection'''

                        # Find the objects for given object id with SSD
                        bboxes = None
                        self.__image_oqueue.put(image)
                        while self.__selection_timer.is_running():
                            print("Searching for objects %d" % self.__selection_timer.count())
                            image = self.get_image()
                            bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                            self.__vision_engine.draw_bbox(image, bboxes)
                            self.__image_oqueue.put(image)
                        self.__selection_timer.reset()

                        if len(bboxes) == 0:
                            ''' No objects identified with SSD. Change the detection algorithm to yolo'''
                            self.__default_object_detector = self.__vision_engine.get_yolo_prediction
                            while self.__selection_timer.is_running():
                                print("Searching for objects %d" % self.__selection_timer.count())
                                image = self.get_image()
                                bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                                self.__vision_engine.draw_bbox(image, bboxes)
                                self.__image_oqueue.put(image)
                            self.__selection_timer.reset()

                        if len(bboxes) == 0:
                            ''' No objects identified with current detector. Change the object detection algorithm to yolo
                              and verify the existence of objects'''
                            print("Couldn't find any object...")
                        elif len(bboxes) == 1:
                            '''No need of pointing since only one object was found'''
                            self.__vision_engine.draw_bbox(image, bboxes)
                            self.__image_oqueue.put(image)
                            while self.__selection_timer.is_running():
                                image = self.get_image()
                                bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                                image = self.__vision_engine.overlay(image, self.__last_operation["object_id"])
                                self.__vision_engine.draw_bbox(image, bboxes)
                                self.__image_oqueue.put(image)
                                self.__selection_timer.count()
                            self.__selection_timer.reset()
                        else:
                            '''Pointing should be done to identify the object'''
                            self.__image_oqueue.put(image)
                            object_bbox = None
                            while self.__selection_timer.is_running():
                                print("Pointing function %d" % self.__selection_timer.count())
                                image = self.get_image()
                                bbox = self.point_out(image, self.__last_operation["object_id"])
                                if bbox is not None:
                                    object_bbox = bbox
                                self.__image_oqueue.put(image)
                            self.__selection_timer.reset()

                            '''Tracking the object'''
                            if len(bboxes) is not None:
                                print("Object has been selected...")
                                bbox = [int(r) for r in object_bbox[:4]]
                                rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                                self.track_object(image, rect)

                        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
                        self.__last_operation = None
                    else:
                        # Find the objects for given object id with SSD
                        bboxes = None
                        self.__image_oqueue.put(image)
                        while self.__selection_timer.is_running():
                            print("Searching for objects %d" % self.__selection_timer.count())
                            image = self.get_image()
                            bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                            self.__vision_engine.draw_bbox(image, bboxes)
                            self.__image_oqueue.put(image)
                        self.__selection_timer.reset()

                        if len(bboxes) == 0:
                            ''' No objects identified with SSD. Change the detection algorithm to yolo'''
                            self.__default_object_detector = self.__vision_engine.get_yolo_prediction
                            while self.__selection_timer.is_running():
                                print("Searching for objects %d" % self.__selection_timer.count())
                                image = self.get_image()
                                bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                                self.__vision_engine.draw_bbox(image, bboxes)
                                self.__image_oqueue.put(image)
                            self.__selection_timer.reset()

                        if len(bboxes) == 0:
                            ''' No objects identified with current detector. Change the object detection algorithm to yolo
                            and verify the existence of objects'''
                            pass
                        elif len(bboxes) == 1:
                            ''' More than one object identified '''
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                print("Only one object found...")
                            else:
                                ''' Speech command was given to identify only one object'''
                                print("we found your object...")

                            while self.__selection_timer.is_running():
                                print("Tracking objects %d" % self.__selection_timer.count())
                                image = self.get_image()
                                bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                                image = self.__vision_engine.overlay(image, self.__last_operation["object_id"])
                                self.__vision_engine.draw_bbox(image, bboxes)
                                self.__image_oqueue.put(image)
                            self.__selection_timer.reset()

                        else:
                            if self.__last_operation["multiple"]:
                                ''' Speech command was given to identify multiple objects'''
                                print("We found your objects...")
                                while self.__selection_timer.is_running():
                                    print("Tracking objects %d" % self.__selection_timer.count())
                                    image = self.get_image()
                                    bboxes = self.__default_object_detector(image, self.__last_operation["object_id"])
                                    image = self.__vision_engine.overlay(image, self.__last_operation["object_id"])
                                    self.__vision_engine.draw_bbox(image, bboxes)
                                    self.__image_oqueue.put(image)
                                self.__selection_timer.reset()
                            else:
                                ''' Speech command was given to identify only one object, pointing is required'''
                                print("Pointing is required")
                                object_bbox = None
                                while self.__selection_timer.is_running():
                                    print("Pointing function %d" % self.__selection_timer.count())
                                    image = self.get_image()
                                    bbox = self.point_out(image, self.__last_operation["object_id"])
                                    if bbox is not None:
                                        object_bbox = bbox
                                    self.__image_oqueue.put(image)
                                self.__selection_timer.reset()

                                '''Tracking the object'''
                                if object_bbox is not None:
                                    print("Object has been selected...")
                                    bbox = [int(r) for r in object_bbox[:4]]
                                    rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                                    self.track_object(image, rect)

                        self.__default_object_detector = self.__vision_engine.get_frcnn_prediction
                        self.__last_operation = None
                self.__image_oqueue.put(image)
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

    def image_enqueue(self, image):
        self.__image_iqueue.put(image)

    def image_dequeue(self):
        return self.__image_oqueue.get()

    def image_is_none(self):
        return self.__image_oqueue.empty()

    def enqueue_command(self, command):
        self.__queue.put(command)

    def get_image(self):
        ret, frame = self.capture.read()
        return frame

    def track_object(self, image, rect):
        tracker = cv2.TrackerKCF_create()
        tracker.init(image, rect)

        while self.__selection_timer.is_running():
            image = self.get_image()
            res, bbox = tracker.update(image)
            image = self.__vision_engine.overlay(image, self.__last_operation["object_id"])
            if res:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(image, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(image,
                            "Tracking Failed",
                            (100, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.75,
                            (0, 0, 255),
                            2)
            self.__image_oqueue.put(image)
            self.__selection_timer.count()

        self.__selection_timer.reset()


if __name__ == '__main__':
    com_queue = Queue()
    FusionEngine(com_queue)
