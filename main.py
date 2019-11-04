from __future__ import print_function
import ray
from gestures_recognition import GestureEngine
from process_control import ProcessManager
import logging

# import _thread

# import numpy as np
# from multiprocessing import Process
from camera import CameraFeed

# from operations import find_op

# object_classes = ['laptop', 'phone', 'book', 'bottle', 'pen', 'cup']
#
# cf = CameraFeed(camera_port=0)
# _thread.start_new_thread(run_engine, (cf,))
# cf.start_camera()

# ProcessManager().start_engines()

# ray.init(num_gpus=1, num_cpus=2, memory= 1024 * 1024 * 1024 * 10, object_store_memory=1024 * 1024 * 1024 * 2)
# ray.init(logging_level=logging.DEBUG)
ray.init()


@ray.remote
def start_gesture_recognition():
    ge = GestureEngine()
    ge.start_prediction()


start_gesture_recognition.remote()
