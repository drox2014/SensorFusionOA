from __future__ import print_function

from process_control import ProcessManager

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

ProcessManager().start_engines()
