from multiprocessing import Process, Queue

from camera import CameraFeed
from gestures_recognition import GestureEngine


def start_camera_feed(queue: Queue):
    cf = CameraFeed(camera_port=2, queue=queue)
    cf.start_camera()


def start_gesture_recognition():
    ge = GestureEngine()
    ge.start_prediction()


class ProcessManager:
    def __init__(self):
        self.procs = []

    def start_engines(self):
        com_queue = Queue()
        print("Starting Engines...")

        # engines = [start_camera_feed, start_gesture_recognition]
        engines = [start_camera_feed]
        for engine in engines:
            proc = Process(target=engine, args=(com_queue,))
            self.procs.append(proc)
            proc.start()

        print("Waiting for Engines...")
        for proc in self.procs:
            try:
                proc.join()
            except KeyboardInterrupt:
                proc.terminate()
                proc.join()

        print("Stopping Engines...")

    def stop_engines(self):
        # complete the processes
        for proc in self.procs:
            # proc.kill()
            proc.join()
