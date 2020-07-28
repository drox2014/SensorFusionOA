import time
from multiprocessing import Process, Queue

from gestures_recognition_demo import GestureEngine
from sensor_fusion import FusionEngine
from speech_recognition_demo import SpeechEngine


def start_gesture_recognition(queue: Queue):
    ge = GestureEngine(queue=queue)
    ge.start_prediction()


def start_speech_engine(queue: Queue):
    se = SpeechEngine(queue=queue)
    se.start_recognition()


def start_fusion_engine(queue: Queue):
    FusionEngine(_queue=queue)


class ProcessManager:
    def __init__(self):
        self.procs = []

    def start_engines(self):
        com_queue = Queue()
        print("Starting Engines...")

        engines = [start_fusion_engine, start_gesture_recognition, start_speech_engine]
        # engines = [start_fusion_engine, start_gesture_recognition]
        # engines = [start_fusion_engine, start_speech_engine]
        for engine in engines:
            proc = Process(target=engine, args=(com_queue,))
            self.procs.append(proc)
            proc.start()
            time.sleep(5)

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
            proc.join()
