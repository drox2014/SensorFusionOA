from multiprocessing import Process

from DoubleFeatureLSTM import GestureEngine


def start_gesture_engine():
    GestureEngine()


class ProcessManager:
    def __init__(self):
        self.procs = []

    def start_engines(self):
        print('Starting process for gesture engine...')
        proc = Process(target=start_gesture_engine)
        self.procs.append(proc)
        print('Started process for gesture engine...')
        proc.start()

    def stop_engines(self):
        # complete the processes
        for proc in self.procs:
            # proc.kill()
            proc.join()
