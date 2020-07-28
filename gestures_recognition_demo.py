import signal
import time
from multiprocessing import Queue

import numpy as np

from utils.logger import Logger


class GestureEngine:
    def __init__(self, queue: Queue):
        self.command_classes = ['Pointing', 'Capture', 'ZoomIn', 'ZoomOut', 'Roaming']
        self.queue = queue
        self.__logger = Logger("gesture")

    def run(self, model):
        from config import config
        gesture_sequence = np.array([])
        while True:
            feature = np.random.rand(1, 9)
            gesture_sequence = np.append(gesture_sequence, feature)
            if gesture_sequence.shape[0] > 270:
                self.__logger.start()
                if config.GR == 1:
                    prediction = model.predict(gesture_sequence[:270].reshape(1, 1, 270))
                    gesture = np.argmax(prediction)
                else:
                    prediction = model.predict([gesture_sequence[:270]])
                    gesture = int(prediction[0])
                # print("Gesture:", self.command_classes[gesture])
                self.__logger.checkpoint("%s" % self.command_classes[gesture])
                gesture_sequence = gesture_sequence[90:]
            time.sleep(0.006)

    def start_prediction(self):
        import tensorflow as tf
        import pickle
        import os
        import psutil
        from config import config
        process = psutil.Process(os.getpid())

        start = process.memory_info()[0]
        if config.GR == 1:
            # Initializing the model
            config = tf.ConfigProto(intra_op_parallelism_threads=4,
                                    inter_op_parallelism_threads=4,
                                    allow_soft_placement=True,
                                    device_count={'CPU': 2, 'GPU': 0})
            session = tf.Session(config=config)
            tf.keras.backend.set_session(session)
            model = tf.keras.models.load_model("./data/models/gesture_lstm_v9.h5")
        else:
            model = pickle.load(open("data/models/gesture_recognition_svm.pkl", "rb"))
        usage = process.memory_info()[0] - start
        print("[Memory Usage | Gesture Recognition]", usage >> 20)

        try:
            self.run(model)
        except KeyboardInterrupt:
            self.__logger.save()
            self.__logger.close()
            print("GestureEngine:KeyboardInterrupt")
            os.kill(os.getpid(), signal.SIGKILL)


def main():
    ge = GestureEngine(queue=Queue())
    ge.start_prediction()


if __name__ == '__main__':
    main()
