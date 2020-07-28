import os
import signal
import time
from multiprocessing import Queue

import numpy as np

import Leap


class GestureEngine:
    def __init__(self, queue: Queue):
        self.command_classes = ['Pointing', 'Capture', 'ZoomIn', 'ZoomOut', 'Roaming']
        self.queue = queue
        self.prev_gesture = -1

    def run(self, controller, model):
        from config import config
        gesture_sequence = np.array([])
        while True:
            frame = controller.frame()
            for hand in frame.hands:
                pv = []
                av = []
                prev_finger = None
                c = hand.palm_position
                m = 1
                for finger in hand.fingers:
                    if prev_finger:
                        ad = finger.tip_position.distance_to(prev_finger.tip_position)
                        av.append(ad)
                        prev_finger = finger
                    else:
                        prev_finger = finger
                    pd = finger.tip_position.distance_to(c)
                    m = pd if finger.type == Leap.Finger.TYPE_MIDDLE else m
                    pv.append(pd)
                gesture_sequence = np.append(gesture_sequence, np.array(pv) / m)
                gesture_sequence = np.append(gesture_sequence, np.array(av) / m)
                if len(gesture_sequence) > 270:
                    if config.GR == 1:
                        prediction = model.predict(gesture_sequence[:270].reshape(1, 1, 270))
                        gesture = np.argmax(prediction)
                    else:
                        prediction = model.predict([gesture_sequence[:270]])
                        gesture = int(prediction[0])
                    if self.prev_gesture != gesture and gesture not in [0, 4]:
                        print("Gesture:", self.command_classes[gesture])
                        # self.queue.put({"operation": self.command_classes[gesture]})
                        self.prev_gesture = gesture
                    gesture_sequence = gesture_sequence[90:]
            time.sleep(0.01)

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

        controller = Leap.Controller()
        controller.set_policy_flags(Leap.Controller.POLICY_OPTIMIZE_HMD)
        try:
            self.run(controller, model)
        except KeyboardInterrupt:
            print("GestureEngine:KeyboardInterrupt")
            os.kill(os.getpid(), signal.SIGKILL)


def main():
    ge = GestureEngine(queue=Queue())
    ge.start_prediction()


if __name__ == '__main__':
    main()
