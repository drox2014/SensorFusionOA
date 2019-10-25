import sys
import time

import numpy as np

import Leap


# import tensorflow as tf
# from tensorflow import keras


class GestureEngine:
    def __init__(self):
        import os
        # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # os.environ["CUDA_VISIBLE_DEVICES"] = ""
        from tensorflow import keras
        import tensorflow as tf

        self.running = True
        self.command_classes = ['Pointing', 'Capture', 'ZoomIn', 'ZoomOut', 'Roaming']

        # Initializing the model
        config = tf.ConfigProto(intra_op_parallelism_threads=4,
                                inter_op_parallelism_threads=4,
                                allow_soft_placement=True,
                                device_count={'CPU': 1, 'GPU': 0})
        session = tf.Session(config=config)
        tf.keras.backend.set_session(session)
        self.model = keras.models.load_model("./data/gesture_lstm_v9.h5")
        controller = Leap.Controller()
        try:
            self.run(controller)
            print("Starting gesture engine...")
        except KeyboardInterrupt:
            sys.exit(0)

    def run(self, controller):
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
                    prediction = self.model.predict(gesture_sequence[:270].reshape(1, 1, 270))
                    gesture = self.command_classes[np.argmax(prediction)]
                    print(gesture)
                    gesture_sequence = gesture_sequence[90:]
            time.sleep(0.01)

    def stop_prediction(self):
        self.running = False


# def main():
#     GestureEngine()
#
#
# if __name__ == '__main__':
#     main()
