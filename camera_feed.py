import cv2


class CameraFeed:
    def __init__(self, camera_port):
        self.camera_port = camera_port

        # Initialize webcam feed
        self.capture = cv2.VideoCapture(self.camera_port)
        self.capture.set(3, 608)
        self.capture.set(4, 608)

    def start_feed(self, fusion_engine):

        while True:
            try:
                ret, frame = self.capture.read()
                fusion_engine.image_enqueue(frame)
            except KeyboardInterrupt:
                break

        # Clean up
        self.capture.release()
