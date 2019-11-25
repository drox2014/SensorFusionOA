from multiprocessing import Queue
import numpy as np
import cv2


class CameraFeed:
    def __init__(self, camera_port):
        # from object_detection import VisionEngine
        self.camera_port = camera_port
        self.operation = 3
        self.is_zoomed = False
        self.frame_size = 608
        self.scale = 40
        self.object_id = -1
        # self.ve = VisionEngine()
        self.__last_operation = {"operation": None}

        # Initialize webcam feed
        self.capture = cv2.VideoCapture(self.camera_port)
        self.capture.set(3, 608)
        self.capture.set(4, 608)

    def zoom_in(self):
        self.is_zoomed = True

    def zoom_out(self):
        self.is_zoomed = False

    def perform(self, gesture):
        self.__last_operation = gesture

    def start_camera(self):

        while True:
            ret, frame = self.capture.read()

            if self.is_zoomed:
                print("zooming...")
                # get the webcam size
                height, width, channels = frame.shape

                frame = frame[50:height - 50, 50:width - 50]
                frame = cv2.resize(frame, (width, height))

                # # prepare the crop
                # center = int(height / 2), int(width / 2)
                # radius = int(self.scale * height / 100), int(self.scale * width / 100)
                #
                # minX, maxX = center[0] - radius[0], center[0] + radius[0]
                # minY, maxY = center[1] - radius[1], center[1] + radius[1]
                #
                # cropped = frame[minX:maxX, minY:maxY]
                # frame = cv2.resize(cropped, (width, height))

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)
            # Press 'q' to quit
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('a'):
                self.is_zoomed = not self.is_zoomed
                # Clean up
        self.capture.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    CameraFeed(2).start_camera()
