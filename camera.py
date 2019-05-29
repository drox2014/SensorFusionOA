import cv2
from operations import focus, count, color

command_classes = ['count', 'color', 'focus', 'no_op']


class CameraFeed:
    def __init__(self, camera_port):
        self.camera_port = camera_port
        self.operation = 3
        self.object_id = -1

        # Initialize webcam feed
        self.capture = cv2.VideoCapture(self.camera_port)
        self.capture.set(3, 1280)
        self.capture.set(4, 720)

    def process_frame(self, operation, object_id):
        self.operation = int(operation[0])
        self.object_id = object_id
        print(command_classes[self.operation], self.operation)

    def start_camera(self):
        while True:

            # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
            # i.e. a single-column array, where each item in the column has the pixel RGB value
            ret, frame = self.capture.read()
            # frame_expanded = np.expand_dims(frame, axis=0)
            if self.operation == 2:
                focus(frame, self.object_id)
            elif self.operation == 0:
                count(frame, self.object_id)
            elif self.operation == 1:
                color(frame, self.object_id)

            # All the results have been drawn on the frame, so it's time to display it.
            cv2.imshow('Object detector', frame)

            # Press 'q' to quit
            if cv2.waitKey(1) == ord('q'):
                break

        # Clean up
        self.capture.release()
        cv2.destroyAllWindows()
