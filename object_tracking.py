import time
import cv2
from object_detection import VisionEngine


# OPENCV_OBJECT_TRACKERS = {
#     "csrt": cv2.TrackerCSRT_create,
#     "kcf": cv2.TrackerKCF_create,
#     "boosting": cv2.TrackerBoosting_create,
#     "mil": cv2.TrackerMIL_create,
#     "tld": cv2.TrackerTLD_create,
#     "medianflow": cv2.TrackerMedianFlow_create,
#     "mosse": cv2.TrackerMOSSE_create
# }


def main():
    tracker = None
    ve = VisionEngine()
    vid = cv2.VideoCapture(0)
    vid.set(3, 608)
    vid.set(4, 608)

    counter = 0


    while True:
        ret, frame = vid.read()
        if counter < 20:
            bboxes = ve.get_frcnn_prediction(frame)
            ve.draw_bbox(frame, bboxes)
            if counter == 19:
                if len(bboxes):
                    bbox = [int(r) for r in bboxes[0][:4]]
                    rect = (bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1])
                    tracker = cv2.TrackerKCF_create()
                    tracker.init(frame, rect)

                    counter += 1
            else:
                counter += 1

        else:
            res, bbox = tracker.update(frame)
            if res:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255),
                            2)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # elif key == ord('a'):
        # bboxes = ve.get_frcnn_prediction(frame)
        # ve.draw_bbox(frame, bboxes)
        cv2.imshow("Object Detector", frame)

    # Clean up
    vid.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
