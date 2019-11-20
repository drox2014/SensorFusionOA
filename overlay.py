import cv2
import time


def __draw_label(img, heading, text, pos, posHeading):
    font_face_head = cv2.FONT_HERSHEY_TRIPLEX
    font_face_text = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    scaleHead = 1
    color = (242, 204, 103)
    thickness = cv2.FILLED
    margin = 0

    # txt_size = cv2.getTextSize(text, font_face, scale, thickness)

    # end_x = pos[0] + txt_size[0][0] + margin
    # end_y = pos[1] - txt_size[0][1] - margin

    # cv2.rectangle(img, pos, (end_x, end_y), color, thickness)
    cv2.putText(img, text, pos, font_face_text, scale, color, 1, cv2.LINE_AA)
    cv2.putText(img, heading, posHeading, font_face_head, scaleHead, color, 1, cv2.LINE_AA)


def overlay(frame, background):
    return cv2.addWeighted(frame, 0.4, background, 0.5, 0)


# frame = overlay()
#
# __draw_label(frame,'heading', 'Hello World', (276,250),(276,145))
# cv2.imshow('combined.png', frame)
# cv2.waitKey(0)

vid = cv2.VideoCapture(2)
vid.set(3, 608)
vid.set(4, 608)

background = cv2.imread('data/overlay.png')

while True:
    ret, frame = vid.read()
    prev_time = time.time()

    resize = cv2.resize(frame, (680, 554))
    frame = overlay(resize, background)

    curr_time = time.time()
    exec_time = curr_time - prev_time
    print("time: %.2f FPS" % exec_time)
    cv2.imshow("Object Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
vid.release()
cv2.destroyAllWindows()
