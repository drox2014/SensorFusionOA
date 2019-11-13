import cv2


def stream(fusion_engine):

    while True:
        frame = fusion_engine.image_dequeue()
        cv2.imshow('Object detector', frame)
        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        # Clean up
    cv2.destroyAllWindows()

