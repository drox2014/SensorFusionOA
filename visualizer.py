import cv2


def stream(fusion_engine):

    while True:
        frame = fusion_engine.image_dequeue()
        cv2.imshow('Object detector', frame)
        # Press 'q' to quit
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('a'):
            fusion_engine.enqueue_command({"operation": "Describe", "object_id": 3, "multiple": False, "pointing": False})
        elif key == ord('b'):
            fusion_engine.enqueue_command({"operation": "Describe", "object_id": 3, "multiple": True, "pointing": False})
        elif key == ord('c'):
            fusion_engine.enqueue_command({"operation": "Describe", "object_id": 3, "multiple": False, "pointing": True})
        # Clean up
    cv2.destroyAllWindows()

