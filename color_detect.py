import numpy as np
import argparse
import cv2

# load the image
image = cv2.imread("/home/darshanakg/Projects/playground/flower.jpg")

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# define the list of boundaries
boundaries = [
    ([0, 70, 50], [10, 255, 255]),
    ([170, 70, 50], [180, 255, 255]),
    ([0, 0, 255], [51, 255, 255]),
    ([103, 0, 255], [145, 255, 255])
]

# loop over the boundaries
for (lower, upper) in boundaries:
    # create NumPy arrays from the boundaries
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # find the colors within the specified boundaries and apply
    # the mask
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(image, image, mask=mask)

    # show the images
    cv2.imshow("images", np.hstack([image, output]))
    cv2.waitKey(0)
