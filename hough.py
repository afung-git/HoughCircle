import cv2
import numpy as np


imgfile = './detect_circles_soda.jpg'

image = cv2.imread(imgfile)
output = image.copy()
# print(image)
imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(imagegray, cv2.HOUGH_GRADIENT, 50, 1)

# print(circles)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv2.circle(output, (x,y), r, (0, 255, 0), 4)
        # cv2.rectangle(output, ?)

cv2.imshow("output", np.hstack([image, output]))
cv2.waitKey(0)