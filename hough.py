import cv2
import numpy as np
from matplotlib import pyplot as plt


imgfile = 'images/multiple.jpg'

image = cv2.imread(imgfile, 0)
output = image.copy()
# print(image)
# imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(image, cv2.HOUGH_GRADIENT,1,5)

print(circles)

if circles is not None:
    circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in circles:
        cv2.circle(output, (x,y), r, (150, 0,0), 2)
        # cv2.rectangle(output, ?)

# plt.imshow(image)

cv2.imshow("output", np.hstack([image, output]))
cv2.waitKey(0)
cv2.destroyAllWindows()