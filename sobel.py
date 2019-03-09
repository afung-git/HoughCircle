import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


imgfile = 'images/one.jpg'
image = cv2.imread(imgfile)
output = image.copy()
imgBlur = cv2.medianBlur(image, 5)
gray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)


startTime = time.time()

edge_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
edge_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)    
edge = np.sqrt(edge_x**2 + edge_y**2)
orientation = np.arctan(edge_x/edge_y)
orientation = np.nan_to_num(orientation)


print(orientation.shape, edge_x.shape, edge.shape)
print(orientation)
edge.max()*.9

edge[edge < edge.max()*.3] = 0

edge /= edge.max()/255.0

edgeLine = edge

[ximg, yimg] = np.nonzero(edge) #(0,0) is upper left
# print(np.nonzero(edge))

edgePts = np.vstack([ximg,yimg]).T

print(edgePts.shape)

testPoint = edgePts[231]

print("Process Time(s):", time.time()-startTime)
print(orientation[(testPoint[0], testPoint[1])]*180/np.pi)

# for i in edgePts:
#     cv2.circle(output, (i[1], i[0]), 1, (0,0,255), 1)


print(testPoint)
print(orientation[testPoint[0], testPoint[1]])
cv2.circle(output, (testPoint[1], testPoint[0]), 1, (0,0,255), 2)

length = 20
p2X = (int)(testPoint[0] + length * np.cos(orientation[testPoint[0], testPoint[1]]))
p2Y = (int)(testPoint[1] - length * np.sin(orientation[testPoint[0], testPoint[1]]))

# print(p2X)
cv2.line(output,(testPoint[1],testPoint[0]), (p2X, p2Y), (0,0,255), 2)

cv2.imshow("output", np.hstack([image, output]))

# plt.imshow(edge, cmap='gray')
# plt.show()
# cv2.imshow("edges", edge)
cv2.waitKey()
cv2.destroyAllWindows()