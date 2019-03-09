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
# orientation = np.arctan(edge_x/edge_y)
orientation = np.arctan2(edge_y,edge_x)

print(orientation.shape, edge_x.shape, edge.shape)
print(orientation)
edge.max()*.9

edge[edge < edge.max()*.3] = 0
[oriX, oriY] = np.nonzero(orientation)
oriPts = np.vstack([oriX,oriY]).T

edge /= edge.max()/255.0

[ximg,yimg] = np.nonzero(edge)
edgePts = np.vstack([ximg,yimg]).T

# edgePts[0]
for j in edgePts:
    angle = orientation[(j[0],j[1])]*180/np.pi
    print(j, angle)
# j = 1
# print("Process Time(s):", time.time()-startTime)
# print(edgePts[j])
# angle = orientation[(edgePts[j][0],edgePts[j][1])]*180/np.pi
# print(angle)
# plt.imshow(edge, cmap='gray')
# plt.show()
cv2.imshow("edges", edge)
cv2.waitKey()
cv2.destroyAllWindows()