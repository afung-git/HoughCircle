import cv2
import numpy as np
import time
from matplotlib import pyplot as plt


imgfile = 'images/test2_400.jpg'
image = cv2.imread(imgfile)
output = image.copy()
imgBlur = cv2.medianBlur(image, 5)
gray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)


startTime = time.time()

edge_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
edge_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)    
edge = np.sqrt(edge_x**2 + edge_y**2)
orientation = np.arctan(edge_x/edge_y)

print(orientation.shape, edge_x.shape, edge.shape)
print(orientation)
edge.max()*.9

edge[edge < edge.max()*.3] = 0

edge /= edge.max()/255.0

[ximg,yimg] = np.nonzero(edge)
edgePts = np.vstack([ximg,yimg]).T


print("Process Time(s):", time.time()-startTime)
print(edgePts.shape)
# plt.imshow(edge, cmap='gray')
# plt.show()
cv2.imshow("edges", edge)
cv2.waitKey()
cv2.destroyAllWindows()