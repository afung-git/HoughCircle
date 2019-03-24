import cv2
import sys
import numpy as np
import math
import time
from matplotlib import pyplot as plt


def findCircles(accumulator, threshold, minRadius):

    def findMax (array):
        max = np.unravel_index(np.argmax(array,axis=None), array.shape)
        # value = array[(np.unravel_index(np.argmax(array,axis=None), array.shape))]
        return (max)
    
    circle = []
    x = y = r = 0
    while accumulator[(findMax(accumulator))] > threshold:  
        # accumulator[x,y,r)] = 0
        (x,y,r) = findMax(accumulator)
        circle.append((x,y,r))
        accumulator[(x,y,r)] = 0
        for i in range(minRadius+1):
            for j in range(minRadius+1):
                if x-i < 0 or x+i > 99 or y-j < 0 or y+j > 99:
                    accumulator[x,y,r] = 0
                else:
                    accumulator[(x-i, y -j, r)] = 0
                    accumulator[(x+i, y +j, r)] = 0
    print(np.max(accumulator), " new Max")
    return circle

imgfile = 'images/test1.jpg'

image = cv2.imread(imgfile)
output = image.copy()
imgBlur = cv2.medianBlur(image, 5)
gray = cv2.cvtColor(imgBlur,cv2.COLOR_BGR2GRAY)

# print(image)
imagegray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)




# implementation based on OpenCV HoughCircles algorithm

# circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT,1,50, param1 = 100, param2 = 30)

# print(circles)

# if circles is not None:
#     circles = np.round(circles[0, :]).astype("int")

#     for (x, y, r) in circles:
#         cv2.circle(output, (x,y), r, (0, 0,255), 2)
        # cv2.rectangle(output, ?)



# implementation using custom algorithm


edge_x = cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=3)
edge_y = cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=3)    
edge = np.sqrt(edge_x**2 + edge_y**2)
orientation = np.arctan2(edge_y,edge_x)

# print(orientation.shape, edge_x.shape, edge.shape)

edge[edge < edge.max()*.3] = 0


edge /= edge.max()/255.0


[ximg, yimg] = np.nonzero(edge) #(0,0) is upper left
# print(np.nonzero(edge))

edgePts = np.vstack([ximg,yimg]).T

# print(edgePts.shape)

aMax = gray.shape[0]
bMax = gray.shape[1]
rMax = int(math.sqrt((aMax)**2 + (bMax)**2)) 
# print(aMax, bMax, rMax)

acc = np.zeros((aMax,bMax, rMax), dtype=int)

# thetaRad = [i*np.pi/180 for i in range(360)]
# print(thetaRad)

# ---- This is only the case for a clean black and white image
# Angles for circle zero is verticle and angles increment counter clockwise
# angles for gradient zero is left horizonal and increment to +180 clockwise, increment to -180 counter clockwise



# i = edgePts[130]
# theta = orientation[(i[0], i[1])]*180/np.pi
# print(theta, " grad angle")
# print(270 - theta, " altered angle")

startTime = time.time()
for i in edgePts:
    for r in range(5, rMax):
        theta = 90 - orientation[(i[0], i[1])]*180/np.pi
        a = i[0] - int(r*np.cos(theta*np.pi/180))
        b = i[1] - int(r*np.sin(theta*np.pi/180))
    #         print(a,b, i,round(r*np.sin(i*np.pi/180)) )
        # cv2.circle(output, (b,a), r,(0,0,255), 1)
        if a >= 0 and a < aMax and b >= 0 and b < bMax:
            acc[(a,b,r)] += 1
                
print("Voting Time(s):",time.time()-startTime)
# print(edge)

# cv2.circle(output, (i[1],i[0]), 1, (255,0,0), 2)
# cv2.circle(output, (b,a), r,(0,0,255), 1)

plt.imshow(acc[:,:,15], cmap='gray')
plt.show()

print(np.amax(acc,axis=None), " max votes")
print(np.unravel_index(np.argmax(acc,axis=None), acc.shape), " x, y, radius")

threshold = np.amax(acc,axis=None)*.6

startTime = time.time()
results = findCircles(acc,threshold, 5)
print("Results Time(s):", time.time()-startTime)
if results is not None:
    # circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in results:
        cv2.circle(output, (y,x), r, (0, 0,255), 1)


cv2.imshow("output", np.hstack([image, output]))
cv2.imshow("edges", edge)
cv2.waitKey()
cv2.destroyAllWindows()
