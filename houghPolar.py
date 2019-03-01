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
        # for i in range(minRadius+1):
        #     for j in range(minRadius+1):
        #         if x-i < 0 or x+i > 99 or y-j < 0 or y+j > 99:
        #             accumulator[x,y,r] = 0
        #         else:
        #             accumulator[(x-i, y -j, r)] = 0
        #             accumulator[(x+i, y +j, r)] = 0
    print(np.max(accumulator))
    return circle

imgfile = 'images/test2_200.jpg'

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


edges = cv2.Canny(gray,30,50)
[ximg,yimg] = np.nonzero(edges)
edgePts = np.vstack([ximg,yimg]).T

aMax = gray.shape[0]
bMax = gray.shape[1]
rMax = int(math.sqrt((aMax)**2 + (bMax)**2)) 
# print(aMax, bMax, rMax)

acc = np.zeros((aMax,bMax, rMax), dtype=int)

thetaRad = [i*np.pi/180 for i in range(360)]
# print(thetaRad)


startTime = time.time()
for i in edgePts:
    for r in range(5,rMax):
        for theta in thetaRad:
            a = i[0] - int(r*np.cos(theta))
            b = i[1] - int(r*np.sin(theta))
    #         print(a,b, i,round(r*np.sin(i*np.pi/180)) )
            if a >= 0 and a < aMax and b >= 0 and b < bMax:
                acc[(a,b,r)] += 1
                    
print("Voting Time(2):",time.time()-startTime)
plt.imshow(acc[:,:,15], cmap='gray')
plt.show()

print(np.amax(acc,axis=None))
print(np.unravel_index(np.argmax(acc,axis=None), acc.shape))

threshold = np.amax(acc,axis=None)*.7

startTime = time.time()
results = findCircles(acc,threshold, 5)
print("Results Time(s):", time.time()-startTime)
if results is not None:
    # circles = np.round(circles[0, :]).astype("int")

    for (x, y, r) in results:
        cv2.circle(output, (y,x), r, (0, 0,255), 2)


cv2.imshow("output", np.hstack([image, output]))
# cv2.imshow("edges", edges)
cv2.waitKey()
cv2.destroyAllWindows()
