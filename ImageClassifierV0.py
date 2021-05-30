import cv2
import numpy as np
import os
from imutils.convenience import resize
import time
import json




def findid(img, desList, thres=20):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalVal = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75*n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    print(matchList)
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalVal = matchList.index(max(matchList))
    return finalVal


def findDes(images):
    desList = []
    for img in images:
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList


def pyramid(image, scale=1.5, minSize=(30, 30)):
    # yield the original image
    yield image

    # keep looping over the pyramid
    while True:
        # compute the new dimensions of the image and resize it
        w = int(image.shape[1] / scale)
        image = resize(image, width=w)

        # if the resized image does not meet the supplied minimum
        # size, then stop constructing the pyramid
        if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
            break

        # yield the next image in the pyramid
        yield image


def sliding_window(image, stepSize, windowSize):
    # slide a window across the image
    for y in range(0, image.shape[0], stepSize):
        for x in range(0, image.shape[1], stepSize):
            # yield the current window
            yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])


with open('Netflix_data.json') as f:
  data = json.load(f)
path = 'cromos'
orb = cv2.ORB_create(nfeatures=1000)
# Import Images
images = []
className = []

myList = os.listdir(path)
print('Total Classes Detectted: ', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])
# print(className)

desList = findDes(images)
print(len(desList))


# load the image and define the window width and height
image = cv2.imread("robertoanimetorcida.jpeg")
#image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
(winW, winH) = (300, 200)
lastid = -1
# loop over the image pyramid
for resized in pyramid(image, scale=1.5):
    # loop over the sliding window for each layer of the pyramid
    for (x, y, window) in sliding_window(resized, stepSize = 150, windowSize=(winW, winH)):
        # if the window does not meet our desired window size, ignore it
        if window.shape[0] != winH or window.shape[1] != winW:
            continue
        clone = resized.copy()
        id = findid(window, desList)
        if id != -1:
            for element in data:
                if str(element['id']) == className[id]:
                    if className[id] == lastid:
                        break
                    cv2.putText(window, element['title'], (50, 50),cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(window, element['title'], (50, 50),cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                    lastid = className[id]
                    break
        cv2.rectangle(clone, (x, y), (x + winW, y + winH), (0, 255, 0), 2)
        cv2.imshow('img2',window)
        cv2.waitKey(1)
cv2.imwrite("resultado.png", image)


'''

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append([m])

print(len(good))


img3 = cv2.drawMatchesKnn(img1,kp1,img2,kp2,good,None,flags=2)
cv2.imshow('img1',img1)
cv2.imshow('img2',img2)
cv2.imshow('img3',img3)

cv2.waitKey(0)
'''
