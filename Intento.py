import cv2
import numpy as np
import os
from imutils.convenience import resize
import time
import json
from itertools import compress
from collections import Counter
import sys


with open('Netflix_data.json') as f:
  data = json.load(f)
path = 'DBAnime'
# Import Images
images = []
className = []

myList = os.listdir(path)
print('Total Classes Detectted: ', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}')
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])

img2 = cv2.imread("Netflix1.jpg")

for img1 in images:
    try:
        sift = cv2.xfeatures2d.SIFT_create()
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        bf = cv2.BFMatcher(crossCheck=True)
        matches = bf.match(des1,des2)
        matches = sorted(matches, key = lambda x:x.distance)
        if len(matches)>100:
            good_matches = matches[:10]
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w = img1.shape[:2]
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            dst += (w, 0)  # adding offset
            img3 = cv2.polylines(img2, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)
            cv2.imshow("result", img3)
            cv2.waitKey(1)
    except:
        print("Except")