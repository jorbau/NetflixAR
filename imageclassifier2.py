import cv2
import numpy as np
import os
from imutils.convenience import resize
import time
import json
from itertools import compress
from collections import Counter
import sys
from matplotlib import pyplot as plt
import statistics as stats



with open('Netflix_data.json') as f:
  data = json.load(f)
path = 'cromos2'
# Import Images
images = []
className = []

myList = os.listdir(path)
print('Total Classes Detectted: ', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}')
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 10
sift = cv2.xfeatures2d.SIFT_create(nfeatures=100000)

img2 = cv2.imread("perspectiva2.jpeg")
#img2 = cv2.resize(img2, (960, 540))  
kp2, des2 = sift.detectAndCompute(img2, None)
area_list= []
dst_list = []
for img1 in images:
    try:
        kp1, des1 = sift.detectAndCompute(img1, None)
        index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        search_params = dict(checks = 50)
        bf = cv2.FlannBasedMatcher(index_params, search_params)
        matches = bf.knnMatch(des1,des2,k=2)
        good = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good.append(m)
        if len(good)>MIN_MATCH_COUNT:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
            matchesMask = mask.ravel().tolist()
            h,w,d = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            dst = cv2.perspectiveTransform(pts,M)
            area_list.append((h-1)*(w-1))
            dst_list.append(dst)
            #img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
        else:
            #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
            matchesMask = None
        draw_params = dict(matchColor = (0,255,0), # draw matches in green color
                   singlePointColor = None,
                   matchesMask = matchesMask, # draw only inliers
                   flags = 2)
        #img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    except:
        print('except')
area_list.sort()
mediana = stats.median(area_list)
for i in range(len(area_list)-1):
    if area_list[i]<=mediana+10000 and area_list[i]>=mediana-10000:
        img2 = cv2.polylines(img2,[np.int32(dst_list[i])],True,255,3, cv2.LINE_AA)



cv2.imwrite('resultado.png',img2)