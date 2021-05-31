import cv2
import numpy as np
import os
import time
import json
from itertools import compress
from collections import Counter
import sys
from matplotlib import pyplot as plt
import statistics as stats

def get_coor(text, coor):
    font = cv2.FONT_HERSHEY_COMPLEX
    textsize = cv2.getTextSize(text, font, 2, 4)[0]
    
    textX = coor[0] - textsize[0]//2
    textY = coor[1] + textsize[1]//2
    return (textX, textY)

with open('Netflix_data.json') as f:
  data = json.load(f)
path = 'cromosFinal'
# Import Images
images = []
className = []

myList = os.listdir(path)
i = -1
for cl in myList:
    if "_" in cl:
        if "_1.png" in cl:
            i += 1
            images.append([cl[:cl.index("_")], []])
        imgCur = cv2.imread(f'{path}/{cl}')
        images[i][1].append(imgCur)
        className.append(os.path.splitext(cl[:cl.index("_")])[0])
print('Total Classes Detectted: ', len(images))
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 10
sift = cv2.xfeatures2d.SIFT_create(nfeatures=100000)

img2 = cv2.imread("Nueva.jpg")
kp2, des2 = sift.detectAndCompute(img2, None)
area_list = []
dst_list = []
name_list = []
for cl, limg in images:
    good_list = []
    kp1_list = []
    for img1 in limg:
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
            good_list.append(good)
            kp1_list.append(kp1)
            
    if len(good_list)>0:
        good = good_list[0]
        kp1 = kp1_list[0]
        for n in range(len(good_list)):
            if len(good_list[n])>len(good):
                good = good_list[n]
                kp1 = kp1_list[n]
                
        try:
            src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
            dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            matchesMask = mask.ravel().tolist()
            h,w,d = img1.shape
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            if len(cv2.convexHull(pts, returnPoints = False)) != 4: continue
            dst = cv2.perspectiveTransform(pts,M)
            area_list.append((h-1)*(w-1))
            dst_list.append(dst)
            name_list.append(cl)
        except:
            pass
            
area_list.sort()
mediana = stats.median(area_list)
overlay = img2.copy()
score_info = []
for i in range(len(area_list)-1):
    if area_list[i]<=mediana+10000 and area_list[i]>=mediana-10000:
        for dic in data:
            if str(dic["id"]) == name_list[i]:
                long = max(abs(dst_list[i][0][0][0] - dst_list[i][1][0][0]), 
                           abs(dst_list[i][0][0][0] - dst_list[i][2][0][0]),
                           abs(dst_list[i][0][0][0] - dst_list[i][3][0][0]),
                           abs(dst_list[i][1][0][0] - dst_list[i][2][0][0]),
                           abs(dst_list[i][1][0][0] - dst_list[i][3][0][0]),
                           abs(dst_list[i][2][0][0] - dst_list[i][3][0][0]))
                if long>0:
                    text_len = len(dic['title'])
                    text_size = cv2.getTextSize(dic['title'], cv2.FONT_HERSHEY_COMPLEX, 1, 2)[0][0]
                    caracter = text_size/text_len
                    cv2.putText(img2, dic['title'][:int(long/caracter)], tuple(dst_list[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    cv2.putText(img2, dic['title'][:int(long/caracter)], tuple(dst_list[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
                else:
                    break
                scr_pos = [0, 0]
                for x in dst_list[i]:
                    scr_pos[0] += int(x[0][0]/4)
                    scr_pos[1] += int(x[0][1]/4)
                
                color = (0, 0, 255)
                if 'score' in dic:
                    score_info.append([str(dic['score']), tuple(scr_pos)])
                    if dic['score'] >= 5:
                        color = (0, 255, 0)
                else:
                    score_info.append(["Sin Nota", tuple(scr_pos)])
                cv2.fillPoly(overlay, [np.int32(dst_list[i])], color)
                break
            
        if str(dic["id"]) != name_list[i]:
            cv2.putText(img2, "None", tuple(dst_list[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(img2, "None", tuple(dst_list[i][0][0]), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 0), 1)
            cv2.fillPoly(overlay, [np.int32(dst_list[i])], (0, 0, 255))

alpha = 0.4
img2 = cv2.addWeighted(overlay, alpha, img2, 1, 0)
for score, pos in score_info:
    pos = get_coor(score, pos)
    cv2.putText(img2, score, pos, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img2, score, pos, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)



cv2.imwrite('resultado.png',img2)