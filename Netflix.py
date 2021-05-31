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


def load_file(cl):
    #Función que se encarga de cargar las features
    if cl[-1] == 'y':
        des = np.loadtxt(cl).astype('float32')
        return des, "d"

    else:
        kps = []
        lines = [line.strip() for line in open(cl)]
        for line in lines:
            list = line.split(',')
            kp = cv2.KeyPoint(x=float(list[0]), y=float(list[1]), _size=float(list[2]), _angle=float(list[3]),
                        _response=float(list[4]), _octave=int(list[5]), _class_id=int(list[6]))
            kps.append(kp)
        return kps, 'k'
    

def get_coor(text, coor):
    font = cv2.FONT_HERSHEY_COMPLEX
    textsize = cv2.getTextSize(text, font, 2, 4)[0]
    
    textX = coor[0] - textsize[0]//2
    textY = coor[1] + textsize[1]//2
    return (textX, textY)

if len(sys.argv) == 2:
    img2 = cv2.imread(sys.argv[1])
else:
    print ("ERROR: Imagen no encontrada")
    exit



print("Cargando dataset de features...")
with open('Netflix_data.json') as f:
  data = json.load(f)
path = 'features/'
# Import Images
images = []
className = []
des_list = []
kp_list = []
myList = os.listdir(path)
i = -1
for cl in myList:
    if "_1.npy" in cl:
        i += 1
        images.append([cl[:cl.index("_")], [],[]])
    des,t = load_file(path+cl)
    if t == 'd':
        images[i][1].append(des)
    else:
        images[i][2].append(des)
print('Total Classes Detectted: ', len(images))
FLANN_INDEX_KDTREE = 1
MIN_MATCH_COUNT = 15
sift = cv2.xfeatures2d.SIFT_create(nfeatures=100000)


print("Iniciando detección...")
kp2, des2 = sift.detectAndCompute(img2, None)
area_list = []
dst_list = []
name_list = []
for cl, deslist, kplist in images:
    good_list = []
    kp1_list = []
    for kp1, des1 in zip(kplist,deslist):
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
            h,w,d = (192, 341, 3)
            pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
            if len(cv2.convexHull(pts, returnPoints = False)) != 4: continue
            dst = cv2.perspectiveTransform(pts,M)
            area_list.append((h-1)*(w-1))
            dst_list.append(dst)
            name_list.append(cl)
        except:
            print(cl)
            
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
                
                color = (255, 255, 255)
                if 'score' in dic:
                    score_info.append([str(dic['score']), tuple(scr_pos)])
                    if dic['score'] < 5:
                        color = (0,0,255)
                    elif 7>dic['score'] >=5:
                        color = (0,131,255)
                    elif 9>dic['score'] >=7:
                        color = (0,255,255)
                    else:
                        color = (0,179,55)
                else:
                    score_info.append(["S.N.", tuple(scr_pos)])
                cv2.fillPoly(overlay, [np.int32(dst_list[i])], color)
                break
        

alpha = 0.6
img2 = cv2.addWeighted(overlay, alpha, img2, 1, 0)
for score, pos in score_info:
    pos = get_coor(score, pos)
    cv2.putText(img2, score, pos, cv2.FONT_HERSHEY_COMPLEX, 2, (255, 255, 255), 4)
    cv2.putText(img2, score, pos, cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
print("FIN")
cv2.imwrite("resultado.jpg",img2)
cv2.imshow("resultado",img2)