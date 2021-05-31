# -*- coding: utf-8 -*-
import shutil
import cv2
import numpy as np
import os
import sys

output= 'features/' 

if len(sys.argv) == 2:
    path = cv2.imread(sys.argv[1])
else:
    print ("ERROR: Imagen no encontrada")
    exit

try:
    shutil.rmtree(output[:-1])
except OSError as e:
    pass
os.mkdir(output)

# featureSun: Calculate the number of feature points
myList = os.listdir(path)
images = []
className = []
detector = cv2.xfeatures2d.SIFT_create(nfeatures=10000)
print('Total Classes Detectted: ', len(myList))
for cl in myList:
    imgCur = cv2.imread(f'{path}/{cl}')
    images.append(imgCur)
    className.append(os.path.splitext(cl)[0])

for img, name in zip(images, className):

    kp , des = detector.detectAndCompute(img,None)
    file = output + name + ".npy"
    np.savetxt(file, des)
    file = output + name + ".txt"
    f = open(file, "w")
    for point in kp:
        p = str(point.pt[0]) + "," + str(point.pt[1]) + "," + str(point.size) + "," + str(point.angle) + "," + str(
            point.response) + "," + str(point.octave) + "," + str(point.class_id) + "\n"
        f.write(p)
    f.close()







