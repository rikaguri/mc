#! /usr/bin/python
# -*- coding: utf-8 -*-
import cv2
import dlib
import numpy as np
import imutils
from imutils import face_utils

def face_shape_detector_dlib(img):
    predictor_path = "./shape_predictor_68_face_landmarks.dat"
    predictor = dlib.shape_predictor(predictor_path)
    detector = dlib.get_frontal_face_detector()
    #img = cv2.imread(str(img))
    size = img.shape
    # 処理高速化のためグレースケール化(任意)
    img_gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dets = detector(img_gry, 1)

    # 検出した全顔に対して処理
    for det in dets:
        # 顔のランドマーク検出
        landmark = predictor(img_gry, det)
        # 処理高速化のためランドマーク群をNumPy配列に変換(必須)
        landmark = face_utils.shape_to_np(landmark)
        points = landmark.tolist()
        # (0,0),(x,0),(0,y),(x,y)
        points.append([0, 0])
        points.append([int(size[1] - 1), 0])
        points.append([0, int(size[0] - 1)])
        points.append([int(size[1] - 1), int(size[0] - 1)])
        # (x/2,0),(0,y/2),(x/2,y),(x,y/2)
        points.append([int(size[1] / 2), 0])
        points.append([0, int(size[0] / 2)])
        points.append([int(size[1] / 2), int(size[0] - 1)])
        points.append([int(size[1] - 1), int(size[0] / 2)])

        ans=[]
        for i,(x,y) in enumerate(points):
            ans.append((x,y))


    return ans

