# !/usr/bin/env python

import cv2
import numpy as np
import tkinter_gui
import capture_face_landmarks
from PIL import Image
import datetime
import pathlib

def changeSize(img):
    h,w,c = img.shape
    if h>= 1000 | w>= 1000:
        img = cv2.resize(img,(int(w*0.5), int(h*0.5)))
    return img



def calculateDelaunayTriangles(rect, points):
    subdiv = cv2.Subdiv2D(rect)
    for p in points:
        subdiv.insert(p)

    triangleList = subdiv.getTriangleList()

    delaunayTri = []

    pt = []

    for t in triangleList:
        pt.append((t[0], t[1]))
        pt.append((t[2], t[3]))
        pt.append((t[4], t[5]))

        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])

        if rectContains(rect, pt1) and rectContains(rect, pt2) and rectContains(rect, pt3):
            ind = []
            for j in range(0, 3):
                for k in range(0, len(points)):
                    if (abs(pt[j][0] - points[k][0]) < 1.0 and abs(pt[j][1] - points[k][1]) < 1.0):
                        ind.append(k)
            if len(ind) == 3:
                delaunayTri.append((ind[0], ind[1], ind[2]))

        pt = []

    return triangleList, delaunayTri


def rectContains(rect, point):
    if point[0] < rect[0]:
        return False
    elif point[1] < rect[1]:
        return False
    elif point[0] > rect[0] + rect[2]:
        return False
    elif point[1] > rect[1] + rect[3]:
        return False
    return True


# Apply affine transform calculated using srcTri and dstTri to src and
# output an image of size.
def applyAffineTransform(src, srcTri, dstTri, size):
    # Given a pair of triangles, find the affine transform.
    warpMat = cv2.getAffineTransform(np.float32(srcTri), np.float32(dstTri))

    # Apply the Affine Transform just found to the src image
    dst = cv2.warpAffine(src, warpMat, (size[0], size[1]), None, flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_REFLECT_101)

    return dst


# Warps and alpha blends triangular regions from img1 and img2 to img
def morphTriangle(img1, img2, img, t1, t2, t, alpha):
    # Find bounding rectangle for each triangle
    r1 = cv2.boundingRect(np.float32([t1]))
    r2 = cv2.boundingRect(np.float32([t2]))
    r = cv2.boundingRect(np.float32([t]))

    # Offset points by left top corner of the respective rectangles
    t1Rect = []
    t2Rect = []
    tRect = []

    for i in range(0, 3):
        tRect.append(((t[i][0] - r[0]), (t[i][1] - r[1])))
        t1Rect.append(((t1[i][0] - r1[0]), (t1[i][1] - r1[1])))
        t2Rect.append(((t2[i][0] - r2[0]), (t2[i][1] - r2[1])))

    # Get mask by filling triangle
    mask = np.zeros((r[3], r[2], 3), dtype=np.float32)
    cv2.fillConvexPoly(mask, np.int32(tRect), (1.0, 1.0, 1.0), 16, 0);

    # Apply warpImage to small rectangular patches
    img1Rect = img1[r1[1]:r1[1] + r1[3], r1[0]:r1[0] + r1[2]]
    img2Rect = img2[r2[1]:r2[1] + r2[3], r2[0]:r2[0] + r2[2]]

    size = (r[2], r[3])
    warpImage1 = applyAffineTransform(img1Rect, t1Rect, tRect, size)
    warpImage2 = applyAffineTransform(img2Rect, t2Rect, tRect, size)

    # Alpha blend rectangular patches
    imgRect = (1.0 - alpha) * warpImage1 + alpha * warpImage2

    # Copy triangular region of the rectangular patch to the output image
    img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] = img[r[1]:r[1] + r[3], r[0]:r[0] + r[2]] * (1 - mask) + imgRect * mask


if __name__ == '__main__':
    filename1 = tkinter_gui.file_maker('jpeg')
    filename2 = tkinter_gui.file_maker('jpeg')

    # Read images
    img1r = cv2.imread(filename1)
    img2r = cv2.imread(filename2)

    #imageのサイズが大きすぎたら調節する
    img1r = changeSize(img1r)
    img2r = changeSize(img2r)

    # Convert Mat to float data type
    img1 = np.float32(img1r)
    img2 = np.float32(img2r)
    # print(img1)

    # get delaunay format
    size = img1.shape
    rect = (0, 0, size[1], size[0])

    # Read array of corresponding points
    points1 = capture_face_landmarks.face_shape_detector_dlib(img1r)
    points2 = capture_face_landmarks.face_shape_detector_dlib(img2r)

    imgfin = []

    for j in range(0, 100):
        # Compute weighted average point coordinates
        alpha = j * 0.01
        points = []
        for i in range(0, len(points1)):
            x = (1 - alpha) * points1[i][0] + alpha * points2[i][0]
            y = (1 - alpha) * points1[i][1] + alpha * points2[i][1]
            points.append((x, y))

        # Allocate space for final output
        imgMorph = np.zeros(img1.shape, dtype=img1.dtype)
        triangles, delaunay = calculateDelaunayTriangles(rect, points)
        # print(delaunay)

        # Read triangles
        for i, (x, y, z) in enumerate(delaunay):
            x = int(x)
            y = int(y)
            z = int(z)

            t1 = [points1[x], points1[y], points1[z]]
            t2 = [points2[x], points2[y], points2[z]]
            t = [points[x], points[y], points[z]]
            # Morph one triangle at a time.
            morphTriangle(img1, img2, imgMorph, t1, t2, t, alpha)
        imgMorph = cv2.cvtColor(imgMorph, cv2.COLOR_BGR2RGB)
        imgMorph = Image.fromarray(np.uint8(imgMorph),'RGB')
        imgfin.append(imgMorph)

    # gif画像にして保存
    now = datetime.datetime.now()
    time = now.strftime('%Y_%m_%d_%H_%M')
    savename = '/Users/erika/PycharmProjects/mc/結果/'+time + '.gif'
    a = pathlib.Path(savename)
    a.touch()

    imgfin[0].save(savename, save_all=True, append_images=imgfin[1:], loop=0, duration=30)

    # # 読み込みと表示
    # filepath3 = tkinter_gui.file_maker('gif')
    # gif = cv2.VideoCapture(filepath3)
    # fps = gif.get(cv2.CAP_PROP_FPS)  # fpsは１秒あたりのコマ数
    #
    # for t in range(len(gif)):
    #     cv2.imshow('test', gif[t])
    #     cv2.waitKey(int(1000 / fps))  # １コマを表示するミリ秒

    #cv2.waitKey(0)
