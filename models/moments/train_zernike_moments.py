import cv2 as cv, imutils as imt
import numpy as np, pandas as pd
import os
import mahotas as mh
from scipy.spatial import distance as dist

def create_train_dataset_zernike(path):
    shapes = ["circle", "triangle", "square", "star"]
    patterns = []
    images = os.listdir(path)
    for image_file in images:
        if image_file[-4:] == ".png":
            image = cv.imread(os.path.join(path, image_file))
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            thresh = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)[1]
            contours = cv.findContours(thresh.copy(), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            contours = imt.grab_contours(contours)
            cnt = max(contours, key = cv.contourArea)
            outline = np.zeros(gray.shape, dtype = "uint8")
            cv.drawContours(outline, [cnt], -1, 255, -1)
            zernike_moments = mh.features.zernike_moments(outline, cv.minEnclosingCircle(cnt)[1], degree = 8)
            for shape in shapes:
                if shape in image_file:
                    patterns.append({
                        "class_name" : shape,
                        "z0" : zernike_moments[0],
                        "z1" : zernike_moments[1],
                        "z2" : zernike_moments[2],
                        "z3" : zernike_moments[3],
                        "z4" : zernike_moments[4],
                        "z5" : zernike_moments[5],
                        "z6" : zernike_moments[6],
                        "z7" : zernike_moments[7],
                        "z8" : zernike_moments[8],
                        "z9" : zernike_moments[9],
                        "z10" : zernike_moments[10],
                        "z11" : zernike_moments[11],
                        "z12" : zernike_moments[12],
                        "z13" : zernike_moments[13],
                        "z14" : zernike_moments[14],
                        "z15" : zernike_moments[15],
                        "z16" : zernike_moments[16],
                        "z17" : zernike_moments[17],
                        "z18" : zernike_moments[18],
                        "z19" : zernike_moments[19],
                        "z20" : zernike_moments[20],
                        "z21" : zernike_moments[21],
                        "z22" : zernike_moments[22],
                        "z23" : zernike_moments[23],
                        "z24" : zernike_moments[24]
                    })
    zernike_dataset = pd.DataFrame(patterns)
    return zernike_dataset
