import cv2 as cv # Import the OpenCV library
import numpy as np, pandas as pd
import os # Import the os library for work with operative system commands

def create_train_dataset_hu(path): # This function gets the directory path with the sorted shapes
    shapes = ["circle", "triangle", "square", "star"]
    patterns = []
    images = os.listdir(path) # List all files in the directory
    for image_file in images: # Iterate the get shapes of directory
        if image_file[-4:] == ".png": # It only acts if the file is a png image
            image = cv.imread(os.path.join(path, image_file)) # Read the image and it's converted to a integer matrix
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY) # Convert the image to be in grayscale
            ret, thresh = cv.threshold(gray,150,255,cv.THRESH_BINARY_INV) # Binarize the image to only have 0's and 255's
            contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE) # Get the contours of the image
            cnt = contours[0] # cnt is from the first get contour
            wrapper = cv.convexHull(cnt,returnPoints = False) # Clean the image
            deffects = cv.convexityDefects(cnt, wrapper) 
            moments = cv.moments(cnt) # Get the image's moments with its first contour
            hu_moments = cv.HuMoments(moments).flatten() # Get the hu moments
            x1, y1 = cnt[0,0]
            cv.drawContours(image,[cnt],-1, (44, 120, 200), 3)
            '''
            print("Hu-Moments of first contour:\n", hu_moments)
                cv.putText(image, 'Figure', (x1, y1), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200, 120, 60), 2)
                cv.imshow("Hu-Moments", image)
                cv.waitKey(0)
                cv.destroyAllWindows()
            '''
            for shape in shapes:
                if shape in image_file:
                    patterns.append({ # Add a dictionary with the hu moments in list
                        "class_name" : shape,
                        "h0" : hu_moments[0],
                        "h1" : hu_moments[1],
                        "h2" : hu_moments[2],
                        "h3" : hu_moments[3],
                        "h4" : hu_moments[4],
                        "h5" : hu_moments[5],
                        "h6" : hu_moments[6]})
    hu_dataset = pd.DataFrame(patterns) # Convert the list to dataframe
    return hu_dataset
# create_dataset("shapes/dataset/")