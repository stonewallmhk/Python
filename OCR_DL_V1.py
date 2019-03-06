import cv2
import os
import imutils
import numpy as np

def show(title, image):
    cv2.imshow(title,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

file_name = 'C:/Hari Docs/Dataset/DL1/11.jpg'
image = cv2.imread(file_name)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
image = imutils.resize(image, height=600)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
#thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
#thresh = cv2.erode(thresh, None, iterations=4)
show('roi',thresh)
