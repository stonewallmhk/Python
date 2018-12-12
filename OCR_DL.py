import cv2
import os
import numpy as np
import pytesseract
from PIL import Image

file_name = 'F:/Data_Science/Datasets/DL/7.jpg'
dir =  os.path.splitext(file_name)[0]
image = cv2.imread(file_name)

# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/ 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

if angle < -45:
	angle = -(90 + angle)
else:
    angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
M = cv2.getRotationMatrix2D(center, angle, 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

# draw the correction angle on the image so we can validate it
#cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
# show the output image
print("[INFO] angle: {:.3f}".format(angle))
#cv2.imshow("Input", image)
#cv2.imshow("Rotated", rotated)
#cv2.waitKey(0)
'''
s = dir + '_skew.jpg'
cv2.imwrite(s,rotated)

resize = cv2.resize(rotated, (640, 480))
s = dir + '_resize.jpg'
cv2.imwrite(s,resize)

roi = resize[50:60, 200:250]
s = dir + '_roi.jpg'
cv2.imwrite(s,roi)

gray_img = cv2.cvtColor(rotated,cv2.COLOR_BGR2GRAY)
gray_img = cv2.fastNlMeansDenoising(gray_img)
_, threshold_img = cv2.threshold(gray_img, 60, 255, cv2.THRESH_BINARY_INV)
'''
img_clean = cv2.resize(rotated, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
img_clean = cv2.bilateralFilter(img_clean, 9, 75,75)

gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

#Thesholding
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
s = dir + '_roi.png'
cv2.imwrite(s, gray)
text = pytesseract.image_to_string(Image.open(s))
print(text)
