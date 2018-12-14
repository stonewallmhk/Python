import cv2
import numpy as np
import os
import pytesseract

file_name = 'C:/Hari Docs/Dataset/DL/1.jpg'
dir =  os.path.splitext(file_name)[0]
image = cv2.imread(file_name)

names = image[40:80,150:280]
cv2.imshow('Crop',image)
cv2.waitKey(0)
text = pytesseract.image_to_string(image)
print(text)
