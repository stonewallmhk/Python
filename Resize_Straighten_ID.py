import cv2
import os
import numpy as np
file_name = 'C:/Hari Docs/Dataset/DL/1.jpg'
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

s = dir + '_modified.jpg'
cv2.imwrite(s,rotated)

resize = cv2.resize(rotated, (640, 480))
s = dir + '_resize.jpg'
cv2.imwrite(s,resize)
