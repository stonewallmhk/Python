import cv2
import numpy as np
import os
import tempfile
import re
from datetime import datetime
from PIL import Image
import pytesseract

def clean_image(crop):

    img_clean = cv2.resize(crop, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)
    img_clean = cv2.bilateralFilter(img_clean, 9, 75,75)

    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

    #Thesholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    return gray

# https://www.pyimagesearch.com/2017/02/20/text-skew-correction-opencv-python/ 
def align_resize(image):

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

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jpg')
    temp_filename = temp_file.name
    cv2.imwrite(temp_filename, rotated)

    print("[INFO] angle: {:.3f}".format(angle))

    #Resizing image
    im = Image.open(temp_filename)
    im_resized = im.resize((640,480), Image.ANTIALIAS)
    im_resized.save(temp_filename, dpi=(300, 300))  # best for OCR

    image_align_resize = cv2.imread(temp_filename)
    return image_align_resize

def text_extract(text):
    textlines = text.splitlines()
    #Get Lastname
    lastname = textlines[0]
    lastname = ''.join(re.findall('[A-Z]',lastname))
    #Get Firstname and Middlename
    firstname = textlines[1]
    firstname = ''.join(re.findall('[A-Z]', firstname))
    middlename = textlines[2]
    middlename = ''.join(re.findall('[A-Z]',middlename))
    if middlename:    
        firstname = firstname+' '+middlename
    #Get Date of Birth and Country
    dob_cntry = textlines[3]
    char_pos,_ = re.search('[A-Z]', dob_cntry).span(0)
    country = dob_cntry[char_pos:]
    match = re.search(r'\d{2}.\d{2}.\d{4}', dob_cntry)
    dob = datetime.strptime(match.group(), '%d.%m.%Y').strftime('%d-%m-%Y')
    #Get Date of DL Issue
    doi = textlines[5]
    match = re.search(r'\d{2}.\d{2}.\d{4}', doi)
    date_of_issue = datetime.strptime(match.group(),'%d.%m.%Y').strftime('%d-%m-%Y')
    #Get Date of DL Expiry
    doe= textlines[7]
    match = re.search(r'\d{2}.\d{2}.\d{4}', doe)
    date_of_expiry = datetime.strptime(match.group(),'%d.%m.%Y').strftime('%d-%m-%Y')
    #Get DL Number
    DL_number = textlines[9]
    if DL_number[0] == '5' or DL_number[0] == '5.':
        DL_number = DL_number.split(' ',1)
        DL_number = DL_number[1]
    return lastname,firstname,country,dob,date_of_issue,date_of_expiry,DL_number


file_name = 'F:/Data_Science/Datasets/DL/7.jpg'
dir =  os.path.splitext(file_name)[0]
image = cv2.imread(file_name)

image_resize = align_resize(image)
#name_image = name_image[60:140,180:400]
#dob_country_image = new_image[141:170,180:500]
crop1 = image_resize[60:250, 180:600]
#crop2 = image_resize[320:365, 180:600]
#cv2.imshow('Crop',crop_image)
#cv2.waitKey(0)
crop1_clean = clean_image(crop1)
#addr_image = clean_image(addr_image)
#cv2.imshow('crop', crop1_clean)
#cv2.waitKey(0)
text = pytesseract.image_to_string(crop1_clean)
print(text)
#lastname,firstname,country,dob,date_of_issue,date_of_expiry,DL_number = text_extract(text)

#print('LastName - {}\nFirstName - {}\nCountry - {}\nDate of Birth - {}\nDate of Issue - {}\nDate of Expiry - {}\nDL Number - {}'.format(lastname,firstname,country,dob,date_of_issue,date_of_expiry,DL_number))
