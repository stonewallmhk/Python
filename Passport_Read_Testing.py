from PIL import Image
import pytesseract
import argparse
import cv2
import os
from datetime import datetime
from pathlib import Path
import numpy as np
import imutils

def extractMRZ(passport):

    # initialize a rectangular and square structuring kernel
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))

    path = os.path.splitext(passport)[0]
    image = cv2.imread(passport)
    
    #image = passport
    image = imutils.resize(image, height=600)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


    # smooth the image using a 3x3 Gaussian, then apply the blackhat
    # morphological operator to find dark regions on a light background
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)

    # compute the Scharr gradient of the blackhat image and scale the
    # result into the range [0, 255]
    gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradX = np.absolute(gradX)
    (minVal, maxVal) = (np.min(gradX), np.max(gradX))
    gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")

    # apply a closing operation using the rectangular kernel to close
    # gaps in between letters -- then apply Otsu's thresholding method
    gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, rectKernel)
    thresh = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    # perform another closing operation, this time using the square
    # kernel to close gaps between lines of the MRZ, then perform a
    # series of erosions to break apart connected components
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, sqKernel)
    thresh = cv2.erode(thresh, None, iterations=4)

    # during thresholding, it's possible that border pixels were
    # included in the thresholding, so let's set 5% of the left and
    # right borders to zero
    p = int(image.shape[1] * 0.05)
    thresh[:, 0:p] = 0
    thresh[:, image.shape[1] - p:] = 0

    # find contours in the thresholded image and sort them by their size
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

    #loop over contours
    for c in cnts:
    # compute the bounding box of the contour and use the contour to
    # compute the aspect ratio and coverage ratio of the bounding box
    # width to the width of the image
        (x, y, w, h) = cv2.boundingRect(c)
        ar = w / float(h)
        crWidth = w / float(gray.shape[1])

    # check to see if the aspect ratio and coverage width are within
    # acceptable criteria
        if ar > 5 and crWidth > 0.75:
            # pad the bounding box since we applied erosions and now need
            # to re-grow it
            pX = int((x + w) * 0.03)
            pY = int((y + h) * 0.03)
            (x, y) = (x - pX, y - pY)
            (w, h) = (w + (pX * 2), h + (pY * 2))

            # extract the ROI from the image and draw a bounding box
            # surrounding the MRZ
            roi = image[y:y + h, x:x + w].copy()
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            break

    # roi is ready now

    #preprocess image for tesseract
    img_clean = cv2.resize(roi, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    img_clean = cv2.bilateralFilter(img_clean, 9, 75, 75)

    #Convert to gray
    gray = cv2.cvtColor(img_clean, cv2.COLOR_BGR2GRAY)

    #Remove Noise by dilation and erosion
    kernel = np.ones((1,1), np.uint8)
    gray = cv2.dilate(gray, kernel, iterations=1)
    gray = cv2.erode(gray, kernel, iterations=1)

    #Thresholding
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #gray = cv2.adaptiveThreshold(cv2.GaussianBlur(gray, (5, 5), 0), 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    #save the image
    '''
    roi = path + "_roi.jpg"
    cv2.imwrite(roi, gray)
    #config = '--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789>< -c load_system_dawg=F -c load_freq_dawg=F'
    text = pytesseract.image_to_string(Image.open(roi), lang = 'ocrb')
    print(text)'''
    ocr = pytesseract.image_to_string(gray, lang = 'ocrb')
    #print(ocr)
    return ocr

#def textparse(text, input_loc, output_loc):
def textparse(text, input_loc, output_loc):
    a,b = text.splitlines()
    doctype = a[0:1]
    country = a[2:5]
    names = a[5:44].split('<<',1)
    if len(names) < 2:
        names += ['']
    lastname, givenname = names
    lastname = lastname.replace('<', ' ')
    givenname = givenname.split('<<')[0]
    givenname = givenname.replace('<', ' ')
    docnumber = b[0:9]
    docnumber = docnumber.replace('<', ' ').strip()
    checkdigit1 = b[9]
    nationality = b[10:13]
    dob = b[13:19]
    if dob.isdigit():
        dob = datetime.strptime(dob, '%y%m%d').strftime('%d-%m-%Y')
    else:
        dob = ''
    #checkdigit2 = text[64:65]
    sex = b[20]
    if sex == 'M':
        sex = 'MALE'
    elif sex == 'F':
        sex = 'FEMALE'
    else:
        sex = ''
    doe = b[21:27]
    if doe.isdigit():
        doe = datetime.strptime(doe, '%y%m%d').strftime('%d-%m-%Y')
    else:
        doe = ''
    personalnum = b[28:42]
    check_personalnum = b[42]
    checkdigit5 = b[43]
        
    #print('IssuingCountry: {}\nLastName: {}\nGivenName: {}\nDocumentNumber: {}\nNationality: {}\nDate of Birth: {}\nSex: {}\nDate of Expiry: {}'.format(country,lastname, givenname,docnumber,nationality,dob, sex,doe))
    
    output_name = output_loc + Path(input_loc).stem + '.txt'
    f = open(output_name, 'w+')
    f.write('IssuingCountry: {}\nLastName: {}\nGivenName: {}\nDocumentNumber: {}\nNationality: {}\nDate of Birth: {}\nSex: {}\nDate of Expiry: {}'.format(country, lastname, givenname, docnumber,nationality,dob,sex,doe))
    f.close()
        
    #print('IssuingCountry: {}\nLastName: {}\nGivenName: {}\nDocumentNumber: {}\nNationality: {}\nDate of Birth: {}\nSex: {}\nDate of Expiry: {}'.format(country,lastname, givenname,docnumber,nationality,dob, sex,doe))

#txt = extractMRZ('C:/Hari Docs/Dataset/Passports_V2/1-001.jpg')
#print(textparse(txt))

'''
#contruct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required = True,
                help = "Path to Input Image")
args = vars(ap.parse_args())
file = args["input"]
'''
input_path = input("Enter Filename location: ")
input_path = input_path.replace('\\', '/')
output_path = input("Enter Directory Location for output file: ")
output_path = output_path.replace('\\', '/')
if os.path.isfile(input_path):
    try:
        txt = extractMRZ(input_path)
        #print(txt)
        textparse(txt,input_path, output_path)
    except:
        print('File Error or Incorrect Image Captured...')
elif os.path.isdir(input_path):
    err_file = open(output_path + 'Error_Files.txt', 'w+')
    err_file.write('Below are the list of Error File could not be processed by the program: \n')
    err_file.close()
    err_list = []
    for file in os.listdir(input_path):
        print('Processing file: {}'.format(file))
        try:
            txt = extractMRZ(input_path + file)
            textparse(txt, input_path + file, output_path)
        except:
            err_list.append(file)

#try:
#    err_list
#    print('List of error files: {}'.format(err_list))
#except: 
#    pass
    
try:
    err_list
    print('List of error files: {}'.format(err_list))
    err_file = open(output_path + 'Error_Files.txt', 'a+')
    for i in err_list:
        err_file.write('\n{}'.format(i))
    err_file.close()
except:
    pass
print('Output files created..')  
