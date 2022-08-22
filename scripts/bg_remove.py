from PIL import Image as im
from tqdm import tqdm
import numpy as np
import cv2
import glob

#Retriving all image names and it's path with .jpg extension from given directory path in imageNames list
imageNames = glob.glob(r"..\data\dataset_origin\*.jpg")

#Count variable to show the progress of image resized
count=0

#Creating for loop to take one image from imageNames list and resize
for i in tqdm(imageNames):
    #opening image for editing
    img = im.open(i)
    img = np.array(img)
    
    #using resize() to resize image
    scale_percent = 30
    width = int(img.shape[1]*scale_percent/100)
    height = int(img.shape[0]*scale_percent/100)
    dim = (width, height)
    img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    
    # grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # canny
    #canned = cv2.Canny(gray, 0, 100);
    
    canned = cv2.Canny(gray, 100/3, 150); 

    # dilate to close holes in lines
    kernel = np.ones((3,3),np.uint8)
    mask = cv2.dilate(canned, kernel, iterations = 1);

    # find contours
    # Opencv 3.4, if using a different major version (4.0 or 2.0), remove the first underscore
    _, contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE);

    # find the biggest contour
    biggest_cntr = None;
    biggest_area = 0;
    for contour in contours:
        area = cv2.contourArea(contour);
        if area > biggest_area:
            biggest_area = area;
            biggest_cntr = contour;

    # draw contours
    crop_mask = np.zeros_like(mask);
    cv2.drawContours(crop_mask, [biggest_cntr], -1, (255), -1);

    # opening + median blur to smooth jaggies
    crop_mask = cv2.erode(crop_mask, kernel, iterations = 5);
    crop_mask = cv2.dilate(crop_mask, kernel, iterations = 5);
    crop_mask = cv2.medianBlur(crop_mask, 21);

    # crop image
    crop = np.zeros_like(img);
    crop[crop_mask == 255] = img[crop_mask == 255];    
    
    img = im.fromarray(crop)
    
    #save() to save image at given path and count is the name of image eg. first image name will be 0.jpg
    img.save(r"..\data\train\\"+str(count)+".jpg") 
    
    #incrementing count value
    count+=1

print(str(count)+" images background is removed!")