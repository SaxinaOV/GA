import cv2 
import numpy as np
import random

def calc_sampling_mask(img_grey, blur_percent):
    img = np.copy(img_grey)
    # Calculate gradient 
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    # Python Calculate gradient magnitude and direction ( in degrees ) 
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    #calculate blur level
    w = img.shape[0] * blur_percent
    if w > 1:
        mag = cv2.GaussianBlur(mag,(0,0), w, cv2.BORDER_DEFAULT)
    #ensure range from 0-255 (mostly for visual debugging, since in sampling we will renormalize it anyway)
    scale = 255.0/mag.max()
    return mag*scale

def create_sampling_mask(s, stages):
    percent = 0.2
    start_stage = int(stages*percent)
    sampling_mask = None
    if s >= start_stage:
        t = (1.0 - (s-start_stage)/max(stages-start_stage-1,1)) * 0.25 + 0.005
        sampling_mask = calc_sampling_mask(t)
    return sampling_mask

def util_sample_from_img(img):
    #possible positions to sample
    pos = np.indices(dimensions=img.shape)
    pos = pos.reshape(2, pos.shape[1]*pos.shape[2])
    img_flat = np.clip(img.flatten() / img.flatten().sum(), 0.0, 1.0)
    return pos[:, np.random.choice(np.arange(pos.shape[1]), 1, p=img_flat)]

def _imgGradient(img):
        #convert to 0 to 1 float representation
        img = np.float32(img) / 255.0 
        # Calculate gradient 
        gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
        gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
        # Python Calculate gradient magnitude and direction ( in degrees ) 
        mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
        #normalize magnitudes
        mag /= np.max(mag)
        #lower contrast
        mag = np.power(mag, 0.3)
        return mag, angle

im = cv2.imread("/home/olga/MyProjects/Polikek/Genetic Algoritms/kurs/images/flower.jpg")
posX = random.randint(0, im.shape[0]-brush.shape[0])
posY = random.randint(0, im.shape[1]-brush.shape[1])
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img_gradient = _imgGradient(imgray)
imgMag = img_gradient[0]
imgAngles = img_gradient[1]
localMag = imgMag[posY][posX]
localAngle = imgAngles[posY][posX] + 90 #perpendicular to the dir
rotation = random.randrange(-180, 180)*(1-localMag) + localAngle

sampling_mask = create_sampling_mask()
pos = util_sample_from_img(sampling_mask)
posY = pos[0][0]
posX = pos[1][0]

"""
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
img = np.float32(imgray) / 255.0 
gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
mag /= np.max(mag)
mag = np.power(mag, 0.3)



im2 = im.copy()
empty = np.zeros((im.shape[0],im.shape[1],3), np.uint8)
imgray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
ret, thresh = cv.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv.findContours(thresh, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
cv.drawContours(im2, contours, -1, (255,255,255), 3)

new_im = abs(im2 - im)
laplacian = cv.Sobel(imgray,cv.CV_64F,1,1,ksize=5)

cv.imshow('',laplacian)
cv.waitKey(0)
cv.destroyAllWindows() """