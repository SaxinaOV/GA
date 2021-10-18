import cv2
import numpy as np
import random
from scipy import ndimage
from skimage.metrics import structural_similarity as ssim

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

def brush_stroke(image, brush, x_pos, y_pos):
    img2gray = cv2.cvtColor(brush,cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
    mask_inv = cv2.bitwise_not(mask)
    #mask = np.zeros(brush.shape[:2], dtype=np.uint8)
    #mask_inv = cv2.bitwise_not(mask)
    #brush = ndimage.rotate(brush, angle)
    #mask = ndimage.rotate(mask, angle)
    #mask_inv = ndimage.rotate(mask_inv, angle)
    roi = image[x_pos:brush.shape[0]+x_pos, y_pos:brush.shape[1]+y_pos]
    try:
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
    except:
        img1_bg = cv2.bitwise_and(roi,roi,mask = mask_inv[0:roi.shape[0], 0:roi.shape[1]])
    img2_fg = cv2.bitwise_and(brush, brush, mask = mask)
    
    b_channel = img2_fg[:,:,0]
    g_channel = img2_fg[:,:,1]
    r_channel = img2_fg[:,:,2]
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 100
    img2_fg = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    try:
        dst = cv2.add(img1_bg,img2_fg)
    except:
        dst = cv2.add(img1_bg,img2_fg[0:img1_bg.shape[0], 0:img1_bg.shape[1]])
    alpha = random.uniform(0.1, 1)
    dst = cv2.addWeighted(dst, alpha, roi, 1-alpha, 0)
    image[x_pos:brush.shape[0]+x_pos, y_pos:brush.shape[1]+y_pos] = dst
    return image

ori_image = cv2.imread("/home/olga/MyProjects/Polikek/Genetic Algoritms/kurs/images/flower.jpg")
ori_brush = cv2.imread("/home/olga/MyProjects/Polikek/Genetic Algoritms/kurs/images/brush.png")

image = ori_image
brush_cp = ori_brush

b_channel, g_channel, r_channel = cv2.split(image)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 0
image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
b_channel, g_channel, r_channel = cv2.split(ori_image)
alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 0
ori_image = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

""" imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(image, contours, -1, (255,255,255), 5) """

""" 
cv2.imshow('',image)
cv2.waitKey(0)
cv2.destroyAllWindows() """

width = int(brush_cp.shape[1] * 0.06)
height = int(brush_cp.shape[0] * 0.06)
dsize = (width, height)
brush_cp = cv2.resize(brush_cp, dsize)
brush_cp[np.where((brush_cp!=[255,255,255]).all(axis=2))] = (255, 0, 0)

for i in range(80):
    brush = brush_cp.copy()
    #angle = random.choice((0, 90, 180, 270))
    posX = random.randint(0, image.shape[0]-brush.shape[0])
    posY = random.randint(0, image.shape[1]-brush.shape[1])
    imgray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    img_gradient = _imgGradient(imgray)
    imgMag = img_gradient[0]
    imgAngles = img_gradient[1]
    localMag = imgMag[posX][posY]
    localAngle = imgAngles[posX][posY] + 90 #perpendicular to the dir
    rotation = random.randrange(-180, 180)*(1-localMag) + localAngle
    brush = ndimage.rotate(brush, rotation)
    #angle = random.randint(0,359)
    
    """ cv2.imshow('',brush)
    cv2.waitKey(0)
    cv2.destroyAllWindows() """
    random_blue = random.randint(0,255)
    random_green = random.randint(0,255)
    random_red = random.randint(0,255)
    color = (random_blue, random_green, random_red)
    brush[np.where((brush==(0,0,0)).all(axis=2))] = (255,255,255)
    
    brush[np.where((brush != (255,255,255)).any(axis=2))] = color
    image = brush_stroke(image, brush, posX, posY)

cv2.imshow('res',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

s = ssim(image, ori_image, full=True, multichannel=True)




