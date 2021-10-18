import random 
import numpy as np
import cv2
from scipy import ndimage

class GA:
    def __init__(self, img, size):
        self.img = img
        self.population_size = size
        self.brushstroke_count = 100

    def generate_population(self):
        self.population = []
        for i in self.population_size:
            dna = DNA(self.brushstroke_count)
            population.append(dna.init_random())
        return self.population

    def fitness_score(self, img2):
        err = np.sum((self.img.astype("float") - img2.astype("float")) ** 2)
        err /= float(self.img.shape[0] * img2.shape[1])
        return err

    def selection(self):
        pass

    def crossover(self):
        pass

    def mutation(self):
        pass

    def draw(self):
        pass

    def image_gradient(self):
        img = np.float32(self.img) / 255.0 
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


class DNA:
    def __init__(self, size):
        self.size = size
        self.dna = []

    def init_random(self):
        for i in self.size:
            gene = Gene()
            self.dna.append(gene.init_random())
        return self.dna

    def draw(self, img):
        img2 = img
        for g in dna:
            img2 = g.draw(img)
        return img2

class Gene:
    def __init__(self, bound, img_gradient, brushstrokes_range):
        self.minSize = brushstrokes_range[0] 
        self.maxSize = brushstrokes_range[1] 
        self.bound = bound
        self.imgMag = img_gradient[0]
        self.imgAngles = img_gradient[1]
        
    def init_random(self):
        self.brushstroke_size = random.randomint(self.minSize, self.maxSize)
        self.posX = random.randint(0, self.bound[0])
        self.posY = random.randint(0, self.bound[1])
        localMag = self.imgMag[posX][posY]
        localAngle = self.imgAngles[posX][posY] + 90
        self.rotation = random.randrange(-180, 180)*(1-localMag) + localAngle
        #self.color = (random.randint(0,179),random.randint(0,255),random.randint(0,255))
        self.color = (random.randint(0,255),random.randint(0,255),random.randint(0,255))
        self.opacity = random.uniform(0.1, 1)
        return (self.posX, self.posY, self.brushstroke_size, self.rotation, self.color, self.opacity)

    def draw(self, image, brush, y_pos, angle, opacity):
        img2gray = cv2.cvtColor(brush,cv2.COLOR_BGR2GRAY)
        brush = ndimage.rotate(brush, rotation)
        ret, mask = cv2.threshold(img2gray, 200, 255, cv2.THRESH_BINARY_INV)
        mask_inv = cv2.bitwise_not(mask)
        roi = image[self.posX:brush.shape[0]+self.posX, y_pos:brush.shape[1]+y_pos]
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
        dst = cv2.addWeighted(dst, opacity, roi, 1-opacity, 0)
        image[self.posX:brush.shape[0]+self.posX, y_pos:brush.shape[1]+y_pos] = dst
        return image
            

    


