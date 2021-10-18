import cv2
import numpy as np
import random
from scipy import ndimage

src = cv2.imread("/home/olga/MyProjects/Polikek/Genetic Algoritms/kurs/images/flower.jpg", -1)
brush = cv2.imread("/home/olga/MyProjects/Polikek/Genetic Algoritms/kurs/images/brush.png", -1)

class A:
    def __init__(self):
        self.a = 5
        self.b = 6

    def initialise(self):
        return(self.a,self.b)
a = A()
print(a.initialise())





