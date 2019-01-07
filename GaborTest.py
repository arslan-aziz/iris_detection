import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as img
import PIL.Image as pil

#load image into ndarray and display
load1=pil.open('iris_test.jpg')
print(load1.format,load1.size,load1.mode)
dims=load1.size
im1=img.pil_to_array(load1)
print(type(im1),im1.shape)
plt.imshow(im1)
plt.show()

#convert to grayscale
grayArray=np.array([.299,.587,.114])
im1gray=np.dot(im1[...,:3],grayArray)
print(type(im1gray),im1gray.shape)
plt.imshow(im1gray)
plt.show()

#segment iris and zero background

