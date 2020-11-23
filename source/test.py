#importing the required libraries
import numpy as np
from skimage.io import imread, imshow
from skimage.filters import prewitt_h,prewitt_v
import matplotlib.pyplot as plt

#reading the image 
image = imread('./train/img_0.jpeg',as_gray=True)

#calculating horizontal edges using prewitt kernel
edges_prewitt_horizontal = prewitt_h(image)
#calculating vertical edges using prewitt kernel
edges_prewitt_vertical = prewitt_v(image)

from misc_tools import show

show(edges_prewitt_vertical)

imshow(edges_prewitt_vertical, cmap='gray')