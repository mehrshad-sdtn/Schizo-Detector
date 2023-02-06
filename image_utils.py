import cv2
import os
import numpy as np

img_height = 224
img_width = 224

def load_image(filepath, dims):
   image = cv2.cvtColor(
      cv2.imread(filepath), cv2.COLOR_BGR2RGB)
   image = cv2.resize(image, dims)
   image = image.astype('float32') / 255.0
   return image


def image_arrays_from_directory(directory):
  arrays = []
  for name in os.listdir(directory):
      arrays.append( load_image(os.path.join(directory, name), (img_height, img_width)) )
  return np.array(arrays)