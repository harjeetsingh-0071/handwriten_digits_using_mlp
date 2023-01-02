from PIL import Image
from numpy import asarray
import numpy as np
image = Image.open('C:/Users/harik/Pictures/test.png')
data = asarray(image)
mask = np.full(data.shape,255)
data = data - mask
data = data.astype(np.uint8)
data = data.flatten()
print(data)
print(data.shape)