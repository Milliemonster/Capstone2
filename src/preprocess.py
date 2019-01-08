import numpy as np
from skimage.color import rgb2grey

def preprocess_img(img):
    img = rgb2grey(img)
    
    img = img.astype(np.float32) / 255.0
    img -= 0.5
    return img * 2
