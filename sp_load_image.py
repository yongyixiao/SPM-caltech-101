from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def sp_load_image(image_fname, i):
    I = Image.open(image_fname)
    if len(np.shape(I)) == 3:
        I = I.convert('L')
        I = np.double(np.array(I))
    else:
        I = np.double(np.array(I))
    print("原始图片%d：%d * %d" % (i, np.shape(I)[1],np.shape(I)[0]))
    return I

