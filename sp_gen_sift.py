import numpy as np
from sp_load_image import *
from scipy.misc import imresize
from sp_dense_sift import *
from set_default import *

def sp_gen_sift(imageFName,params, i):

    features = []
    I = sp_load_image(imageFName, i)
    hgt, wid = np.shape(I)
    print("Loaded %s: original size %d x %d" % (imageFName, wid, hgt))
    if min(hgt, wid) > params.maxImageSize:
        I = imresize(I, params.maxImageSize / min(hgt, wid))
        print("Loaded %s: original size %d x %d, resizing to %d x %d " % (imageFName, wid, hgt, np.shape(I)[1],
                                                                            np.shape(I)[0]))
        hgt, wid = np.shape(I)

    print("load dense_sift")
    sift_Arr, gridX, gridY = sp_dense_sift(I, params.gridSpacing, params.patchSize)
    sift_Arr = np.reshape(sift_Arr, (np.shape(sift_Arr)[0]*np.shape(sift_Arr)[1], np.shape(sift_Arr)[2]))

    print("sift_arr的维度:%d * %d \n" % (np.shape(sift_Arr)[0], np.shape(sift_Arr)[1])) # [1386, 128]
    params.siftdata.append(sift_Arr)
    features.append(sift_Arr)
    features.append(np.reshape(gridX, (np.shape(gridX)[0] * np.shape(gridX)[1], 1)))
    features.append(np.reshape(gridY, (np.shape(gridY)[0] * np.shape(gridY)[1], 1)))
    features.append(wid)
    features.append(hgt)
    # features.data = sift_Arr
    # features.x = np.reshape(gridX, (np.shape(gridX)[0] * np.shape(gridX)[1], 1))
    # features.y = np.reshape(gridY, (np.shape(gridY)[0] * np.shape(gridY)[1], 1))
    #features.x = gridX
    #features.y = gridY
    # features.wid = wid
    # features.hgt = hgt

    return features

