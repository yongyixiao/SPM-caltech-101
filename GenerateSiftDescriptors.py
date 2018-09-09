import numpy as np
from sp_gen_sift import *
from os.path import join
from scipy.io import loadmat, savemat

"""
Generate the dense grid of sift descriptors for each image
"""
def GenerateSiftDescriptors(imagefiles,dataDir, params, canSkip):

    print('Building Sift Descriptors\n\n')
    for i in range(len(imagefiles)):
        features = sp_gen_sift(imagefiles[i], params, i)
        feature_path = join(dataDir, str(i)+'_features.mat')
        savemat(feature_path, {'features':features})



    return params

