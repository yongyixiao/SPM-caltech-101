import numpy as np
from sp_kmeans import *
from scipy.io import loadmat, savemat
from os.path import join
from scipy.cluster.vq import vq, kmeans, whiten
"""

First, all of the sift descriptors are loaded for a random set of images. The
size of this set is determined by numTextonImages. Then k-means is run
on all the descriptors to find N centers, where N is specified by

params.dictionarySize: size of descriptor dictionary (200 has been found to be a good size)
params.numTextonImages: number of images to be used to create the histogram bins
dictionarySize.
"""
def CalculateDictionary(imagefiles,dataDir, params, canSkip):

    print("'Building Dictionary")
    reduce_flag = 1
    ndata_max = 200000

    if params.numTextonImages > len(imagefiles):
        params.numTextonImages = len(imagefiles)

    #if params.numTextonImages > len(imagefiles):
    #    params.numTextonImages = len(imagefiles)
    # load all SIFT descriptors
    sift_all = []
    # 返回一个随机排列
    R = np.random.permutation(len(imagefiles))
    # R = np.random.permutation(len(imagefiles))
    for f in range(params.numTextonImages):
        image = R[f]
        #ndata = np.shape(params.siftdata[image])[0]
        feature_path = join(dataDir, str(f)+'_features.mat')
        features = loadmat(feature_path)['features']
        #data2add = params.siftdata[image]
        data2add = features[0,0]
        if np.shape(data2add)[0] > (ndata_max / params.numTextonImages):
            p = np.random.permutation(np.shape(data2add)[0])
            data2add_new = np.zeros((int(ndata_max / params.numTextonImages), np.shape(data2add)[1]))
            for i in range(int(ndata_max / params.numTextonImages)):
                data2add_new[i,:] = data2add[p[i], :]
            data2add = data2add_new
        if f == 0:
            sift_all = data2add
        else:
            sift_all = np.vstack((sift_all, data2add))
        #print("Loaded %s, %d descriptors,total %d so far" % (imagefiles[f], ndata, np.shape(sift_all)[0]))

    print("\n total descriptors loaded: %d"  % (np.shape(sift_all)[0]))

    ndata = np.shape(sift_all)[0]
    if reduce_flag > 0 and ndata > ndata_max:
        print("reducing to %d descriptors" % ndata_max)
        p = np.random.permutation(ndata)
        sift_all = sift_all[0:ndata_max, :]
        for i in range(ndata_max):
            sift_all[i, :] = sift_all[p[i], :]

    # perform clustering
    options = np.zeros((1, 14))
    options[0, 0] = 1 # display
    options[0, 1] = 1
    options[0, 2] = 0.1 # precision
    options[0, 4] = 1 # initialization
    options[0, 13] = 150 # maximum iterations

    centers = np.zeros((params.dictionarySize, np.shape(sift_all)[1]))

    # run k-means
    print(np.shape(sift_all))
    print("Running k-means\n")
    dictionary = kmeans(sift_all, 200, iter=100)[0]
    print(np.shape(dictionary))
    #dictionary = sp_kmeans(centers, sift_all, options)
    print("Saving texton dictionary")
    vocalpath = join(dataDir, 'vocab.py.mat')
    savemat(vocalpath, {'dictionary': dictionary})






#
# sif = []
# data = np.random.rand(3,3)
# sif = data
# sif = np.vstack((sif, data))
# print(sif)
