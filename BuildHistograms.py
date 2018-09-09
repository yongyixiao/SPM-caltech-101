import numpy as np
from scipy.io import loadmat, savemat
from os.path import join
from set_default import *
from sp_dist2 import *
from scipy import histogram

"""
find texton labels of patches and compute texton histograms of all images。
For each image the set of sift descriptors is loaded and then each
descriptor is labeled with its texton label. Then the global histogram
is calculated for the image. If you wish to just use the Bag of Features
image descriptor you can stop at this step, H_all is the histogram or
Bag of Features descriptor for all input images.
"""
def BuildHistograms(imagefiles, dataDir, params, canSkip):

    print("Building Histograms\n\n")
    vocalpath = join(dataDir, 'vocab.py.mat')
    dictionary = loadmat(vocalpath)['dictionary']
    print("Load texton dictionary:%d texton" % params.dictionarySize)

    # compute texton labels of patches and whole-image histograms
    H_all = []

    for f in range(len(imagefiles)):
        texton_ind = []
        # load sift descriptors
        feature_path = join(dataDir, str(f)+'_features.mat')
        features = loadmat(feature_path)['features']
        siftdata = features[0, 0]
        ndata = np.shape(features[0, 0])[0]
        # find texton indices and compute histogram
        texton_ind_data = np.zeros((ndata, 1))
        #features.x = np.reshape(gridX, (np.shape(gridX)[0] * np.shape(gridX)[1], 1))
        texton_ind_x = features[0,1]
        texton_ind_y = features[0,2]
        texton_ind_wid = features[0,3]
        texton_ind_hgt = features[0,4]

        batchsize = 200000
        if ndata < batchsize:
            dist_mat = sp_dist2(siftdata, dictionary)
            min_dist = np.min(dist_mat, 1)
            min_ind = np.argmin(dist_mat, 1)
            min_ind = np.reshape(min_ind, (np.shape(min_ind)[0], 1))
            texton_ind_data = min_ind
        else:
            for j in range(0, batchsize+1,ndata):
                lo = j
                hi = min(j+batchsize, ndata)
                dist_mat = sp_dist2(siftdata[lo:hi, :], dictionary)
                min_dist = np.min(dist_mat, 1)
                min_ind = np.argmin(dist_mat, 1)
                min_ind = np.reshape(min_ind, (np.shape(min_ind)[0], 1))
                texton_ind_data[lo:hi, :] = min_ind

        H = histogram(texton_ind_data, bins=range(params.dictionarySize + 1))[0]
        if f == 0:
            H_all = H
        else:
            H_all = np.vstack((H_all, H))
        texton_ind.append(texton_ind_data)
        texton_ind.append(texton_ind_x)
        texton_ind.append(texton_ind_y)
        texton_ind.append(texton_ind_wid)
        texton_ind.append(texton_ind_hgt)
        texton_path = join(dataDir, str(f)+'_texton.mat')
        savemat(texton_path, {'texton_ind': texton_ind})

    hists = H_all  # [num. 200]
    hists_path = join(dataDir, 'hists.py.mat')
    savemat(hists_path, {'hists': hists})
    return H_all


