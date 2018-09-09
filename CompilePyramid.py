import numpy as np
from scipy.io import loadmat, savemat
from os.path import join
from scipy import histogram

"""
Generate the pyramid from the texton lablels.
For each image the texton labels are loaded. Then the histograms are
calculated for the finest level. The rest of the pyramid levels are
generated by combining the histograms of the higher level.
"""
def CompilePyramid(imagefiles, dataDir, params, canSkip):

    print("Building Spatial Pyramid\n\n")
    binsHigh = 2 ** (params.pyramidLevels - 1) # 4

    a = (2 ** (np.arange(params.pyramidLevels))) ** 2
    pyramid_all = np.zeros((len(imagefiles), params.dictionarySize * np.sum(a)))

    for f in range(len(imagefiles)):
        # load texton indices
        texton_path = join(dataDir, str(f)+'_texton.mat')
        texton_ind = loadmat(texton_path)['texton_ind']
        # get width and height of input image
        wid = texton_ind[0, 3]
        hgt = texton_ind[0, 4]
        #print("loaded %s:wid %d, hgt %d" % (imagefiles[f], wid, hgt))

        # compute histogram at the finest level
        pyramid_cell = []
        for i in range(params.pyramidLevels):
            pyramid_cell.append([])
        pyramid_cell[0] = np.zeros((binsHigh, binsHigh, params.dictionarySize)) #[4,4,200]

        for i in range(binsHigh):
            for j in range(binsHigh):

                # find the coordinates of the current bin
                x_lo = np.floor(wid / binsHigh * i)
                x_hi = np.floor(wid / binsHigh * (i+1))
                y_lo = np.floor(hgt / binsHigh * j)
                y_hi = np.floor(hgt / binsHigh * (j+1))
                xx = texton_ind[0, 1]
                yy = texton_ind[0, 2]
                indice = []
                for k in range(np.shape(xx)[0]):
                    if xx[k, 0] > x_lo and xx[k, 0] <= x_hi and yy[k, 0] > y_lo and yy[k, 0] <= y_hi:
                        indice.append(k)
                texton_patch = np.zeros((len(indice), 1))
                for k in range(len(indice)):
                    texton_patch[k, 0] = texton_ind[0,0][indice[k], 0]

                # make histogram of features in bin
                pyramid_cell[0][i, j, :] = histogram(texton_patch, bins=range(params.dictionarySize + 1))[0] / np.shape(texton_ind[0,0])[0]

        # compute histograms at the coarser levels
        num_bins = binsHigh // 2 # 2
        for lev in range(1,params.pyramidLevels):
            pyramid_cell[lev] = np.zeros((num_bins, num_bins, params.dictionarySize))
            for i in range(num_bins):
                for j in range(num_bins):
                    pyramid_cell[lev][i, j, :] = pyramid_cell[lev-1][2*i, 2*j, :] + pyramid_cell[lev-1][2*i+1, 2*j, :] +\
                                                 pyramid_cell[lev-1][2*i, 2*j+1, :] + pyramid_cell[lev-1][2*i+1,2*j+1,:]
            num_bins = num_bins // 2

        # stack all the histograms with appropriate weights
        pyramid = []
        for lev in range(params.pyramidLevels - 1):
            pyramid_cell[lev] = np.reshape(pyramid_cell[lev], (np.shape(pyramid_cell[lev])[0]*np.shape(pyramid_cell[lev])[1]*
                                                               np.shape(pyramid_cell[lev])[2], 1)).T
            if lev == 0:
                pyramid = pyramid_cell[lev] * (2**(-lev-1))
            else:
                pyramid = np.hstack((pyramid, pyramid_cell[lev] * (2**(-lev-1))))
        pyramid_cell[params.pyramidLevels-1] = np.reshape(pyramid_cell[params.pyramidLevels-1], (np.shape(pyramid_cell[params.pyramidLevels-1])[0]*np.shape(pyramid_cell[params.pyramidLevels-1])[1]*
                                                               np.shape(pyramid_cell[params.pyramidLevels-1])[2], 1)).T
        pyramid = np.hstack((pyramid, pyramid_cell[params.pyramidLevels-1] * (2**(1-params.pyramidLevels))))

        pyramid_all[f, :] = pyramid

    pyramid_path = join(dataDir,'pyramid.mat')
    savemat(pyramid_path, {'pyramid_all': pyramid_all})
    print("pyramid 的维度：%d * %d" %(np.shape(pyramid_all)[0], np.shape(pyramid_all)[1]))
    return pyramid_all

# a = np.random.rand(10,1)
# b = np.random.rand(10,1)
# ind = []
# for i in range(10):
#     if a[i,0] > 0 and b[i,0]<0.5:
#         ind.append(i)
# print(ind)