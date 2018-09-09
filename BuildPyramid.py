import numpy as np
from set_default import *
from GenerateSiftDescriptors import *
from CalculateDictionary import *
from BuildHistograms import *
from CompilePyramid import *

"""
To build the pyramid this function first extracts the sift descriptors
for each image. It then calculates the centers of the bins for the
dictionary. Each sift descriptor is given a texton label corresponding
to the appropriate dictionary bin. Finally the spatial pyramid
is generated from these label lists.

params.gridSpacing: the space between dense sift samples
params.patchSize: the size of each patch for the sift descriptor
params.maxImageSize: the max image size. If the image is larger it will be resampeled.
params.dictionarySize: size of descriptor dictionary (200 has been found to be a good size)
params.numTextonImages: number of images to be used to create the histogram bins
params.pyramidLevels: number of levels of the pyramid to build
"""

def BuildPyramid(imagefiles, dataDir):
    params = Init_param()
    canSkip = 1
    saveSift = 0

    if saveSift:
        GenerateSiftDescriptors(imagefiles, dataDir, params, canSkip)
    CalculateDictionary(imagefiles, dataDir, params, canSkip)
    H_all = BuildHistograms(imagefiles, dataDir, params, canSkip)
    pyramid_all = CompilePyramid(imagefiles, dataDir, params, canSkip)
    print("H_all 的维度：")
    print(np.shape(H_all))
    return pyramid_all


