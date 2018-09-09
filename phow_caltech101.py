from os.path import exists, isdir, basename, join, splitext
from os import makedirs
from glob import glob
from random import sample, seed
from scipy import ones, mod, arange, array, where, ndarray, hstack, linspace, histogram, vstack, amax, amin
from scipy.misc import imread, imresize
from scipy.cluster.vq import vq
import numpy
#from vl_phow import vl_phow
#from vlfeat import vl_ikmeans
from scipy.io import loadmat, savemat
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import pylab as pl
from datetime import datetime
from sklearn.kernel_approximation import AdditiveChi2Sampler
from pickle import dump, load
import numpy as np
import matplotlib.pyplot as plt

IDENTIFIER = '18.08.24'
SAVETODISC = False
FEATUREMAP = True
OVERWRITE = False  # DON'T load mat files genereated with a different seed!!!
SAMPLE_SEED = 42
TINYPROBLEM = False
VERBOSE = True  # set to 'SVM' if you want to get the svm output
MULTIPROCESSING = False

class Configuration(object):
    def __init__(self, identifier=''):
        #self.calDir = '.\dataset\\101_ObjectCategories'
        self.calDir = '/mlspace/xyy/dev_kmean_svm/dataset/101_ObjectCategories'
        self.dataDir = 'tempresults'
        if not exists(self.dataDir):
            makedirs(self.dataDir)
            print ("folder " + self.dataDir + " created")
        self.autoDownloadData = True
        self.numTrain = 30
        self.numTest = 50
        self.imagesperclass = self.numTrain + self.numTest
        self.numClasses = 102
        self.numWords = 600
        self.numSpatialX = [2, 4]
        self.numSpatialY = [2, 4]
        self.quantizer = 'vq'
        self.svm = SVMParameters(C=10)
        self.phowOpts = PHOWOptions(Verbose=False, Sizes=[4, 6, 8, 10], Step=3)
        self.clobber = False
        self.tinyProblem = TINYPROBLEM
        self.prefix = 'baseline'
        self.randSeed = 1
        self.verbose = True
        self.extensions = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
        self.images_for_histogram = 30
        self.numbers_of_features_for_histogram = 100000

        self.vocabPath = join(self.dataDir, identifier + '-vocab.py.mat')
        self.histPath = join(self.dataDir, identifier + '-hists.py.mat')
        self.modelPath = join(self.dataDir, self.prefix + identifier + '-model.py.mat')
        self.resultPath = join(self.dataDir, self.prefix + identifier + '-result')

        if self.tinyProblem:
            print ("Using 'tiny' protocol with different parameters than the .m code")
            # self.prefix = 'tiny'
            # self.numClasses = 5
            # self.images_for_histogram = 10
            # self.numbers_of_features_for_histogram = 1000
            # self.numTrain
            # self.numSpatialX = 2
            # self.numWords = 100
            # self.numTrain = 2
            # self.numTest = 2
            # self.phowOpts = PHOWOptions(Verbose=2, Sizes=7, Step=5)

         # tests and conversions
        self.phowOpts.Sizes = ensure_type_array(self.phowOpts.Sizes)
        self.numSpatialX = ensure_type_array(self.numSpatialX)
        self.numSpatialY = ensure_type_array(self.numSpatialY)
        if (self.numSpatialX != self.numSpatialY).any():
            messageformat = [str(self.numSpatialX), str(self.numSpatialY)]
            message = "(self.numSpatialX != self.numSpatialY), because {0} != {1}".format(*messageformat)
            raise ValueError(message)


def ensure_type_array(data):
    if type(data) is not ndarray:
        if type(data) is list:
            data = array(data)
        else:
            data = array([data])
    return data

def standarizeImage(im):
    im = array(im, 'float32')
    if np.shape(im)[0] > 480:
        resize_factor = 480.0 / np.shape(im)[0]
        im = imresize(im, resize_factor)
    if amax(im) > 1.1:
        im = im / 255.0
    assert((amax(im) > 0.01) & (amax(im) <= 1))
    assert((amin(im) >= 0.00))
    return im



def get_classes(datasetpath, numClasses):
    classes_paths = [files for files in glob(datasetpath + '/*')]
    #classes_paths.sort()
    # basename获取对应路径下文件的名字
    classes = [basename(class_path) for class_path in classes_paths]
    print(len(classes))
    if len(classes) == 0:
        raise ValueError('no classes found')
    if len(classes) < numClasses:
        raise ValueError('conf.numClasses is bigger than the number of folders')
    classes = classes[:numClasses]
    return classes

def get_imgfiles(path, extensions):
    all_files = []
    # extend类似append，添加一个序列
    # splitext文件路径（path）和文件的扩展名（ext)
    all_files.extend([join(path, basename(fname)) for fname in glob(path+'/*')
                      if splitext(fname)[-1].lower() in extensions])
    return all_files

def get_all_images(classes, conf):
    all_images = []
    all_images_class_labels = []
    sel_train = []
    sel_test = []

    # 同时列出数据和数据下标
    k = 0
    p = 0
    for i, imageclass in enumerate(classes):
        path = join(conf.calDir, imageclass)
        extensions = conf.extensions
        imgs = get_imgfiles(path, extensions)
        if len(imgs) == 0:
             raise ValueError('no images for class ' + str(imageclass))
        # 从序列a中随机抽取n个元素，并将n个元素生以list形式返回 imagesperclass:30
        if len(imgs)<=conf.imagesperclass:
            imgs = sample(imgs, len(imgs))
            class_labels = list(i * ones(len(imgs)))
        else:
            imgs = sample(imgs, conf.imagesperclass)
            class_labels = list(i * ones(conf.imagesperclass))
        all_images = all_images + imgs
        #class_labels = list(i * ones(conf.imagesperclass))
        all_images_class_labels = all_images_class_labels + class_labels
        for j in range(k,len(all_images)):
            if (j-p) < conf.numTrain:
                sel_train.append(j)
            else:
                sel_test.append(j)
        k = len(all_images)
        p = k
    all_images_class_labels = array(all_images_class_labels, 'int')
    return all_images, all_images_class_labels,sel_train, sel_test


def create_split(all_images, conf):
    temp = mod(arange(len(all_images)), conf.imagesperclass) < conf.numTrain
    selTrain = where(temp == True)[0]
    selTest = where(temp == False)[0]

    return selTrain, selTest


class SVMParameters(object):
    def __init__(self, C):
        self.C = C


class PHOWOptions(object):
    def __init__(self, Verbose, Sizes, Step):
        self.Verbose = Verbose
        self.Sizes = Sizes
        self.Step = Step

def main_fun():
    seed(SAMPLE_SEED)
    conf = Configuration(IDENTIFIER)
    if VERBOSE:
        print(str(datetime.now()) + ' finished conf')
    classes = get_classes(conf.calDir, conf.numClasses) # 所以类别名字
    classes = classes[1:]
    print(classes)
    #print(classes)
    all_images, all_iamges_class_labels, selTrain, selTest = get_all_images(classes, conf)
    #print(all_iamges_class_labels) #(3060,1)# 图片标签
    #selTrain, selTest = create_split(all_images, conf)
    print(len(all_iamges_class_labels))
    print(len(all_images))
    # im = imread(all_images[30])
    # plt.title(classes[all_iamges_class_labels[30]])
    # plt.imshow(im)
    # plt.show()
    return classes, all_images, all_iamges_class_labels, selTrain, selTest


#main_fun()
