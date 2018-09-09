import numpy as np
from os.path import join
from glob import glob
import os
from phow_caltech101 import get_classes
from scipy.io import loadmat, savemat
from sklearn import svm
from sklearn.metrics import confusion_matrix, accuracy_score
import operator
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.cross_validation import cross_val_score

def main_test():
    source_data = '/mlspace/xyy/dev_kmean_svm/dataset/101_ObjectCategories'
    data_dir = '/mlspace/xyy/dev_kmean_svm/new_image'
    save_data_dir = '/mlspace/xyy/dev_kmean_svm/data'
    # source_data = 'D:\\python code\\dev_kmean_svm\dataset\\101_ObjectCategories'
    # data_dir = 'D:\\python code\\dev_kmean_svm\\new_image'
    # save_data_dir = 'D:\\python code\\dev_kmean_svm\\data'
    extensions = [".jpg", ".bmp", ".png", ".pgm", ".tif", ".tiff"]
    test = 30
    train = 30
    total_use = train + test
    a = 1
    all_images = []
    all_images_class_labels = []
    sel_train = []
    sel_test = []
    mat_dir = []
    total_class = get_classes(source_data, 102)
    print(total_class)
    k=0
    p=0
    imgs = os.listdir(data_dir)
    for i in range(len(total_class)):
        # for fname in glob(data_dir+'/*'):
        #     if os.path.splitext(fname)[-1].lower() in extensions:
        #         #print(os.path.basename(fname))
        #         name = os.path.basename(fname)
        #
        #         #print(os.path.basename(fname)[-14:])
        #     if total_class[i] in name:
        #         mat_name = name[0:-4] + '_pyramid_200_3.mat'
        #         mat = join(save_data_dir, mat_name)
        #         mat_dir.append(mat)
        #         all_images.append(join(data_dir,name))
        #         all_images_class_labels.append(i)
        for num in range(len(imgs)):
            name = imgs[num]
            class_len = len(total_class[i])
            if total_class[i] == name[0:-15] or total_class[i] == name[0:-16]:
                a=a+1
                mat_name = name[0:-4] + '_pyramid_200_3.mat'
                mat = join(save_data_dir, mat_name)
                mat_dir.append(mat)
                all_images.append(join(data_dir,name))
                all_images_class_labels.append(i)
        for j in range(k,len(all_images)):
            if (j-p) < train:
                sel_train.append(j)
            else:
                sel_test.append(j)
        k = len(all_images)
        p = k

        a=0
    # train_path = join('D:\\python code\\dev_kmean_svm\\tempresults','train.mat')
    # test_path = join('D:\\python code\\dev_kmean_svm\\tempresults','test.mat')
    # savemat(train_path,{'sel_train':sel_train})
    # savemat(test_path,{'sel_test':sel_test})

    train_data = np.zeros((len(sel_train), 4200))
    test_data = np.zeros((len(sel_test), 4200))
    train_labels = []
    test_labels = []

    print("load mat data")
    for i in range(len(sel_train)):
        pyramid_dir = mat_dir[sel_train[i]]
        pyramid = loadmat(pyramid_dir)['pyramid']
        train_data[i,:] = pyramid
        train_labels.append(all_images_class_labels[sel_train[i]])

    for j in range(len(sel_test)):
        pyramid_dir = mat_dir[sel_test[j]]
        pyramid = loadmat(pyramid_dir)['pyramid']
        test_data[j, :] = pyramid
        test_labels.append(all_images_class_labels[sel_test[j]])
    #
    #print(train_labels)
    #
    c_type = 200
    for i in range(5):
        print("epoch:%d" % i)

        clf = svm.LinearSVC(C=c_type)
        clf.fit(train_data, train_labels)

        # Test SVM
        predicted_class = clf.predict(test_data)
        accuracy = accuracy_score(test_labels, predicted_class)
        cm = confusion_matrix(test_labels, predicted_class)
        c_type = c_type + 30
        print("accuracy %f:" % accuracy)
        print("混淆矩阵为:" )
        print(cm)
        print(np.shape(cm))
        # classfiler = LogisticRegression()
        # classfiler.fit(train_data, train_labels)
        # prediction = classfiler.predict(test_data)
        # scores = accuracy_score(test_labels, prediction)
        # print("准确率为:%s" % scores)

main_test()