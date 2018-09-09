from  phow_caltech101 import *
from BuildPyramid import *
from hist_isect import *
from sklearn import svm
import os

def test():

    #data_dir = 'tempresults';
    data_dir = '/mlspace/xyy/dev_kmean_svm/tempresults'
    classes, all_images, all_iamges_class_labels, selTrain, selTest = main_fun()
    print("data_classes:" + str(classes))
    #print("all_images_path:" + str(all_images))
    # print("all_images_class_label:" + str(all_iamges_class_labels))
    # print(len(all_iamges_class_labels))
    # print("train_data:" + str(selTrain))
    # print("test_data:" + str(selTest))

    # path = join(data_dir,'pyramid.mat')
    # if os.path.exists(path):
    #     print("load file data:")
    #     pyramid_all = loadmat(path)['pyramid_all']
    # else:
    pyramid_all = BuildPyramid(all_images, data_dir)

    # K = hist_isect(pyramid_all, pyramid_all)
    # print(np.shape(K))
    print("pyramid_all的维度:%d * %d" % (np.shape(pyramid_all)[0], np.shape(pyramid_all)[1]))
    train_data = np.zeros((len(selTrain), np.shape(pyramid_all)[1]))
    test_data = np.zeros((len(selTest), np.shape(pyramid_all)[1]))
    train_labels = []
    test_labels = []
    for i in range(len(selTrain)):
        train_data[i, :] = pyramid_all[selTrain[i], :]
        train_labels.append(all_iamges_class_labels[selTrain[i]])
    for j in range(len(selTest)):
        test_data[j, :] = pyramid_all[selTest[j], :]
        test_labels.append(all_iamges_class_labels[selTest[j]])

    # hist_path = join(data_dir, 'hists.py.mat')
    # hist = loadmat(hist_path)['hists']
    # print(np.shape(hist))
    # Train SVM
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
    # file = open('/mlspace/xyy/dev_kmean_svm/file.txt', 'w')
    # for i in range(np.shape(cm)[0]):
    #     for j in range(np.shape(cm)[1]):
    #         file.write(str(cm[i,j]))
    #     file.write('\n')
    # file.close()


#test()

