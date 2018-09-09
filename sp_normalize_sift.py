import numpy as np

def sp_normalize_sift(sift_arr):

    # find indices of descriptors to be normalized (those whose norm is larger than 1)
    tmp = np.sqrt(np.sum((sift_arr ** 2), 1))
    tmp = np.reshape(tmp, (np.shape(tmp)[0], 1))
    indices = []
    for j in range(np.shape(tmp)[0]):
        if tmp[j] > 1:
            indices.append(j)
    sift_arr_norm = np.zeros((len(indices), np.shape(sift_arr)[1]))
    tmp_indices = np.zeros((len(indices), 1))
    for i in range(len(indices)):
        sift_arr_norm[i, :] = sift_arr[indices[i], :]
        tmp_indices[i, 0] = tmp[indices[i], 0]

    sift_arr_norm = sift_arr_norm / tmp_indices
    # suppress large gradients
    sift_indices = []
    sift_arr_norm = np.reshape(sift_arr_norm, (np.shape(sift_arr_norm)[0]*np.shape(sift_arr_norm)[1], 1))
    for j in range(np.shape(sift_arr_norm)[0]):
        if sift_arr_norm[j, 0] > 0.2:
            sift_arr_norm[j, 0] = 0.2
    sift_arr_norm = np.reshape(sift_arr_norm, (len(indices), np.shape(sift_arr)[1]))
    # finally, renormalize to unit length
    tmp = np.sqrt(np.sum(sift_arr_norm ** 2, 1))
    tmp = np.reshape(tmp, (np.shape(tmp)[0], 1))
    sift_arr_norm = sift_arr_norm / tmp
    for j  in range(len(indices)):
        sift_arr[indices[j], :] = sift_arr_norm[j, :]

    return sift_arr
