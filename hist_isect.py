import numpy as np

"""
Evaluate a histogram intersection kernel
"""
def hist_isect(x1, x2):

    n = np.shape(x1)[0]
    m = np.shape(x2)[0]
    K = np.zeros((n, m))

    if m <= n:
        for p in range(m):
            indices = []
            for j in range(np.shape(x1)[1]):
                if x1[p, j] > 0:
                    indices.append(j)
            t_x1 = np.zeros((1, len(indices)))
            for i in range(len(indices)):
                t_x1[0, i] = x1[p, indices[i]]
            tmp_x1 = np.tile(t_x1, (n, 1))
            t_x2 = np.zeros((n, len(indices)))
            for i in range(len(indices)):
                t_x2[:, i] = x2[:, indices[2]]

            min_value = np.zeros((n, len(indices)))
            for i in range(n):
                for j in range(len(indices)):
                    if tmp_x1[i, j] <= t_x2[i, j]:
                        min_value[i, j] = t_x2[i, j]
                    else:
                        min_value[i, j] = tmp_x1[i, j]
            K[p, :] = np.sum(min_value, 1).T
    else:
        for p in range(n):
            indices = []
            for j in range(np.shape(x2)[1]):
                if x2[p, j] > 0:
                    indices.append(j)
            t_x2 = np.zeros((1, len(indices)))
            for i in range(len(indices)):
                t_x2[0, i] = x2[p, indices[i]]
            tmp_x2 = np.tile(t_x2, (n, 1))
            t_x1 = np.zeros((n, len(indices)))
            for i in range(len(indices)):
                t_x1[:, i] = x1[:, indices[2]]

            min_value = np.zeros((n, len(indices)))
            for i in range(n):
                for j in range(len(indices)):
                    if tmp_x2[i, j] <= t_x2[i, j]:
                        min_value[i, j] = t_x2[i, j]
                    else:
                        min_value[i, j] = tmp_x2[i, j]
            K[p, :] = np.sum(min_value, 1).T

    return K
