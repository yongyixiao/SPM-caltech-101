import numpy as np
from sp_dist2 import *

"""
Trains a k means cluster model.
"""

def sp_kmeans(centres, data, options):

    ndata, data_dim = np.shape(data)
    ncentres, dim = np.shape(centres) # 200, 128

    if data_dim != dim:
        raise ValueError("Data dimension does not match dimension of centres")

    if ncentres > ndata:
        raise ValueError("More centres than data")

    if options[0,13]:
        niters = options[0, 13]
    else:
        niters = 100

    store = 0

    #  Check if centres and posteriors need to be initialised from data
    if options[0, 4] == 1:
        # Do the initialisation
        perm = np.random.permutation(ndata)
        perm = perm[0:ncentres]

        # Assign first ncentres (permuted) data points as centres
        for j in range(np.shape(perm)[0]):
            centres[j, :] = data[j, :]

    # Matrix to make unit vectors easy to construct
    id = np.eye(ncentres)

    # Main loop of algorithm
    for n in range(int(niters)):
        # Save old centres to check for termination, [200,128]
        old_centres = centres

        # Calculate posteriors based on existing centres
        d2 = sp_dist2(data, centres)

        # Assign each point to nearest centre
        minvals = np.min(d2.T, 0)
        index = np.argmin(d2.T, 0) # 获得最小值的索引
        post = np.zeros((len(index), ncentres))
        for i in range(len(index)):
            post[i, :] = id[index[i], :]

        num_points = np.sum(post, 0)
        # Adjust the centres based on new posteriors
        for j in range(ncentres):
            if num_points[j] > 0:
                non_zero = np.nonzero(post[:, j] != 0)[0]
                a = np.zeros((len(non_zero), np.shape(data)[1]))
                for k in range(len(non_zero)):
                    a[k, :] = data[non_zero[k], :]
                centres[j, :] = np.sum(a, 0) / num_points[j]

        # Error value is total squared distance from cluster centres
        e = np.sum(minvals)
        if options[0, 0] > 0:
            print('Cycle %d  Error %f' % (n, e))

        if n > 1:
            # Test for
            if np.max(np.max(np.abs(centres - old_centres))) < options[0, 1] and np.abs(old_e - e) < options[0, 2]:
                options[0, 7] = e
        old_e = e


    return centres

