import numpy as np
from scipy import signal
from sp_normalize_sift import *


def gaussian_2d_kernal(kersize, sigma):
    kernal = np.zeros([kersize, kersize])
    center = kersize//2
    if sigma == 0:
       sigma = ((kersize-1)*0.5-1)*0.3+0.8
    s = (sigma**2)*2
    sum_val = 0
    for i in range(kersize):
        for j in range(kersize):
            x = i - center
            y = j - center
            kernal[i, j] = np.exp(-(x**2+y**2) / s)
            sum_val += kernal[i, j]
    sum_val = 1/sum_val
    aa = kernal * sum_val
    return aa

def sp_dense_sift(I, grid_spacing, patch_size):

    I = np.double(I)
    I = I / np.max(I)
    num_angles = 8
    num_bins = 4
    num_samples = num_bins * num_bins
    alpha = 9  # parameter for attenuation of angles (must be odd)
    sidgma = 1

    angle_step = 2 * np.pi / num_angles
    angles = np.arange(0, 2*np.pi, angle_step)
    #print(angles)
    hgt, wid = np.shape(I)
    GX, GY = gen_dgauss(sidgma)

    # add boundary
    I = np.vstack((I[0:2,:], I))
    I = np.vstack((I, I[-2:,:]))
    I = np.hstack((I[:,0:2], I))
    I = np.hstack((I, I[:,-2:]))

    I = I - np.mean(I)
    I_X = signal.convolve2d(I, np.fliplr(np.flipud(GX)), mode='same')
    I_Y = signal.convolve2d(I, np.fliplr(np.flipud(GY)), mode='same')

    I_X = I_X[2:-2, 2:-2]
    I_Y = I_Y[2:-2, 2:-2]

    I_mag = np.sqrt((I_X ** 2 + I_Y ** 2))
    I_theta = np.arctan2(I_Y, I_X)
    for i in range(np.shape(I_theta)[0]):
        for j in range(np.shape(I_theta)[1]):
            if np.isnan(I_theta[i][j]):
                I_theta[i][j] = 0

    # grid
    grid_x = np.arange(patch_size/2, wid-patch_size/2+1, grid_spacing)
    grid_y = np.arange(patch_size/2, hgt-patch_size/2+1, grid_spacing)

    # make orientation images
    I_orientation = np.zeros((hgt, wid, num_angles))

    # for each histogram angle
    cosI = np.cos(I_theta)
    sinI = np.sin(I_theta)

    for a in range(num_angles):
        # compute each orientation channel
        tmp = (cosI * np.cos(angles[a]) + sinI * np.sin(angles[a])) ** alpha
        tmp = tmp * np.double(tmp > 0)

        I_orientation[:, :, a] = tmp * I_mag

    # Convolution formulation:
    weight_kernel = np.zeros((patch_size, patch_size)) # [16, 16]
    r = patch_size / 2  # 8
    cx = r - 0.5   # 7.5
    sample_res = patch_size / num_bins  # 4
    weight_x = np.arange(1,patch_size+1)
    weight_x = np.abs(weight_x - cx) / sample_res
    weight_x = np.reshape(weight_x, (np.shape(weight_x)[0], 1))
    new_weight = 1 - weight_x
    for k in range(np.shape(weight_x)[0]):
        if weight_x[k, 0] <= 1:
            weight_x[k, 0] = 1
        else:
            weight_x[k, 0] = 0
    weight_x = new_weight * weight_x
    for a in range(num_angles):
        I_orientation[:, :, a] = signal.convolve2d(I_orientation[:, :, a], weight_x, 'same')
        I_orientation[:, :, a] = signal.convolve2d(I_orientation[:, :, a], weight_x.T, 'same')

    # Sample SIFT bins at valid locations (without boundary artifacts)  find coordinates of sample points (bin centers)
    x = np.linspace(1, patch_size+1, num_bins+1)
    sample_x, sample_y = np.meshgrid(x, x)
    sample_x  =sample_x[0:num_bins, 0:num_bins]
    sample_x = np.reshape(sample_x, (np.shape(sample_x)[0]*np.shape(sample_x)[1], 1))
    sample_x = sample_x - patch_size/2
    sample_y = sample_y[0:num_bins, 0:num_bins]
    sample_y = np.reshape(sample_y, (np.shape(sample_y)[0]*np.shape(sample_y)[1], 1))
    sample_y = sample_y - patch_size/2

    sift_arr = np.zeros((len(grid_y), len(grid_x), num_angles*num_bins*num_bins)) #[,,8*4*4]
    b = 0
    x_indices = sample_x[0]+grid_x
    y_indices = sample_y[0]+grid_y
    for n in range(num_bins*num_bins):
        # for i in range(len(grid_x)):
        #     for j in range(len(grid_y)):
        #         sift_arr[i, j, b:b+num_angles] = I_orientation[x_indices[i], y_indices[j], :]
        ss = I_orientation[patch_size//2+int(sample_y[n]):hgt-patch_size//2+1+int(sample_y[n]):8,
                                         patch_size//2+int(sample_x[n]):wid-patch_size//2+1+int(sample_x[n]):8, :]
        sift_arr[:, :, b:b+num_angles] = ss
        b = b + num_angles

    # outputs
    grid_x, grid_y = np.meshgrid(grid_x, grid_y)
    nrows, ncols, cols = np.shape(sift_arr)

    # normalize SIFT descriptors  slow good normailzation that respects the flat areas
    sift_arr = np.reshape(sift_arr, (nrows*ncols, num_angles*num_bins*num_bins))
    sift_arr = sp_normalize_sift(sift_arr)
    sift_arr = np.reshape(sift_arr, (nrows, ncols, num_angles*num_bins*num_bins))

    return sift_arr, grid_x, grid_y



def gen_dgauss(sigma):

    G = gen_one_dgauss(sigma)
    GY, GX = np.gradient(G)
    #print("gussian: %f" % np.sum(np.sum(np.abs(GX), 0)))
    GX = GX * 2 / np.sum(np.sum(np.abs(GX),0), 0)
    GY = GY * 2 / np.sum(np.sum(np.abs(GY), 0), 0)

    return GX, GY


def gen_one_dgauss(sigam):

    f_wid = np.int(4 * np.ceil(sigam) + 1) # 5
    G =gaussian_2d_kernal(f_wid, sigam)
    return G

# a = np.array([[6,9,3,4,0],
#              [5,4,1,2,5],
#              [6,7,7,8,0],
#              [7,8,9,10,0]])
# sp_dense_sift(a,1,1)
# a = a[1:-2,:]
# grid_x = np.arange(16/2, 200-16/2+1, 8)
# print(grid_x)
#print(a)
# print(a)
# b = np.vstack((a[0:2,:], a))
# a = np.vstack((a,a[-2:,:]))
# print(a)
# # G =gaussian_2d_kernal(5, 1)
# b, c= np.gradient(G)
# print(G)
# print(b)
# print(c)