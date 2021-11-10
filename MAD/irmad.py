"""
Python implementation of IRMAD
A. A. Nielsen, “The regularized iteratively reweighted MAD method for change detection in multi- and hyperspectral data,” IEEE Trans. Image Process., vol. 16, no. 2, pp. 463–478, 2007.

This implementation is based on the one available here:
https://github.com/I-Hope-Peace/ChangeDetectionRepository/tree/master/Methodology/Traditional/MAD
"""
import numpy as np
from numpy.linalg import inv, eig
from scipy.stats import chi2
from sklearn.cluster import KMeans

import gdal
import time
import imageio


def covw(center_X, center_Y, weight):
    n = weight.shape[1]
    V = np.sqrt(weight) * np.concatenate((center_X, center_Y), axis=0)
    dis = np.dot(V, V.T) / weight.sum() * (n / (n - 1))
    return dis


def IRMAD(img_X, img_Y, max_iter=50, epsilon=1e-3):
    bands_count_X, num = img_X.shape

    weight = np.ones((1, num))  # (1, height * width)
    can_corr = 100 * np.ones((bands_count_X, 1))

    for _iter in range(max_iter):
        # Centralize the input signal to obtain zero mean
        sum_w = np.sum(weight)
        mean_X = np.sum(weight * img_X, axis=1, keepdims=True) / sum_w
        mean_Y = np.sum(weight * img_Y, axis=1, keepdims=True) / sum_w
        center_X = img_X - mean_X
        center_Y = img_Y - mean_Y

        # also can use np.cov, but the result would be sightly different with author' result acquired by MATLAB code
        cov_XY = covw(center_X, center_Y, weight)
        size = cov_XY.shape[0]
        sigma_11 = cov_XY[0:bands_count_X, 0:bands_count_X]  # + 1e-4 * np.identity(3)
        sigma_22 = cov_XY[bands_count_X:size, bands_count_X:size]  # + 1e-4 * np.identity(3)
        sigma_12 = cov_XY[0:bands_count_X, bands_count_X:size]  # + 1e-4 * np.identity(3)
        sigma_21 = sigma_12.T

        target_mat = np.dot(np.dot(np.dot(inv(sigma_11), sigma_12), inv(sigma_22)), sigma_21)

        eigenvalue, eigenvector_X = eig(target_mat)  # the eigenvalue and eigenvector of image X
        eigenvalue = np.sqrt(eigenvalue)
        idx = eigenvalue.argsort()
        eigenvalue = eigenvalue[idx]

        if (_iter + 1) == 1:
            print('Canonical correlations')
        print(eigenvalue)

        # sort eigenvector based on the size of eigenvalue
        eigenvector_X = eigenvector_X[:, idx]

        eigenvector_Y = np.dot(np.dot(inv(sigma_22), sigma_21), eigenvector_X)  # the eigenvector of image Y

        # tune the size of X and Y, so the constraint condition can be satisfied
        norm_X = np.sqrt(1 / np.diag(np.dot(eigenvector_X.T, np.dot(sigma_11, eigenvector_X))))
        norm_Y = np.sqrt(1 / np.diag(np.dot(eigenvector_Y.T, np.dot(sigma_22, eigenvector_Y))))
        eigenvector_X = norm_X * eigenvector_X
        eigenvector_Y = norm_Y * eigenvector_Y

        mad_variates = np.dot(eigenvector_X.T, center_X) - np.dot(eigenvector_Y.T, center_Y)  # (6, width * height)

        if np.max(np.abs(can_corr - eigenvalue)) < epsilon:
            break

        can_corr = eigenvalue

        # calculate chi-square distance and probility of unchanged
        mad_var = np.reshape(2 * (1 - can_corr), (bands_count_X, 1))
        chi_square_dis = np.sum(mad_variates * mad_variates / mad_var, axis=0, keepdims=True)
        weight = 1 - chi2.cdf(chi_square_dis, bands_count_X)

    if (_iter + 1) == max_iter:
        print('the canonical correlation may not have converged')
    else:
        print('the canonical correlation has converged, the iteration is %d' % (_iter + 1))

    return mad_variates, can_corr, mad_var, eigenvector_X, eigenvector_Y, \
           sigma_11, sigma_22, sigma_12, chi_square_dis, weight


def get_binary_change_map(data):
    cluster_center = KMeans(n_clusters=2, max_iter=1500).fit(data.T).cluster_centers_.T  # shape: (1, 2)

    print('k-means cluster is done, the cluster center is ', cluster_center)
    dis_1 = np.linalg.norm(data - cluster_center[0, 0], axis=0, keepdims=True)
    dis_2 = np.linalg.norm(data - cluster_center[0, 1], axis=0, keepdims=True)

    bcm = np.zeros((1, data.shape[1]), dtype=np.uint8)
    if cluster_center[0, 0] > cluster_center[0, 1]:
        bcm[dis_1 <= dis_2] = 255
    else:
        bcm[dis_1 > dis_2] = 255

    return bcm


def read_mad_images(image1, image2):
    data_set_X = gdal.Open(image1)
    data_set_Y = gdal.Open(image2)

    img_width = data_set_X.RasterXSize
    img_height = data_set_X.RasterYSize
    # num_channels = data_set_X.RasterCount

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    return img_X, img_Y


def run_irmad(img_X, img_Y):
    """
    Wrapper of the Multivariate Alteration Detection algorithm.

    :param img_X: First image.
    :param img_Y: Second image.
    :return:
        k_means_bcm: Binary change matrix between both images
    """
    channel, img_height, img_width = img_X.shape

    img_X = np.reshape(img_X, (channel, -1))
    img_Y = np.reshape(img_Y, (channel, -1))

    # when max_iter is set to 1, IRMAD becomes MAD
    _, _, _, _, _, _, _, _, chi2, _ = IRMAD(img_X, img_Y, max_iter=1, epsilon=1e-3)
    sqrt_chi2 = np.sqrt(chi2)

    k_means_bcm = get_binary_change_map(sqrt_chi2)
    k_means_bcm = np.reshape(k_means_bcm, (img_height, img_width))
    return k_means_bcm


def main():
    image1 = "../test_images/area2_20200610_100759_89_1061_3B_AnalyticMS_SR.tif"
    image2 = "../test_images/area2_20200620_100924_87_1066_3B_AnalyticMS_SR.tif"

    img_X, img_Y = read_mad_images(image1, image2)

    tic = time.time()

    bcm = run_irmad(img_X, img_Y)

    toc = time.time()
    print(toc - tic)

    imageio.imwrite('IRMAD_out.png', bcm)


if __name__ == '__main__':
    main()
