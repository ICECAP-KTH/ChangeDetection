"""
Slow feature analysis
C. Wu, B. Du, and L. Zhang, “Slow feature analysis for change detection in multispectral imagery,” IEEE Trans. Geosci. Remote Sens., vol. 52, no. 5, pp. 2858–2874, 2014.

This implementation is based on the one available here:
https://github.com/I-Hope-Peace/ChangeDetectionRepository/tree/master/Methodology/Traditional/SFA
"""
import numpy as np
from scipy.linalg import eig
from scipy.stats import chi2

import gdal
import time
import imageio


def otsu(data, num=400):
    """
    Generate threshold value based on Otsu.

    :param data: cluster data
    :param num: intensity number
    :return:
        selected threshold
    """
    max_value = np.max(data)
    min_value = np.min(data)

    total_num = data.shape[1]
    best_threshold = min_value
    best_inter_class_var = 0
    step_value = (max_value - min_value) / num

    value = min_value + step_value
    while value <= max_value:
        data_1 = data[data <= value]
        data_2 = data[data > value]

        if data_1.shape[0] == 0 or data_2.shape[0] == 0:
            value += step_value
            continue

        w1 = data_1.shape[0] / total_num
        w2 = data_2.shape[0] / total_num

        mean_1 = data_1.mean()
        mean_2 = data_2.mean()

        inter_class_var = w1 * w2 * np.power((mean_1 - mean_2), 2)

        if best_inter_class_var < inter_class_var:
            best_inter_class_var = inter_class_var
            best_threshold = value

        value += step_value

    return best_threshold


class ISFA(object):
    def __init__(self, img_X, img_Y, data_format='CHW'):
        """
        the init function
        :param img_X: former temporal image, its dim is (band_count, width, height)
        :param img_Y: latter temporal image, its dim is (band_count, width, height)
        """
        if data_format == 'HWC':
            self.img_X = np.transpose(img_X, [2, 0, 1])
            self.img_Y = np.transpose(img_Y, [2, 0, 1])
        else:
            self.img_X = img_X
            self.img_Y = img_Y

        channel, height, width = self.img_X.shape
        self.L = np.zeros((channel - 2, channel))  # (C-2, C)
        for i in range(channel - 2):
            self.L[i, i] = 1
            self.L[i, i + 1] = -2
            self.L[i, i + 2] = 1
        self.Omega = np.dot(self.L.T, self.L)  # (C, C)
        self.norm_method = ['LSR', 'NR', 'OR']

    def isfa(self, max_iter=30, epsilon=1e-6, norm_trans=False, regular=True):
        """
         extract change and unchange info of temporal images based on USFA
         if max_iter == 1, ISFA is equal to SFA
        :param max_iter: the maximum count of iteration
        :param epsilon: convergence threshold
        :param norm_trans: whether normalize the transformation matrix
        :return:
            ISFA_variable: ISFA variable, its dim is (band_count, width * height)
            lamb: last lambda
            all_lambda: all lambda in convergence process
            trans_mat: transformation matrix
            T: last IWD, if max_iter == 1, T is chi-square distance
            weight: the unchanged probability of each pixel
        """

        bands_count, img_height, img_width = self.img_X.shape
        P = img_height * img_width
        # row-major order after reshape
        img_X = np.reshape(self.img_X, (-1, img_height * img_width))  # (band, width * height)
        img_Y = np.reshape(self.img_Y, (-1, img_height * img_width))  # (band, width * height)
        lamb = 100 * np.ones((bands_count, 1))
        all_lambda = []
        weight = np.ones((img_width, img_height))  # (1, width * height)
        # weight[302:343, 471] = 1  # init seed
        # weight[209, 231:250] = 1
        # weight[335:362, 570] = 1
        # weight[779, 332:387] = 1

        weight = np.reshape(weight, (-1, img_width * img_height))
        for _iter in range(max_iter):
            sum_w = np.sum(weight)

            # Centralize the input signal to obtain zero mean
            mean_X = np.sum(weight * img_X, axis=1, keepdims=True) / sum_w  # (band, 1)
            mean_Y = np.sum(weight * img_Y, axis=1, keepdims=True) / sum_w  # (band, 1)
            center_X = (img_X - mean_X)
            center_Y = (img_Y - mean_Y)

            # Normalize image
            var_X = np.sum(weight * np.power(center_X, 2), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
            var_Y = np.sum(weight * np.power(center_Y, 2), axis=1, keepdims=True) / ((P - 1) * sum_w / P)
            std_X = np.reshape(np.sqrt(var_X), (bands_count, 1))
            std_Y = np.reshape(np.sqrt(var_Y), (bands_count, 1))
            norm_X = center_X / std_X
            norm_Y = center_Y / std_Y

            diff_img = (norm_X - norm_Y)

            mat_A = np.dot(weight * diff_img, diff_img.T) / ((P - 1) * sum_w / P)
            mat_B = (np.dot(weight * norm_X, norm_X.T) +
                     np.dot(weight * norm_Y, norm_Y.T)) / (2 * (P - 1) * sum_w / P)

            if regular:
                penalty = np.trace(mat_B) / np.trace(self.Omega)
                mat_B += penalty * self.Omega

            # solve generalized eigenvalue problem and get eigenvalues and eigenvector
            eigenvalue, eigenvector = eig(mat_A, mat_B)
            eigenvalue = eigenvalue.real  # discard imaginary part
            idx = eigenvalue.argsort()
            eigenvalue = eigenvalue[idx]

            # make sure the max absolute value of vector is 1,
            # and the final result will be more closer to the matlab result
            aux = np.reshape(np.abs(eigenvector).max(axis=0), (1, bands_count))
            eigenvector = eigenvector / aux

            # print sqrt(lambda)
            if (_iter + 1) == 1:
                print('sqrt lambda:')
            print(np.sqrt(eigenvalue))

            eigenvalue = np.reshape(eigenvalue, (bands_count, 1))  # (band, 1)
            threshold = np.max(np.abs(np.sqrt(lamb) - np.sqrt(eigenvalue)))
            # if sqrt(lambda) converge
            if threshold < epsilon:
                break

            lamb = eigenvalue

            all_lambda = lamb if (_iter + 1) == 1 else np.concatenate((all_lambda, lamb), axis=1)

            # the order of the slowest features is determined by the order of the eigenvalues
            trans_mat = eigenvector[:, idx]

            # satisfy the constraints(3)
            if norm_trans:
                output_signal_std = 1 / np.sqrt(np.diag(np.dot(trans_mat.T, np.dot(mat_B, trans_mat))))
                trans_mat = output_signal_std * trans_mat
            ISFA_variable = np.dot(trans_mat.T, norm_X) - np.dot(trans_mat.T, norm_Y)

            T = np.sum(np.square(ISFA_variable) / np.sqrt(lamb), axis=0, keepdims=True)

            weight = 1 - chi2.cdf(T, bands_count)

        if (_iter + 1) == max_iter:
            print('Lambda may not have converged')
        else:
            print('Lambda has converged, the iteration is %d' % (_iter + 1))

        return ISFA_variable, lamb, all_lambda, trans_mat, T, weight


def read_isfa_images(image1, image2):
    data_set_X = gdal.Open(image1)
    data_set_Y = gdal.Open(image2)

    img_width = data_set_X.RasterXSize
    img_height = data_set_X.RasterYSize
    # num_channels = data_set_X.RasterCount

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    return img_X, img_Y


def run_isfa(img_X, img_Y):
    """
    Wrapper of the Slow Feature Analysis algorithm.

    :param img_X: First image.
    :param img_Y: Second image.
    :return:
        bcm: Binary change matrix between both images
    """
    channel, img_height, img_width = img_X.shape

    sfa = ISFA(img_X, img_Y)
    # when max_iter is set to 1, ISFA becomes SFA
    _, _, _, _, bn_iwd, _ = sfa.isfa(max_iter=50, epsilon=1e-3, norm_trans=True)

    sqrt_chi2 = np.sqrt(bn_iwd)

    bcm = np.zeros((1, img_height * img_width), dtype=np.uint8)
    thre = otsu(sqrt_chi2)

    bcm[sqrt_chi2 > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))

    return bcm


def main():
    image1 = "../test_images/area2_20200610_100759_89_1061_3B_AnalyticMS_SR.tif"
    image2 = "../test_images/area2_20200620_100924_87_1066_3B_AnalyticMS_SR.tif"

    img_X, img_Y = read_isfa_images(image1, image2)

    tic = time.time()

    bcm = run_isfa(img_X, img_Y)

    toc = time.time()
    print(toc - tic)

    imageio.imwrite('ISFA_out.png', bcm)


if __name__ == '__main__':
    main()
