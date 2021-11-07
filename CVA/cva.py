"""
Change Vector Analysis algorithm

This implementation is based on the one available here:
https://github.com/I-Hope-Peace/ChangeDetectionRepository/tree/master/Methodology/Traditional/CVA
"""
import numpy as np
import imageio
import gdal
import time


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


def normalize_img(img, channel_first=True):
    """
    Normalize image.

    :param img: (C, H, W) or (H, W, C)
    :param channel_first:
    :return:
        norm_img: (C, H, W)
    """
    if channel_first:
        channel, img_height, img_width = img.shape
        img = np.reshape(img, (channel, img_height * img_width))  # (channel, height * width)
        mean = np.mean(img, axis=1, keepdims=True)  # (channel, 1)
        center = img - mean  # (channel, height * width)
        var = np.sum(np.power(center, 2), axis=1, keepdims=True) / (img_height * img_width)  # (channel, 1)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (channel, img_height, img_width))
    else:
        img_height, img_width, channel = img.shape
        img = np.reshape(img, (img_height * img_width, channel))  # (height * width, channel)
        mean = np.mean(img, axis=0, keepdims=True)  # (1, channel)
        center = img - mean  # (height * width, channel)
        var = np.sum(np.power(center, 2), axis=0, keepdims=True) / (img_height * img_width)  # (1, channel)
        std = np.sqrt(var)  # (channel, 1)
        nm_img = center / std  # (channel, height * width)
        nm_img = np.reshape(nm_img, (img_height, img_width, channel))
    return nm_img


def CVA(img_X, img_Y):
    """
    Change Vector Analysis algorithm.

    :param img_X: First image.
    :param img_Y: Second image.
    :return:
        bcm: Binary change matrix between both images
    """
    channel, img_height, img_width = img_X.shape

    # CVA has not affinity transformation consistency, so it is necessary to 
    # normalize multi-temporal images to eliminate the radiometric 
    # inconsistency between them
    img_X = normalize_img(img_X)
    img_Y = normalize_img(img_Y)

    img_diff = img_X - img_Y
    L2_norm = np.sqrt(np.sum(np.square(img_diff), axis=0))

    bcm = np.zeros((img_height, img_width), dtype=np.uint8)
    thre = otsu(L2_norm.reshape(1, -1))

    bcm[L2_norm > thre] = 255
    bcm = np.reshape(bcm, (img_height, img_width))

    return bcm

def read_cva_images(image1, image2):
    data_set_X = gdal.Open(image1)
    data_set_Y = gdal.Open(image2)

    img_width = data_set_X.RasterXSize
    img_height = data_set_X.RasterYSize
    # num_channels = data_set_X.RasterCount

    img_X = data_set_X.ReadAsArray(0, 0, img_width, img_height)
    img_Y = data_set_Y.ReadAsArray(0, 0, img_width, img_height)

    return img_X, img_Y


def main():
    image1 = "../test_images/area2_20200610_100759_89_1061_3B_AnalyticMS_SR.tif"
    image2 = "../test_images/area2_20200620_100924_87_1066_3B_AnalyticMS_SR.tif"

    img_X, img_Y = read_cva_images(image1, image2)

    tic = time.time()

    bcm = CVA(img_X, img_Y)
    imageio.imwrite('CVA_out.png', bcm)

    toc = time.time()
    print(toc - tic)


if __name__ == '__main__':
    main()
