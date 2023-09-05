import cv2
import numpy as np
# coding=UTF8
def show_img(img):
    cv2.imshow(str(img), img)
    cv2.waitKey()
    cv2.destroyAllWindows()
    return 0


def get_grain_length(contours):
    # """
    # 通过谷粒轮廓求取谷粒长度
    # """
    point = contours[0]
    len_contours = len(point)
    arr = np.zeros((len_contours, len_contours))
    for i in range(len_contours):
        for j in range(len_contours):
            x1 = point[i][0][0]
            x2 = point[j][0][0]
            y1 = point[i][0][1]
            y2 = point[j][0][1]
            dist = np.sqrt(abs(x1-x2)**2+abs(y1-y2)**2)
            arr[i][j] = dist
    col_max_idx = np.argmax(arr, axis=0) 
    print(arr.shape)
    col_max = []
    for m in range(len_contours):
        col_max.append(arr[m][col_max_idx[m]])
    n = np.argmax(col_max)
    print(np.max(arr))
    # print(max(max(arr)))
    # print(col_max)
    point1 = contours[0][np.unravel_index(np.argmax(arr), arr.shape)[0]][0]
    point2 = contours[0][np.unravel_index(np.argmax(arr), arr.shape)[1]][0]
    return m, n, point1, point2

def rmse_mape(real, predict):
    rmse = np.sqrt(sum((real[i]-predict[i])**2 for i in range(len(real)))/len(real))
    mape = sum(np.abs((predict[i]-real[i])/real[i]) for i in range(len(real)))/len(real)
    return rmse, mape*100

def thresh_Seg(img, thresh, type = 3):
    """
    对输入图像进行阈值分割
    :param img: 输入图像，三维彩色图像或者灰度图像
    :param thresh: 分割阈值
    :param type: 默认为3代表输入彩色图像， 若为灰度图需指定为2
    :return: 二值图
    """
    if type == 3:
        gray = img[:, :, 2]
        BN = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)[1]
    elif type == 2:
        BN = cv2.threshold(img, thresh, 255, cv2.THRESH_BINARY)[1]
    return BN

def max_cont_idx(contours):
    """
    找到面积最大的轮廓并返回它的索引值和面积
    :param contours: 所有轮廓
    :return: 面积最大的轮廓的索引和面积
    """
    area = []
    for k in range(len(contours)):
        area.append(cv2.contourArea(contours[k]))
    max_idx = np.argmax(np.array(area))
    max_area = area[max_idx]
    return max_idx, max_area

def filter_dst_cont(contours, thresh):
    """
    输入图像所有轮廓，取出轮廓面积大于某一值的所有轮廓
    :param contours: 由二值图得到的所有轮廓
    :param thresh: 面积阈值，轮廓大于这个值则保留，否着丢弃
    :return: 所有面积大于thresh的轮廓
    """
    cont = []
    for k in range(len(contours)):
        if cv2.contourArea(contours[k]) > thresh:
            cont.append(contours[k])
    return cont

def get_max_contour(img, contours, idx):
    """

    :param img: 输入rgb图像
    :return: 返回一张图像，大小和原图保持一致，但只包括原图中最大轮廓的部分，其它地方置零
    """
    mask = np.zeros_like(img)
    cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
    mask[np.where(mask)] = img[np.where(mask)]
    return mask



def To_Tempalate(img, size=256):
    """
    创建一个size大小的画布，将输入图像放在画布中间
    :param img: rgb图像
    :param size: 画布大小
    :return:
    """
    template = np.zeros((size, size, 3), np.uint8)

    x, y, _ = img.shape
    x1 = size//2 - x//2 - 1 if x % 2 != 0 else size//2 - x//2
    y1 = size//2 - y // 2 - 1 if y % 2 != 0 else size//2 - y // 2
    template[x1:size//2 + x//2, y1:size//2 + y//2] = img
    return template

def Get_W_H(image):
    gray = image[:, :, 2]
    bn = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY)[1]
    kernel = np.ones((5, 5), dtype=np.uint8)
    bn = cv2.morphologyEx(bn, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(bn, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_idx, max_area = max_cont_idx(contours)

    rect = cv2.minAreaRect(contours[max_idx])
    perimeter = cv2.arcLength(contours[max_idx], closed=True)
    # print(rect)
    w, h = max(rect[1]), min(rect[1])
    return w, h, max_area, perimeter

def max_dist(arr):
    if len(arr) == 1:
        return 0
    else:
        dist = np.zeros((len(arr), len(arr)))
        for i in range(len(arr)):
            for j in range(len(arr)):
                dist[i][j] = np.sqrt(np.power((arr[j][0] - arr[i][0]), 2) + np.power((arr[j][1] - arr[i][1]), 2))

    return np.max(dist)


def Get_length_width(contours):
    dist = np.zeros((len(contours[0]), len(contours[0])))

    for i in range(len(contours[0])):
        for j in range(len(contours[0])):
            x1 = contours[0][i][0][0]
            y1 = contours[0][i][0][1]
            x2 = contours[0][j][0][0]
            y2 = contours[0][j][0][1]
            dist[i][j] = np.sqrt(np.power((x2 - x1), 2) + np.power((y2 - y1), 2))

    point1 = contours[0][np.unravel_index(np.argmax(dist), dist.shape)[0]][0]
    point2 = contours[0][np.unravel_index(np.argmax(dist), dist.shape)[1]][0]

    intercept = int(256+256*((point2[0]-point1[0])/(point2[1]-point1[1])))
    width_dist = []
    for i in range(intercept):
        poi = []  # point of intersection
        for j in range(len(contours[0])):
            if contours[0][j][0][1] == i - int(((point2[0]-point1[0])/(point2[1]-point1[1]))*contours[0][j][0][0]):
                poi.append((contours[0][j][0][0], contours[0][j][0][1]))
                width_dist.append(max_dist(poi))
    length = np.max(dist)*0.042333
    width = np.max(width_dist)*0.042333
    return length, width