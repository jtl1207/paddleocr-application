
import numpy as np
import cv2

def cmpHash(hash1, hash2):
    '''
    Hash值对比
    两个640位的hash值有多少是不一样的，不同的位数越小，图片越相似
    :param hash1:
    :param hash2:
    :return: 相似度
    '''
    n = 0
    # hash长度不同则返回-1代表传参出错
    if len(hash1) != len(hash2):
        return -1
    # 遍历判断
    for i in range(len(hash1)):
        # 不相等则n计数+1，n最终为相似度
        if hash1[i] != hash2[i]:
            n = n + 1
    return (len(hash1) - n) / len(hash1)

def aHash(img):
    '''
    均值哈希算法
    缩放为80*8
    :param img:
    :return: hash
    '''
    img = cv2.resize(img, (80, 8))
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # s为像素和初值为0，hash_str为hash值初值为''
    s = 0
    hash_str = ''
    # 遍历累加求像素和
    for i in range(8):
        for j in range(80):
            s = s + gray[i, j]
    # 求平均灰度
    avg = s / 640
    # 灰度大于平均值为1相反为0生成图片的hash值
    for i in range(8):
        for j in range(80):
            if gray[i, j] > avg:
                hash_str = hash_str + '1'
            else:
                hash_str = hash_str + '0'
    return hash_str

def order_points(box):
    '''矩形框顺序排列
    :param box: numpy.array, shape=(4, 2)
    :return:
    '''
    center_x, center_y = np.mean(box[:, 0]), np.mean(box[:, 1])
    if np.any(box[:, 0] == center_x) and np.any(box[:, 1] == center_y):  # 有两点横坐标相等，有两点纵坐标相等，菱形
        p1 = box[np.where(box[:, 0] == np.min(box[:, 0]))]
        p2 = box[np.where(box[:, 1] == np.min(box[:, 1]))]
        p3 = box[np.where(box[:, 0] == np.max(box[:, 0]))]
        p4 = box[np.where(box[:, 1] == np.max(box[:, 1]))]
    elif np.any(box[:, 0] == center_x) and np.all(box[:, 1] != center_y):  # 只有两点横坐标相等，先上下再左右
        p12, p34 = box[np.where(box[:, 1] < center_y)], box[np.where(box[:, 1] > center_y)]
        p1, p2 = p12[np.where(p12[:, 0] == np.min(p12[:, 0]))], p12[np.where(p12[:, 0] == np.max(p12[:, 0]))]
        p3, p4 = p34[np.where(p34[:, 0] == np.max(p34[:, 0]))], p34[np.where(p34[:, 0] == np.min(p34[:, 0]))]
    else:  # 只有两点纵坐标相等，或者是没有相等的，先左右再上下
        p14, p23 = box[np.where(box[:, 0] < center_x)], box[np.where(box[:, 0] > center_x)]
        p1, p4 = p14[np.where(p14[:, 1] == np.min(p14[:, 1]))], p14[np.where(p14[:, 1] == np.max(p14[:, 1]))]
        p2, p3 = p23[np.where(p23[:, 1] == np.min(p23[:, 1]))], p23[np.where(p23[:, 1] == np.max(p23[:, 1]))]
    return np.array([p1, p2, p3, p4]).reshape(-1, 2)


def Intelligent_cut(img):
    '''
    寻找最小box旋转并裁剪
    :param img: 输入黑白图
    :return: 返回裁剪完成的图片
    '''
    img2 = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, img2 = cv2.threshold(img2, 127, 255, cv2.THRESH_BINARY_INV)
    contours1, heriachy = cv2.findContours(img2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    listnp1 = []
    for i in contours1:
        if cv2.contourArea(np.array(i)) > 3:
            listnp1 += list(i)
    cnt1 = np.array(listnp1)
    rec1 = cv2.minAreaRect(cnt1)  # 最小外接矩形
    box1 = np.int0(cv2.boxPoints(rec1))  # 矩形的四个角点并取整
    box1=order_points(box1)
    width, height = cv2.minAreaRect(box1)[1]  # 输出的图片大小
    if width < height:
        width, height = height,width
    theta = cv2.minAreaRect(box1)[2]
    angle = 0
    if abs(theta) <= 45:
        angle = theta
    else:
        if theta>0:
            angle = 90 - theta
        else:
            angle = 90 + theta
    if angle != 0:
        h, w = img2.shape
        M = cv2.getRotationMatrix2D(tuple(box1[0]), angle, 1.0)
        try:
            outimg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
        except:
            outimg = img
    else:
        outimg =img
    x = round(box1[0][1])
    y = round(box1[0][0])
    outimg = outimg[x:x + round(height), y:y + round(width)]
    return outimg


def cut(img,box):
    '''
    裁剪图片
    :param img: 输入图片
    :param box: 输入box , shape=(4, 2)
    :return: 输出裁剪完成的图片
    '''
    box = np.int0(box)  # 矩形的四个角点并取整
    box = order_points(box)
    width, height = cv2.minAreaRect(box)[1]# 输出的图片大小
    if width < height:
        width, height = height,width
    theta = cv2.minAreaRect(box)[2]
    angle = 0
    if abs(theta) <= 45:
        angle = theta
    else:
        if theta > 0:
            angle = 90 - theta
        else:
            angle = 90 + theta
    h, w= img.shape[:2]
    if angle!=0:
        M = cv2.getRotationMatrix2D(tuple(box[0]), angle, 1.0)
        outimg = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,borderMode=cv2.BORDER_REPLICATE)
    else:
        outimg = img
    # 裁剪
    x = box[0][1]
    y = box[0][0]
    outimg = outimg[x:x + round(height), y:y + round(width)]
    return outimg

