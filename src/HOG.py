import cv2
import numpy as np


class HOG:
    def __init__(self, winSize):
        """

        :param winSize: 检测窗口的大小
        """
        self.winSize = winSize
        self.blockSize = (8, 8)
        self.blockStride = (4, 4)
        self.cellSize = (4, 4)
        self.nBins = 9
        self.hog = cv2.HOGDescriptor(winSize, self.blockSize, self.blockStride,
                                     self.cellSize, self.nBins)

    def get_feature(self, image):
        winStride = self.winSize
        w, h = self.winSize
        w_block, h_block = self.blockStride
        w = w//w_block - 1
        h = h//h_block - 1
        # 计算给定图像的HOG特征描述子，一个n*1的特征向量
        hist = self.hog.compute(img=image, winStride=winStride, padding=(0, 0))
        return hist.reshape(w, h, 36).transpose(2, 1, 0)    # 交换轴的顺序

    def show_hog(self, hog_feature):
        c, h, w = hog_feature.shape
        feature = hog_feature.reshape(2, 2, 9, h, w).sum(axis=(0, 1))
        grid = 16
        hgrid = grid // 2
        img = np.zeros((h*grid, w*grid))

        for i in range(h):
            for j in range(w):
                for k in range(9):
                    x = int(10 * feature[k, i, j] * np.cos(x=np.pi / 9 * k))
                    y = int(10 * feature[k, i, j] * np.sin(x=np.pi / 9 * k))
                    cv2.rectangle(img=img, pt1=(j*grid, i*grid), pt2=((j + 1) * grid, (i + 1) * grid),
                                  color=(255, 255, 255))
                    x1 = j * grid + hgrid - x
                    y1 = i * grid + hgrid - y
                    x2 = j * grid + hgrid + x
                    y2 = i * grid + hgrid + y
                    cv2.line(img=img, pt1=(x1, y1), pt2=(x2, y2), color=(255, 255, 255), thickness=1)
        cv2.imshow("img", img)
        cv2.waitKey(0)


def tesHog():
    img = cv2.imread("../data/lena1.jpg")
    # cv2.imshow("img1", img[:,
    # print(img.shape) :, 0])
    hog = HOG((64, 128))

    feature = hog.get_feature(img)
    print(feature.shape)

# tesHog()

