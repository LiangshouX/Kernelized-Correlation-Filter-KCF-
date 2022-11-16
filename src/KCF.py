import cv2
import numpy as np
from numpy.fft import fft2, ifft2, fftshift
from HOG import HOG

class Tracker:
    def __init__(self):
        # 超参数设置
        self.max_patch_size = 256
        self.padding = 2.5
        self.sigma = 0.6
        self.lambdar = 0.0001
        self.update_rate = 0.012
        self.gray_feature = False
        self.debug = False

        # 算法变量定义
        self.scale_h = 0.
        self.scale_w = 0.

        self.ph = 0
        self.pw = 0
        self.hog = HOG((self.pw, self.pw))
        self.alphaf = None
        self.x = None
        self.roi = None

    def first_frame(self, image, roi):
        """
        对视频的第一帧进行标记，更新tracer的参数
        :param image: 第一帧图像
        :param roi: 第一帧图像的初始ROI元组
        :return: None
        """
        x1, y1, w, h = roi
        cx = x1 + w // 2
        cy = y1 + h // 2
        roi = (cx, cy, w, h)

        # 确定Patch的大小，并在此Patch中提取HOG特征描述子
        scale = self.max_patch_size / float(max(w, h))
        self.ph = int(h * scale) // 4 * 4 + 4
        self.pw = int(w * scale) // 4 * 4 + 4
        self.hog = HOG((self.pw, self.ph))

        # 在矩形框的中心采样、提取特征
        x = self.get_feature(image, roi)
        y = self.gaussian_peak(x.shape[2], x.shape[1])

        self.alphaf = self.train(x, y, self.sigma, self.lambdar)
        self.x = x
        self.roi = roi

    def update(self, image):
        """
        对给定的图像，重新计算其目标的位置
        :param image:
        :return:
        """
        # 包含矩形框信息的四元组(min_x, min_y, w, h)
        cx, cy, w, h = self.roi
        max_response = -1   # 最大响应值

        for scale in [0.95, 1.0, 1.05]:
            # 将ROI值处理为整数
            roi = map(int, (cx, cy, w * scale, h * scale))

            z = self.get_feature(image, roi)    # tuple(36, h, w)
            # 计算响应
            responses = self.detect(self.x, z, self.sigma)
            height, width = responses.shape
            if self.debug:
                cv2.imshow("res", responses)
                cv2.waitKey(0)
            idx = np.argmax(responses)
            res = np.max(responses)
            if res > max_response:
                max_response = res
                dx = int((idx % width - width / 2) / self.scale_w)
                dy = int((idx / width - height / 2) / self.scale_h)
                best_w = int(w * scale)
                best_h = int(h * scale)
                best_z = z

        # 更新矩形框的相关参数
        self.roi = (cx + dx, cy + dy, best_w, best_h)

        # 更新模板
        self.x = self.x * (1 - self.update_rate) + best_z * self.update_rate
        y = self.gaussian_peak(best_z.shape[2], best_z.shape[1])
        new_alphaf = self.train(best_z, y, self.sigma, self.lambdar)
        self.alphaf = self.alphaf * (1 - self.update_rate) + new_alphaf * self.update_rate

        cx, cy, w, h = self.roi
        return cx - w // 2, cy - h // 2, w, h

    def get_feature(self, image, roi):
        """
        对特征进行采样
        :param image:
        :param roi: 包含矩形框信息的四元组(min_x, min_y, w, h)
        :return:
        """
        # 对矩形框做2.5倍的Padding处理
        cx, cy, w, h = roi
        w = int(w*self.padding)//2*2
        h = int(h*self.padding)//2*2
        x = int(cx - w//2)
        y = int(cy - h//2)

        # 矩形框所覆盖的距离
        sub_img = image[y:y+h, x:x+w, :]
        resized_img = cv2.resize(src=sub_img, dsize=(self.pw, self.ph))

        if self.gray_feature:
            feature = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
            feature = feature.reshape(1, self.ph, self.pw)/255.0 - 0.5
        else:
            feature = self.hog.get_feature(resized_img)
            if self.debug:
                self.hog.show_hog(feature)

        # Hog特征的通道数、高估、宽度
        fc, fh, fw = feature.shape
        self.scale_h = float(fh)/h
        self.scale_w = float(fw)/w

        # 两个二维数组，前者(fh，1)，后者(1，fw)
        hann2t, hann1t = np.ogrid[0:fh, 0:fw]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (fw - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (fh - 1)))

        # 一个fh x fw的矩阵
        hann2d = hann2t * hann1t

        feature = feature * hann2d
        return feature

    def gaussian_peak(self, w, h):
        """

        :param w:
        :param h:
        :return:      一个w*h的高斯矩阵
        """
        output_sigma = 0.125
        sigma = np.sqrt(w * h) / self.padding * output_sigma
        syh, sxh = h//2, w//2

        y, x = np.mgrid[-syh:-syh + h, -sxh:-sxh + w]
        x = x + (1 - w % 2) / 2.
        y = y + (1 - h % 2) / 2.
        g = 1. / (2. * np.pi * sigma ** 2) * np.exp(-((x ** 2 + y ** 2) / (2. * sigma ** 2)))
        return g

    def kernel_correlation(self, x1, x2, sigma):
        """
        核化的相关滤波操作
        :param x1:
        :param x2:
        :param sigma:   高斯参数sigma
        :return:
        """
        # 转换到傅里叶空间
        fx1 = fft2(x1)
        fx2 = fft2(x2)
        # \hat{x^*} \otimes \hat{x}'，x*的共轭转置与x'的乘积
        tmp = np.conj(fx1) * fx2
        # 离散傅里叶逆变换转换回真实空间
        idft_rbf = ifft2(np.sum(tmp, axis=0))
        # 将零频率分量移到频谱中心。
        idft_rbf = fftshift(idft_rbf)

        # 高斯核的径向基函数
        d = np.sum(x1 ** 2) + np.sum(x2 ** 2) - 2.0 * idft_rbf
        k = np.exp(-1 / sigma ** 2 * np.abs(d) / d.size)
        return k

    def train(self, x, y, sigma, lambdar):
        """
        原文所给参考train函数
        :param x:
        :param y:
        :param sigma:
        :param lambdar:
        :return:
        """
        k = self.kernel_correlation(x, x, sigma)
        return fft2(y) / (fft2(k) + lambdar)

    def detect(self, x, z, sigma):
        """
        原文所给参考detect函数
        :param x:
        :param z:
        :param sigma:
        :return:
        """
        k = self.kernel_correlation(x, z, sigma)
        # 傅里叶逆变换的实部
        return np.real(ifft2(self.alphaf * fft2(k)))

