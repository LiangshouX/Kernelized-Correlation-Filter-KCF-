# 源码结构说明

**姓名：肖良寿**  **学号：1120200563**





* HOG.py ：借助OpenCV封装的HOGDescriptor类提取图像的HOG特征，在KCF.py中调用它
* KCF.py：  实现KCF算法的主文件，实现了Tracker类，含有KCF算法的各个实现部分，在runKCF.py与KCF_image.py中调用它
* runKCF.py： 以视频作为数据
* KCF_image.py： 以图片数据集作为数据