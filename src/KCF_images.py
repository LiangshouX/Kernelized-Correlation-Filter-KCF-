import cv2
from KCF import Tracker
import time

if __name__ == '__main__':

    base_path = "../data/football/"
    tracker = Tracker()
    first_img = cv2.imread(base_path+"00000001.jpg")
    cv2.imshow("first", first_img)
    roi = cv2.selectROI(img=first_img)
    tracker.first_frame(first_img, roi)

    for i in range(2, 272):
        if i < 10:
            path = base_path + "0000000" + str(i) + ".jpg"
        else:
            path = base_path + "000000" + str(i) + ".jpg"

        img = cv2.imread(path)

        x, y, w, h = tracker.update(img)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 1)
        cv2.imshow('tracking', img)

        c = cv2.waitKey(0)







        if c == 27 or c == ord('q'):
            continue
