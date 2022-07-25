import numpy as np
import cv2
from mss import mss
from PIL import Image
import time


def nothing():
    pass


if __name__ == '__main__':
    cv2.namedWindow('Hough Transform')
    cv2.createTrackbar('Threshold', 'Hough Transform', 0, 200, nothing)
    with mss() as sct:
        while True:
            last_time = time.time()
            # noinspection PyTypeChecker
            sct_img = np.array(sct.grab(sct.monitors[1]))
            img_gray = cv2.cvtColor(sct_img, cv2.COLOR_BGR2GRAY)
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            img_edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)

            final = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
            threshold = cv2.getTrackbarPos('Threshold', 'Hough Transform')
            linesP = cv2.HoughLinesP(img_edges, 1, np.pi / 180, threshold, None, 50, 10)
            if linesP is not None:
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(final, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
            cv2.imshow("Hough Transform", final)

            print("fps: {}".format(1 / (time.time() - last_time)))
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
