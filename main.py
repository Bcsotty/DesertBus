import numpy as np
import cv2
from mss import mss
import time
import keyboard
import sys
import timeit


def nothing(t):
    pass


# Yellow color filter for image passed through, uses values determined via trackbar in open-cv window.
# Will return the image with just the yellow lines.
def yellow_filter(image):
    mask = cv2.inRange(cv2.cvtColor(image, cv2.COLOR_BGR2HSV), (20, 40, 200), (40, 255, 255))
    filtered = cv2.bitwise_and(image, image, mask=mask)
    return filtered

def edge_detection(image):
    # Applies blurring then edge detection
    img_blur = cv2.GaussianBlur(image, (3, 3), 0) # try 5,5 kernel
    img_edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)
    return img_edges


def image_processing(image):
    # begin_tim = time.time()
    # Resizing image to reduce amount of time needed to process the image
    resized = cv2.resize(image, (64, 64))

    # Creating a color mask to get rid of noise in the image, results in just the lines on the road
    mask = cv2.inRange(cv2.cvtColor(resized, cv2.COLOR_BGR2HSV), (20, 40, 200), (40, 255, 255))
    filtered = cv2.bitwise_and(resized, resized, mask=mask)
    # print("Time after yellow filter + resizing: {}".format(time.time()-begin_tim))
    # Splitting the image channels to get the grayscale version of the image
    h, s, v, _ = cv2.split(filtered)
    img_gray = v

    edges = edge_detection(img_gray)
    # print("Time after edge detection: {}".format(time.time()-begin_tim))

    slope = draw_image(edges, cv2.getTrackbarPos("Threshold", "Hough Transform"), cv2.getTrackbarPos("minLineLength", "Hough Transform"), cv2.getTrackbarPos("maxLineGap", "Hough Transform"))
    # print("Time after draw function: {}".format(time.time()-begin_tim))
    return slope


def draw_image(image, threshold=10, minLineLength=7, maxLineGap=3):
    final = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    linesP = cv2.HoughLinesP(image, 1, np.pi / 180, threshold, None, minLineLength, maxLineGap)
    slope = 0
    if linesP is not None:
        # Draws the lines from the hough transform onto the image to be shown in open-cv window
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(final, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 1, cv2.LINE_AA)
        line = linesP[0][0]
        slope = (line[3] - line[1]) / (line[2] - line[0]) if line[2] - line[0] != 0 else 0
    final = cv2.resize(final, (600, 600))
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(final, f"fps: {int(1 / (time.time() - begin_tim))}", (10, 500), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.putText(final, f"Points: {counter}", (10, 450), font, 1, (255, 255, 255), 2,
                cv2.LINE_AA)
    cv2.imshow("Hough Transform", final)
    return slope

setup = '''
import cv2
from mss import mss
import numpy as np
from __main__ import edge_detection
from __main__ import edge_detection2

with mss() as sct:
    sct_img = np.array(sct.grab((241, 236, 1150, 809)))
    resized = cv2.resize(sct_img, (64, 64))
'''

code1 = '''
t = edge_detection(resized)
'''
code2 = '''
t = edge_detection2(resized)
'''

if __name__ == '__main__':  
    # 2.5-second grace period to tab into the game
    cv2.namedWindow("Hough Transform", cv2.WINDOW_AUTOSIZE)
    time.sleep(2.5)

    cv2.createTrackbar("Threshold", "Hough Transform", 10, 255, nothing)
    cv2.createTrackbar("minLineLength", "Hough Transform", 7, 255, nothing)
    cv2.createTrackbar("maxLineGap", "Hough Transform", 3, 255, nothing)
    increment = True
    counter = sys.argv[0] if isinstance(sys.argv[0], int) else 0
    # Main loop that gets screenshot and processes it
    with mss() as sct:
        while True:
            begin_tim = time.time()

            # Holds down the 'a' key to make bus go forward
            keyboard.press('a')

            keyboard.release('left')
            keyboard.release('right')

            # noinspection PyTypeChecker
            # Grabs screenshot using specific BBOX, as long as emulator is windowed maximized, it will grab full left
            # windshield. Then passes the screenshot through the yellow filter.
            sct_img = np.array(sct.grab((241, 236, 1150, 809)))

            s = image_processing(sct_img)
            # resized = cv2.resize(sct_img, (300, 300))
            # yellow_filtered = yellow_filter(resized)

            # # Taking the HSV image, and obtaining the greyscale version of the image by splitting channels.
            # h, s, v, _ = cv2.split(yellow_filtered)
            # img_gray = v
            # # Blurs the image and runs Canny edge detection.
            # img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            # img_edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)

            # # Creating the final image to draw the hough transform results on, and running the actual hough transform.
            # final = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2BGR)
            # linesP = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 10, None, 7, 3)

            # if linesP is not None:
            #     # Draws the lines from the hough transform onto the image to be shown in open-cv window
            #     for i in range(0, len(linesP)):
            #         l = linesP[i][0]
            #         cv2.line(final, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
            #     # Just grabbing the first located line and using its values to find the slope of the line
            #     line = linesP[0][0]
            #     slope = (line[3] - line[1]) / (line[2] - line[0]) if line[2] - line[0] != 0 else 0
            #     # Depending on the slope of the line, move a specific direction to stay on road. Needs to be adjusted,
                # as of now the car will sway back and forth so ideally would like to make movement smoother
            if s < 0:
                # pass
                keyboard.press('left')
            # if s > 0:
                # pass
                # keyboard.press('right')
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(final, f"fps: {int(1 / (time.time() - begin_tim))}", (10, 500), font, 1, (255, 255, 255), 2,
            #             cv2.LINE_AA)
            # cv2.putText(final, f"Points: {counter}", (10, 450), font, 1, (255, 255, 255), 2,
            #             cv2.LINE_AA)
            # cv2.imshow("Hough Transform", final)
            keyboard.release('left')
            time.sleep(0.05)
            keyboard.release('a')

            end_check = np.array(sct.grab((794, 560, 795, 561)))
            if 62 < end_check[0][0][0] < 67 and 65 < end_check[0][0][1] < 70 and 29 < end_check[0][0][2] < 34:
                if increment:
                    counter += 1
                    increment = not increment
                #keyboard.press('enter')
                time.sleep(0.1)
                #keyboard.release('enter')
            else:
                increment = True
            # Checks if Q key is pressed while open-cv window is focused, then closes it and ends the program
            if cv2.waitKey(25) & 0xFF == ord("q"):
                cv2.destroyAllWindows()
                break
