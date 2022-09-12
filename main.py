import numpy as np
import cv2
from mss import mss
import time
import keyboard
import sys
import threading
from queue import Queue, Empty


def press_a():
    while running:
        keyboard.press('a')
        time.sleep(0.05)
        keyboard.release('a')


def movement(queue):
    slope = 0
    while running:
        keyboard.release('left')
        try:
            slope = queue.get(block=False)
        except Empty:
            slope = slope / 2 + 0.25
        if slope < 0:
            keyboard.press('left')
        time.sleep(0.008)


def edge_detection(image):
    # Applies blurring then edge detection
    img_blur = cv2.GaussianBlur(image, (3, 3), 0)
    img_edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)
    return img_edges


def image_processing(image):
    # Resizing image to reduce amount of time needed to process the image
    resized = cv2.resize(image, (64, 64))

    # Creating a color mask to get rid of noise in the image, results in just the lines on the road
    mask = cv2.inRange(cv2.cvtColor(resized, cv2.COLOR_BGR2HSV), (20, 40, 200), (40, 255, 255))
    filtered = cv2.bitwise_and(resized, resized, mask=mask)

    # Splitting the image channels to get the grayscale version of the image
    h, s, v, _ = cv2.split(filtered)
    img_gray = v

    edges = edge_detection(img_gray)

    slope = draw_image(edges)
    return slope


def draw_image(image, threshold=10, minLineLength=7, maxLineGap=3):
    linesP = cv2.HoughLinesP(image, 1, np.pi / 180, threshold, None, minLineLength, maxLineGap)
    slope = 0
    if linesP is not None:
        line = linesP[0][0]
        slope = (line[3] - line[1]) / (line[2] - line[0]) if line[2] - line[0] != 0 else 0
    return slope


if __name__ == '__main__':  
    global running 
    running = True
    queue = Queue()
    # 2.5-second grace period to tab into the game
    time.sleep(2.5)

    t1 = threading.Thread(target=press_a)
    t2 = threading.Thread(target=movement, args=(queue,))
    t1.start()
    t2.start()
    increment = True
    counter = sys.argv[0] if isinstance(sys.argv[0], int) else 0
    # Main loop that gets screenshot and processes it
    try:
        with mss() as sct:
            while True:
                begin_tim = time.time()

                # Grabs screenshot using specific BBOX, as long as emulator is windowed maximized, it will grab full left
                # windshield. Then passes the screenshot through the yellow filter.
                sct_img = np.array(sct.grab((241, 236, 1150, 809)))

                s = image_processing(sct_img)
                
                queue.put(s)

                print(f"fps: {int(1 / (time.time() - begin_tim))}")
                time.sleep(0.025)
                end_check = np.array(sct.grab((794, 560, 795, 561)))
                if 62 < end_check[0][0][0] < 67 and 65 < end_check[0][0][1] < 70 and 29 < end_check[0][0][2] < 34:
                    if increment:
                        counter += 1
                        increment = not increment
                    keyboard.press('enter')
                    time.sleep(0.05)
                    keyboard.release('enter')
                else:
                    increment = True
                # Checks if Q key is pressed while open-cv window is focused, then closes it and ends the program
                if cv2.waitKey(25) & 0xFF == ord("q"):
                    cv2.destroyAllWindows()
                    break
    except KeyboardInterrupt:
        running = False
        t1.join()
        t2.join()
