import numpy as np
import cv2
from mss import mss
from PIL import Image
import time
import keyboard


def nothing():
    pass


# Yellow color filter for image passed through, uses values determined via trackbar in open-cv window.
# Will return the image with just the yellow lines.
def yellow_filter(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (20, 180, 200), (40, 200, 250))
    filtered = cv2.bitwise_and(image, image, mask=mask)
    return filtered


if __name__ == '__main__':
    # 2.5-second grace period to tab into the game
    time.sleep(2.5)

    # Main loop that gets screenshot and processes it
    with mss() as sct:
        while True:
            # noinspection PyTypeChecker
            # Grabs screenshot using specific BBOX, as long as emulator is windowed maximized, it will grab full left
            # windshield. Then passes the screenshot through the yellow filter.
            sct_img = np.array(sct.grab((241, 236, 1150, 809)))
            yellow_filtered = yellow_filter(sct_img)

            # Taking the HSV image, and obtaining the greyscale version of the image by splitting channels.
            h, s, v, _ = cv2.split(yellow_filtered)
            img_gray = v
            # Blurs the image and runs Canny edge detection.
            img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)
            img_edges = cv2.Canny(image=img_blur, threshold1=50, threshold2=200)

            # Creating the final image to draw the hough transform results on, and running the actual hough transform.
            final = img_edges
            linesP = cv2.HoughLinesP(img_edges, 1, np.pi / 180, 50, None, 30, 20)

            if linesP is not None:
                # Draws the lines from the hough transform onto the image to be shown in open-cv window
                for i in range(0, len(linesP)):
                    l = linesP[i][0]
                    cv2.line(final, (l[0], l[1]), (l[2], l[3]), (255, 0, 0), 3, cv2.LINE_AA)
                # Just grabbing the first located line and using its values to find the slope of the line
                line = linesP[0][0]
                slope = (line[3] - line[1]) / (line[2] - line[0]) if line[2] - line[0] != 0 else 0
                # Depending on the slope of the line, move a specific direction to stay on road. Needs to be adjusted,
                # as of now the car will sway back and forth so ideally would like to make movement smoother
                if slope < 0:
                    print("left")
                    keyboard.press('k')
                    time.sleep(0.1)
                    keyboard.release('k')
                elif slope > 0:
                    print("right")
                    keyboard.press('l')
                    time.sleep(0.1)
                    keyboard.release('l')
                else:
                    print("straight")

            # Holds down the 'a' key to make bus go forward
            keyboard.press('a')
            cv2.imshow("Hough Transform", final)

            # Checks if Q key is pressed while open-cv window is focused, then closes it and ends the program
            if cv2.waitKey(25) & 0xFF == ord("q"):
                keyboard.release('a')
                cv2.destroyAllWindows()
                break
