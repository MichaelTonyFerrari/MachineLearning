import cv2
import numpy as np
from matplotlib import pyplot as plt


# Some constants

SCALING_FACTOR = 6

lower_canny = 200
upper_canny = 600
trackbar_lower = 0
trackbar_higher = 800
lower_hue = 10
upper_hue = 30
lower_saturation = 0
upper_saturation = 255
lower_value = 0
upper_value = 255

def lower_canny_callback(num):
    global lower_canny
    lower_canny = num

def upper_canny_callback(num):
    global upper_canny
    upper_canny = num

def lower_hue_callback(num):
    global lower_hue
    lower_hue = num

def upper_hue_callback(num):
    global upper_hue
    upper_hue = num
def lower_saturation_callback(num):
    global lower_saturation
    lower_saturation = num

def upper_saturation_callback(num):
    global upper_saturation
    upper_saturation = num

def lower_value_callback(num):
    global lower_value
    lower_value = num

def upper_value_callback(num):
    global upper_value
    upper_value = num

def show(img):
    resized = cv2.resize(img, (len(img[0]) / SCALING_FACTOR, len(img) / SCALING_FACTOR))
    cv2.namedWindow("window", cv2.WINDOW_AUTOSIZE)
    cv2.imshow("window", resized)
    cv2.waitKey(0)


def auto_canny(image, sigma=0.2):
    # compute the median of the single channel pixel intensities
    v = np.median(image)
    # apply automatic Canny edge detection using the computed median
    lower = int(max((0, (1.0 - sigma) * v)))
    upper = int(min((255, (1.0 + sigma) * v)))
    edged = cv2.Canny(image, lower, upper)

    # return the edged image
    return edged

def  main():

    # Processing
    # load and show the original image
    src = cv2.imread("sidewalk_green_border.png")
    #img = cv2.resize(img, (len(img[0]) / SCALING_FACTOR, len(img) / SCALING_FACTOR))

    # blur the image first
    src = cv2.GaussianBlur(src, (3,3), 0)
    cv2.imshow("img",src)

    cv2.namedWindow("edges")
    cv2.createTrackbar('lower', 'edges', trackbar_lower, trackbar_higher, lower_canny_callback)
    cv2.setTrackbarPos("lower","edges",lower_canny)
    cv2.createTrackbar('upper', 'edges', trackbar_lower, trackbar_higher, upper_canny_callback)
    cv2.setTrackbarPos("upper", "edges", upper_canny)

    cv2.namedWindow("HSV")
    cv2.createTrackbar('lower hue', 'HSV', 0, 180, lower_hue_callback)
    cv2.setTrackbarPos("lower hue", "HSV", lower_hue)
    cv2.createTrackbar('upper hue', 'HSV', 0, 180, upper_hue_callback)
    cv2.setTrackbarPos("upper hue", "HSV", upper_hue)
    cv2.createTrackbar('lower sat', 'HSV', 0, 255, lower_saturation_callback)
    cv2.setTrackbarPos("lower sat", "HSV", lower_saturation)
    cv2.createTrackbar('upper sat', 'HSV', 0, 255, upper_saturation_callback)
    cv2.setTrackbarPos("upper sat", "HSV", upper_saturation)
    cv2.createTrackbar('lower val', 'HSV', 0, 255, lower_value_callback)
    cv2.setTrackbarPos("lower val", "HSV", lower_value)
    cv2.createTrackbar('upper val', 'HSV',0, 255, upper_value_callback)
    cv2.setTrackbarPos("upper val", "HSV", upper_value)


    while cv2.waitKey(10) != 27:

        img = src.copy()

        # convert to hsv
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        cv2.imshow("HSV", hsv)

        # mask the image
        # define range of sidewalk color in HSV
        lower_threshold = np.array([lower_hue, lower_saturation, lower_value])
        upper_threshold = np.array([upper_hue, upper_saturation, upper_value])

        # construct the mask
        mask = cv2.inRange(hsv, lower_threshold, upper_threshold)
        cv2.imshow("mask", mask)
        # mask = cv2.bitwise_not(mask)


        # apply it to the original image
        masked_image = cv2.bitwise_and(img, img, mask=mask)
        cv2.imshow("masked_image", masked_image)

        # run canny
        res = cv2.cvtColor(masked_image, cv2.COLOR_HSV2BGR)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(res,lower_canny,upper_canny)
        cv2.imshow("edges",np.hstack([edges]))


        cnts, tmp = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cnts = sorted(cnts, key = cv2.contourArea, reverse= True)

        for idx in range(0,   min(10,len(cnts))):

            shape = cnts[idx]
            print cv2.contourArea(shape)

            cv2.drawContours(img, [shape], -1, (0,255,0), 3)
            cv2.imshow("contours",img)


if __name__ == "__main__":
    main()



#
# lines = cv2.HoughLinesP(edges,
#                         1,
#                         np.pi/180,
#                         100)
#
# if not lines.any():
#     exit()
#
#
# for x in range(0, len(lines)):
#     for x1,y1,x2,y2 in lines[x]:
#         cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)
#
# cv2.imshow('hough',img)
# cv2.waitKey(0)
#
#

cv2.waitKey(0)
