# Computer Vision Course (CSE 40535/60535)
# University of Notre Dame, Fall 2024
# ________________________________________________________________
# Adam Czajka, Andrey Kuehlkamp, September 2017 - 2024

# Here are your tasks:
#
# Task 2a:
# - Select one object that you want to track and set the RGB
#   channels to the selected ranges (found by colorSelection.py).
# - Check if HSV color space works better. Can you ignore one or two
#   channels when working in HSV color space? Why?
# - Try to track candies of different colors (blue, yellow, green).
# 
# Task 2b:
# - Adapt your code to track multiple objects of *the same* color simultaneously, 
#   and show them as separate objects in the camera stream.
#
# Task 2c:
# - Adapt your code to track multiple objects of *different* colors simultaneously,
#   and show them as separate objects in the camera stream. Make your code elegant 
#   and requiring minimum changes when the number of different objects to be detected increases.
#
# Task for students attending 60000-level course:
# - Choose another color space (e.g., LAB or YCrCb), modify colorSelection.py, select color ranges 
#   and after some experimentation say which color space was best (RGB, HSV or the additional one you selected).
#   Try to explain the reasons why the selected color space performed best. 

import cv2
import numpy as np

cam = cv2.VideoCapture(0)

while (True):
    retval, img = cam.read()

    res_scale = 0.5 # rescale the input image if it's too large
    img = cv2.resize(img, (0,0), fx = res_scale, fy = res_scale)


    #######################################################
    # Use colorSelection.py to find good color ranges for your object(s):

    # Detect selected color (NOTE: OpenCV uses BGR instead of RGB)
    # This example is tuned to blue, in a relatively dark room
    #lower = np.array([50, 0, 0])
    #upper = np.array([100, 50, 50])
    #objmask = cv2.inRange(img, lower, upper)

    # Uncomment this if you want to use HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    kernel = np.ones((5,5), np.uint8)

    blue_lower = np.array([100,100,50])
    blue_upper = np.array([130,255,255])
    blue_objmask = cv2.inRange(hsv, blue_lower, blue_upper) 
    blue_objmask = cv2.morphologyEx(blue_objmask, cv2.MORPH_OPEN, kernel)
    blue_objmask = cv2.morphologyEx(blue_objmask, cv2.MORPH_CLOSE, kernel=kernel)
    blue_objmask = cv2.morphologyEx(blue_objmask, cv2.MORPH_DILATE, kernel=kernel)
    
    green_lower = np.array([35,100,100])
    green_upper = np.array([85,255,255])
    green_objmask = cv2.inRange(hsv, green_lower, green_upper)
    green_objmask = cv2.morphologyEx(green_objmask, cv2.MORPH_OPEN, kernel)
    green_objmask = cv2.morphologyEx(green_objmask, cv2.MORPH_CLOSE, kernel=kernel)
    green_objmask = cv2.morphologyEx(green_objmask, cv2.MORPH_DILATE, kernel=kernel)
    
    red_lower = np.array([115,100,100])
    red_upper = np.array([135,255,255])
    red_objmask = cv2.inRange(hsv, red_lower, red_upper)
    
    red_lower1 = np.array([0,150,100])
    red_upper1 = np.array([10,255,255])

    red_lower2 = np.array([170,150,100])
    red_upper2 = np.array([180,255,255])

    red_mask1 = cv2.inRange(hsv, red_lower1, red_upper1)
    red_mask2 = cv2.inRange(hsv, red_lower2, red_upper2)
    red_objmask = cv2.bitwise_or(red_mask1, red_mask2)
    red_objmask = cv2.morphologyEx(red_objmask, cv2.MORPH_OPEN, kernel)
    red_objmask = cv2.morphologyEx(red_objmask, cv2.MORPH_CLOSE, kernel=kernel)
    red_objmask = cv2.morphologyEx(red_objmask, cv2.MORPH_DILATE, kernel=kernel)
    #######################################################

    
    # You may use this for debugging
    #cv2.imshow("Binary image", objmask)

    # Resulting binary image may have large number of small objects.
    # You may check different morphological operations to remove these unnecessary
    # elements. You may need to check your ROI defined in step 1 to
    # determine how many pixels your object may have.
   # kernel = np.ones((5,5), np.uint8)
    #objmask = cv2.morphologyEx(objmask, cv2.MORPH_CLOSE, kernel=kernel)
    #objmask = cv2.morphologyEx(objmask, cv2.MORPH_DILATE, kernel=kernel)
    #cv2.imshow("Image after morphological operations", objmask)

    # find connected components
    #cc = cv2.connectedComponents(objmask)
    #ccimg = cc[1].astype(np.uint8)

    # Find contours of these objects
    #contours, hierarchy = cv2.findContours(ccimg,
     #                                           cv2.RETR_TREE,
      #                                          cv2.CHAIN_APPROX_SIMPLE)[-2:]

    # We are using [-2:] to select the last two return values from the above function to make the code work with
    # both opencv3 and opencv4. This is because opencv3 provides 3 return values but opencv4 discards the first.

    # You may display the countour points if you want:
    # cv2.drawContours(img, contours, -1, (0,255,0), 3)

    # Ignore bounding boxes smaller than "minObjectSize"
    minObjectSize = 20;
    
    
    blue_contours, hierarchy = cv2.findContours(blue_objmask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for c in blue_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > minObjectSize or h > minObjectSize:
            cv2.rectangle(img, (x, y), (x+w, y+h), (255,0,0), 3)
            cv2.putText(img,            # image
            "Here's my blue object!",        # text
            (x, y-10),                  # start position
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            0.7,                        # size
            (255, 0, 0),                # BGR color
            1,                          # thickness
            cv2.LINE_AA)                # type of line


    green_contours, hierarchy = cv2.findContours(green_objmask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for c in green_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > minObjectSize or h > minObjectSize:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,255,0), 3)
            cv2.putText(img,            # image
            "Here's my green object!",        # text
            (x, y-10),                  # start position
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            0.7,                        # size
            (0, 255, 0),                # BGR color
            1,                          # thickness
            cv2.LINE_AA)                # type of line


    red_contours, hierarchy = cv2.findContours(red_objmask,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)[-2:]
    for c in red_contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > minObjectSize or h > minObjectSize:
            cv2.rectangle(img, (x, y), (x+w, y+h), (0,0,255), 3)
            cv2.putText(img,            # image
            "Here's my red object!",        # text
            (x, y-10),                  # start position
            cv2.FONT_HERSHEY_SIMPLEX,   # font
            0.7,                        # size
            (0, 0, 255),                # BGR color
            1,                          # thickness
            cv2.LINE_AA)                # type of line
    cv2.imshow("Live WebCam", img)

    action = cv2.waitKey(1)
    if action==27:
        break
    
cam.release()
cv2.destroyAllWindows()
