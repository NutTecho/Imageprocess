import cv2
import numpy as np


def empty(img):
    pass

cap = cv2.VideoCapture(1)
cv2.namedWindow("TrackBar")
cv2.resizeWindow("TrackBar",600,300)
cv2.createTrackbar("hue_min","TrackBar",0,179,empty)
cv2.createTrackbar("hue_max","TrackBar",179,179,empty)
cv2.createTrackbar("sat_min","TrackBar",0,255,empty)
cv2.createTrackbar("sat_max","TrackBar",255,255,empty)
cv2.createTrackbar("val_min","TrackBar",0,255,empty)
cv2.createTrackbar("val_max","TrackBar",255,255,empty)


while True:
    ret,frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    hue_min = cv2.getTrackbarPos("hue_min","TrackBar")
    hue_max = cv2.getTrackbarPos("hue_max","TrackBar")
    sat_min = cv2.getTrackbarPos("sat_min","TrackBar")
    sat_max = cv2.getTrackbarPos("sat_max","TrackBar")
    val_min = cv2.getTrackbarPos("val_min","TrackBar")
    val_max = cv2.getTrackbarPos("val_max","TrackBar")

    # red_lower = np.array([hue_min,sat_min,val_min],np.uint8)
    # red_upper = np.array([hue_max,sat_max,val_max],np.uint8)
    # red_mask = cv2.inRange(hsv,red_lower,red_upper)

    # green_lower = np.array([25, 52, 72], np.uint8)
    # green_upper = np.array([102, 255, 255], np.uint8)
    # green_mask = cv2.inRange(hsv, green_lower, green_upper)
  

    blue_lower = np.array([94, 80, 2], np.uint8)
    blue_upper = np.array([120, 255, 255], np.uint8)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)

    # Morphological Transform, Dilation
    # for each color and bitwise_and operator
    # between imageFrame and mask determines
    # to detect only that particular color

    kernal = np.ones((5,5),"uint8")
    # red_mask = cv2.dilate(red_mask,kernal)
    # res_red = cv2.bitwise_and(frame,frame,mask = red_mask)

    # green_mask = cv2.dilate(green_mask,kernal)
    # res_green = cv2.bitwise_and(frame,frame,mask = green_mask)

    blue_mask = cv2.dilate(blue_mask,kernal)
    res_blue = cv2.bitwise_and(frame,frame,mask = blue_mask)


    contours,hierachy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    for pic,cnt in enumerate(contours):
        area = cv2.contourArea(cnt)
        x,y,w,h = cv2.boundingRect(cnt)
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
        # cv2.putText(frame,"Red_Color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

        # if (area >  2000)  :
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.02*peri,True)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),1)
        # cv2.putText(frame,"Red_Color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

        if len(approx) == 4 :
            cv2.putText(frame,"Rectanglr",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
        elif len(approx) > 4:
            cv2.putText(frame,"Circle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

    # contours,hierachy = cv2.findContours(green_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # for pic,cnt in enumerate(contours):
    #     area = cv2.contourArea(cnt)
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
    #     # cv2.putText(frame,"Red_Color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

    #     if (area >  2000)  :
    #        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),1)
    #        cv2.putText(frame,"Green_Color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    # contours,hierachy = cv2.findContours(blue_mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    # for pic,cnt in enumerate(contours):
    #     area = cv2.contourArea(cnt)
    #     x,y,w,h = cv2.boundingRect(cnt)
    #     # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),1)
    #     # cv2.putText(frame,"Red_Color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)

        # if (area >  2000)  :
        #    cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),1)
        #    cv2.putText(frame,"Blue_Color",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),1)

    cv2.imshow("SHOW",frame)
    cv2.imshow("res",blue_mask)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()