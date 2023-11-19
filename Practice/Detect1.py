import cv2
import time
import numpy as np
from imutils.video import FPS , WebcamVideoStream , FileVideoStream
import imutils
from StackImage import stackimage


def empty():
    pass

def getContours(img,imgcontour):
    contours,hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 1000:
            cv2.drawContours(imgcontour,cnt,-1,(0,255,0),5)
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02 * peri,True)
            x,y,w,h = cv2.boundingRect(approx)
            cv2.rectangle(imgcontour,(x,y),(x+w,y+h),(0,255,0),5)

def main():
    pTime = 0
    cTime = 0
    # === opencv normal read vdo webcam ======
    frameWidth = 320
    frameHeight = 240
    # cap = cv2.VideoCapture(0)

    # ==== set fram on aspect raio ======
    # cap.set(3,frameWidth)
    # cap.set(4,frameHeight)

    # ===== imutils increase fps ======
    cap = WebcamVideoStream(src=0).start()

    cv2.namedWindow("Parameters")
    cv2.resizeWindow("Parameters",640,240)
    cv2.createTrackbar("Threshold1","Parameters",8,255,empty)
    cv2.createTrackbar("Threshold2","Parameters",150,255,empty)
    

    while True:
        frame = cap.read()
        frame = cv2.flip(frame,1)

        # ==== get frame size =====
        himg,wimg,_ =  frame.shape
        frame = cv2.resize(frame,(frameWidth, frameHeight))
        imgcontour = frame.copy()
        threshold1 = cv2.getTrackbarPos("Threshold1","Parameters")
        threshold2 = cv2.getTrackbarPos("Threshold2","Parameters")

        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        canny = cv2.Canny(blur,threshold1,threshold2)
        kernel = np.ones((5,5))
        dil = cv2.dilate(canny,kernel,iterations=1)

        getContours(dil,imgcontour)
        

        labels = (["frame","gray","canny"],
                  ["dil","imgcontour","dil"])
        
        stack = stackimage(0.8,([frame,gray,canny],
                                [dil,imgcontour,dil]),labels)
        
        #=== frame rate monitor========
        # cTime = time.time()
        # fps = 1/(cTime-pTime)
        # pTime = cTime

        # cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        #==============================

        cv2.imshow('Result',stack)
      
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()