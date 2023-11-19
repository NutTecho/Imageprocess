import cv2
import time
import numpy as np
from imutils.video import FPS , WebcamVideoStream , FileVideoStream
import imutils


def main():
    pTime = 0
    cTime = 0
    # === opencv normal read vdo webcam ======
    frameWidth = 320
    frameHeight = 240
    cap = cv2.VideoCapture(0)

    # ==== set fram on aspect raio ======
    # cap.set(3,frameWidth)
    # cap.set(4,frameHeight)

    # ===== imutils increase fps ======
    cap = WebcamVideoStream(src=0).start()
    

    while True:
        frame = cap.read()
        frame = cv2.flip(frame,1)

        # ==== get frame size =====
        himg,wimg,_ =  frame.shape
    
        # ===== resize frame ======
        # reframe = cv2.resize(frame,(frameWidth, frameHeight))
        # frame = imutils.resize(frame,width = 800)
        
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(5,5),0)
        canny = cv2.Canny(blur,10,50)

        scale = 0.5
        imgre = cv2.resize(frame,(0,0),None,scale,scale)
        imggray = cv2.resize(gray,(0,0),None,scale,scale)
        imgblur = cv2.resize(blur,(0,0),None,scale,scale)
        imgcanny = cv2.resize(canny,(0,0),None,scale,scale)
        imgblank = np.zeros_like(imgre)
        imgcontours = imgre.copy()
        contours,hierarchy = cv2.findContours(imgcanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(imgcontours,contours,-1,(0,255,0),2)

        # imgre = cv2.cvtColor(imgre,cv2.COLOR_GRAY2BGR)
        imggray = cv2.cvtColor(imggray,cv2.COLOR_GRAY2BGR)
        imgblur = cv2.cvtColor(imgblur,cv2.COLOR_GRAY2BGR)
        imgcanny = cv2.cvtColor(imgcanny,cv2.COLOR_GRAY2BGR)
        # imgblank = cv2.cvtColor(imgblank,cv2.COLOR_GRAY2BGR)
        # imgcontours = cv2.cvtColor(imgcontours,cv2.COLOR_GRAY2BGR)
        
      
        # imgarr = [[img,gray,blur,canny],
        #   [imgcontours,imgBigcontours,imgWrap,imgThresh],
        #   [imgresult,imgRawDraw,invimgWrap,imgfinal]]

        label = (["rawimage","gray","blur","canny"],
                ["contours","bigcontours","wrapimage","Threshhold"],
                ["result","imagedraw","imgwrap","imagefinal"])

        hor1 = np.hstack((imgre,imggray,imgblur))
        hor2 = np.hstack((imgcanny,imgcontours,imgblank))
        ver = np.vstack((hor1, hor2))
        
        
        
        #=== frame rate monitor========
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        #==============================

        cv2.imshow("stackimage",ver)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()