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
        reframe = cv2.resize(frame,(frameWidth, frameHeight))
        # frame = imutils.resize(frame,width = 800)


        # ====== crop image [Heigh , width]=====
        cropframe = reframe[100:150,0:]

        # ====== zoom crop =====
        zoomcrop = cv2.resize(cropframe,(himg,wimg))
        
        
        
        #=== frame rate monitor========
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime

        cv2.putText(frame,str(int(fps)),(10,70),cv2.FONT_HERSHEY_PLAIN,3,(0,0,255),2)
        #==============================

        cv2.imshow('frame',frame)
        cv2.imshow('reframe',reframe)
        cv2.imshow('cropframe',cropframe)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # cap.release()
    cap.stop()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()