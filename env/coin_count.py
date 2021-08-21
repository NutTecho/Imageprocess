import cv2
import numpy as np
cap = cv2.VideoCapture('rtsp://192.168.43.1:8080/h264_pcm.sdp')

while (True):
    ref,frame = cap.read()
    # roi = frame[:600,0:800]

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(15,15),0)
    _,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
    # kernal = np.ones((2.5,2.5),np.uint8)
    # closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernal,iterations=5)

    # result_img = closing.copy()
    contours,hierachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    counter = 0
    onebath = 0
    tenbath = 0
    for cnt in contours:
        # ===== get area and detect shape==============
        area = cv2.contourArea(cnt)
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.01*peri,True)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5

        #  ======= if shape is cycle detected ==============
        if (len(approx) > 10 ):
            (x,y,w,h) = cv2.boundingRect(cnt)
            # cv2.drawContours(frame,[approx],0,(0,0,0),2)    
            # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if area > 10000  :
                # continue
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,'10',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                tenbath+=1

            elif area > 6500 and area < 9500 :
                cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
                cv2.putText(frame,'1',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
                onebath+=1
            # elif area > 4500 and area < 5000:
            # ellipse = cv2.fitEllipse(cnt)
            # cv2.ellipse(roi,ellipse,(0,255,0),2)
            counter = (tenbath * 10 )+ onebath 
            # cv2.putText(roi,str(area),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(frame,str(counter),(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("SHOW",frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()