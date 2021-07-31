import cv2
import numpy as np
cap = cv2.VideoCapture(1)

while (True):
    ref,frame = cap.read()
    roi = frame[:600,0:800]

    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(15,15),0)
    _,thresh = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)
    # thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,1)
    # kernal = np.ones((2.5,2.5),np.uint8)
    # closing=cv2.morphologyEx(thresh,cv2.MORPH_CLOSE,kernal,iterations=5)

    # result_img = closing.copy()
    contours,hierachy = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    counter = 0
    onebath = 0
    tenbath = 0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x,y,w,h) = cv2.boundingRect(cnt)
        
        if area > 2000  and area < 3000  :
            # continue
            cv2.putText(roi,'1',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            tenbath+=1
        else :
            cv2.putText(roi,'10',(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
            onebath+=1
        
        # ellipse = cv2.fitEllipse(cnt)
        # cv2.ellipse(roi,ellipse,(0,255,0),2)
        counter = (tenbath *10 )+ onebath 
        # cv2.putText(roi,str(area),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

    cv2.putText(roi,str(counter),(10,100),cv2.FONT_HERSHEY_SIMPLEX,2,(255,0,0),2,cv2.LINE_AA)
    cv2.imshow("SHOW",roi)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()