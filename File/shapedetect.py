import cv2
import numpy as np

# imgshape = cv2.resize(cv2.imread('picshape.jpg'),(800,600))
cap = cv2.VideoCapture(0)

while True:
    ref,frame = cap.read()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    _,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY)
    contours,hierachy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.01*peri,True)
        cv2.drawContours(frame,[approx],0,(0,0,0),2)
        x = approx.ravel()[0]
        y = approx.ravel()[1] - 5
        # print(len(cnt))
        if(len(approx) == 3 ):
            cv2.putText(frame,"Triangle",(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        if(len(approx) == 4 ):
            x1,y1,w,h = cv2.boundingRect(approx)
            aspectRetio = float(w)/h
            # print(aspectRetio)
            if aspectRetio >= 0.95 and aspectRetio <= 1.05 :
                cv2.putText(frame,"Square",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            else:
                cv2.putText(frame,"Rectangle",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        if(len(approx) == 5 ):
            cv2.putText(frame,"Pentagon",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        if(len(approx) == 6 ):
            cv2.putText(frame,"Hexagon",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        if(len(approx) == 10 ):
            cv2.putText(frame,"Star",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        elif (len(approx) > 10 ):
            cv2.putText(frame,"Circle",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
    
    cv2.imshow('frame', frame)
    cv2.imshow('thres', thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# cv2.imshow("thresh",thresh)
# cv2.imshow("result",imgshape)
# cv2.waitKey(0)
# cv2.destroyAllWindows()