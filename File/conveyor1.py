import cv2
cap = cv2.VideoCapture(0)

while True:
    ret,frame = cap.read()
    # belt = frame[209:327,137:530]
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,threshold = cv2.threshold(gray,80,255,cv2.THRESH_BINARY)

    contours,hierachy = cv2.findContours(threshold,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        (x,y,w,h) = cv2.boundingRect(cnt)

        if area > 3000 :
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            # cv2.putText(frame,str(area),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        else :
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        
        cv2.putText(frame,str(area),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)

        
    cv2.imshow('frame', frame)
    cv2.imshow('thres', threshold)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()