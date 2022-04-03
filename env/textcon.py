import cv2
import numpy as np
import pytesseract

# cap = cv2.VideoCapture('rtsp://192.168.43.1:8080/h264_pcm.sdp')
cap = cv2.VideoCapture(1)
pytesseract.pytesseract.tesseract_cmd = 'D:/tesseract-ocr/tesseract.exe'
config = "--psm 4"

while (True):
    ref,frame = cap.read()
    himg,wimg,_ =  frame.shape()
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(gray,240,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    # ==== for white font ==========
    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # lower = np.array([0,0,218])
    # upper = np.array([157,54,255])
    # mask = cv2.inRange(hsv,lower,upper)

    # ========== get corner============
    # canny = cv2.Canny(blur,10,50)
    # imgThres = cv2.Canny(blur,thres[0],thres[1])
    # kernal = np.ones((5,5))
    # dial = cv2.dilate(canny,kernal,iterations = 2)
    # imgThreshold = cv2.erode(dial,kernal,iterations = 1)
    # adth = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV ,85,11)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,3))
    dialat = cv2.dilate(thresh,kernel,iterations=3)

    result = 255 - cv2.bitwise_and(dialat,thresh)
    boxes = pytesseract.image_to_boxes(result)
    textdata = pytesseract.image_to_string(result,config=config)

    contours,hierachy = cv2.findContours(dialat,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
    #     # area = cv2.contourArea(cnt)
        (x,y,w,h) = cv2.boundingRect(cnt)
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        # cv2.putText(frame,textdata,(x,himg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

      # boxes = pytesseract.image_to_boxes(adth)
    # boxes = pytesseract.image_to_boxes(reault)
    # print(data)
    for b in boxes.splitlines():
        # print(b)
        b = b.split(' ')
        # print(b)
        t,x,y,w,h = b[0],int(b[1]),int(b[2]),int(b[3]),int(b[4])
        # cv2.rectangle(frame,(x,himg-y),(w,himg-h),(0,0,255),2)
        cv2.putText(frame,t,(x,himg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)
    #     peri = cv2.arcLength(cnt,True)
    #     approx = cv2.approxPolyDP(cnt,0.01*peri,True)
    #     cv2.drawContours(frame,[approx],0,(0,0,0),2)
    #     x = approx.ravel()[0]
    #     y = approx.ravel()[1] - 5
    #     if (len(approx) > 10 ):
    #         cv2.putText(frame,"Circle",(x,y),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
            # cv2.putText(frame,str(area),(x,y),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)

    cv2.imshow("SHOW",frame)
    cv2.imshow("dialat",dialat)
    cv2.imshow("thresh",thresh)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()