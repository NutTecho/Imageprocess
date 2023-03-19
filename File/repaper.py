import cv2
import numpy as np

def nothing(x):
    pass

def initializeTrackbar(val = 0):
    cv2.namedWindow("TrackBar")
    cv2.resizeWindow("TrackBar",360,240)
    cv2.createTrackbar("Threshold1","TrackBar",200,255,nothing)
    cv2.createTrackbar("Threshold2","TrackBar",200,255,nothing)

def valTackbar():
    Threhold1 = cv2.getTrackbarPos("Threshold1","TrackBar")
    Threshold2 = cv2.getTrackbarPos("Threshold2","TrackBar")
    src = Threhold1,Threshold2
    return src

def biggestContour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area > 5000:
            peri = cv2.arcLength(i,True)
            approx = cv2.approxPolyDP(i,0.02 * peri,True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def drawRectangle(img,biggest,thickness):
    cv2.line(img,(biggest[0][0][0], biggest[0][0][1]), (biggest[1][0][0], biggest[1][0][1]),(0,0,255),thickness)
    cv2.line(img,(biggest[0][0][0], biggest[0][0][1]), (biggest[2][0][0], biggest[2][0][1]),(0,0,255),thickness)
    cv2.line(img,(biggest[3][0][0], biggest[3][0][1]), (biggest[2][0][0], biggest[2][0][1]),(0,0,255),thickness)
    cv2.line(img,(biggest[3][0][0], biggest[3][0][1]), (biggest[1][0][0], biggest[1][0][1]),(0,0,255),thickness)

    return img

def reorder(mypoints):
    mypoint = mypoints.reshape((4,2))
    mypointnew = np.zeros((4,1,2),dtype=np.int32)
    add = mypoint.sum(1)
    mypointnew[0] = mypoint[np.argmin(add)]
    mypointnew[3] = mypoint[np.argmax(add)]
    diff = np.diff(mypoint,axis=1)
    mypointnew[1] = mypoint[np.argmin(diff)]
    mypointnew[2] = mypoint[np.argmax(diff)]
    return mypointnew

cap = cv2.VideoCapture(1)
cap.set(10,160)
widthImage = 640
heightImage = 480
initializeTrackbar()

while True:
    ret,frame = cap.read()
    img = cv2.resize(frame,(widthImage,heightImage))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray,(5,5),1)
    thres = valTackbar()
    imgThres = cv2.Canny(blur,thres[0],thres[1])
    kernal = np.ones((5,5))
    dial = cv2.dilate(thres,kernal,iterations = 2)
    imgThreshold = cv2.erode(dial,kernal,iterations = 1)


    imgContours = img.copy()
    imgBigContour = img.copy()
    imgWarpColored = img.copy()

    contours,hierarchy = cv2.findContours(imgThres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(imgContours,contours,-1,(0,255,0),10)
        
    biggest,maxarea = biggestContour(contours)
    print(biggest)
    if biggest.size != 0:
        biggest = reorder(biggest)
        cv2.drawContours(imgContours,biggest,-1,(0,255,0),20)
        imgBigContour = drawRectangle(imgBigContour,biggest,2)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0,0],[widthImage,0],[0,heightImage],[widthImage,heightImage]])
        matrix = cv2.getPerspectiveTransform(pts1,pts2)
        imgWarpColored = cv2.warpPerspective(img,matrix,(widthImage,heightImage))

        # imgWarpColored=imgWarpColored[20:imgWarpColored.shape[0] - 20, 20:imgWarpColored.shape[1] - 20]
        # imgWarpColored = cv2.resize(imgWarpColored,(widthImage,heightImage))

    cv2.imshow('normal', imgContours)
    cv2.imshow('biggest', imgBigContour)
    cv2.imshow('wrap', imgWarpColored)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()