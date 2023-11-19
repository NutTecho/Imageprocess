import cv2
import numpy as np


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

def warpImg(img,points,w,h,pad = 20):
    points = reorder(points)
    pts1 = np.float32(points)
    pts2 = np.float32([[0,0],[w,0],[0,h],[w,h]])
    matrix = cv2.getPerspectiveTransform(pts1,pts2)
    imgWarp = cv2.warpPerspective(img,matrix,(widthImage,heightImage))
    imgWarp = imgWarp[pad:imgWarp.shape[0]-pad,pad:imgWarp.shape[1]-pad]
    
    return imgWarp

def findDis(pts1,pts2):
    return ((pts2[0] - pts1[0])**2 + (pts2[1] - pts1[1]) ** 2) ** 0.5

def getContours(img,cThr=[100,100],showCanny=False,minArea=1000,filter=0,draw =False):
    ret,frame = cap.read()
    img = cv2.resize(frame,(widthImage,heightImage))
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY) 
    blur = cv2.GaussianBlur(gray,(5,5),1)
    # thres = valTackbar()
    imgCanny = cv2.Canny(blur,cThr[0],cThr[1])
    kernal = np.ones((5,5))
    dial = cv2.dilate(imgCanny,kernal,iterations = 2)
    imgThres = cv2.erode(dial,kernal,iterations = 1)
    finalCountours = []
    contours,hierachy = cv2.findContours(imgThres,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > minArea :
            peri = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.01*peri,True)
            # cv2.drawContours(frame,[approx],0,(0,0,0),2)
            bbox = cv2.boundingRect(approx)
            if Filter > 0 and len(approx) == filter:
                finalCountours.append([len(approx),area,approx,bbox,cnt])
            else:
                finalCountours.append([len(approx),area,approx,bbox,cnt])

            finalCountours = sorted(finalCountours,key = lambda x:x[1] ,reverse= True)
            if draw:
                for con in finalCountours:
                    cv2.drawContours(img,con[4],-1,(0,0,255),3)
    return img,finalCountours

cap = cv2.VideoCapture(1)
widthImage = 640
heightImage = 480
cThr=[100,100]
minArea=1000
Filter=0


while True:
    
    imgContours , conts = getContours(cap,minArea=50000,filter=4)           

    if len(conts) != 0:
        for obj in conts:
            cv2.polylines(imgContours,[obj[2]],True,(0,255,0),2)
            nPoints = utlis.reorder(obj[2])
            nW = round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
            nH = round((utlis.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
            cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
            cv2.arrowedLine(imgContours, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),(255, 0, 255), 3, 8, 0, 0.05)
            x, y, w, h = obj[3]
            cv2.putText(imgContours, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)
            cv2.putText(imgContours, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (255, 0, 255), 2)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()