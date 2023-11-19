import cv2
import numpy as np

def stackimage(scale,imageArr,labels):
    rows = len(imageArr)
    cols = len(imageArr[0])
    rowAvailable = isinstance(imageArr[0],list)
    width = imageArr[0][0].shape[1]
    height = imageArr[0][0].shape[0]
    if rowAvailable:
        for x in range(0,rows):
            for y in range(0,cols):
                if imageArr[x][y].shape[:2] == imageArr[0][0].shape[:2]:
                    imageArr[x][y] = cv2.resize(imageArr[x][y],(0,0),None,scale,scale)
                else:
                    imageArr[x][y] = cv2.resize(imageArr[x][y],(imageArr[0][0].shape[1],imageArr[0][0].shape[0]),None,scale,scale)
                if len(imageArr[x][y].shape) == 2:
                    imageArr[x][y] = cv2.cvtColor(imageArr[x][y],cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height,width,3),np.uint8)
        hor = [imageBlank]*rows
        hor_cons = [imageBlank]*rows
        for x in range(0,rows):
            hor[x] = np.hstack(imageArr[x])
        ver = np.vstack(hor) 
    
    else:
        for x in range(0,rows):
            if imageArr[x].shape[:2] == imageArr[0].shape[:2]:
                imageArr[x] = cv2.resize(imageArr[x],(0,0),None,scale,scale)
            else:
                imageArr[x] = cv2.resize(imageArr[x],(imageArr[0].shape[1],imageArr[0].shape[0]),None,scale,scale)
            if len(imageArr[x].shape) == 2:
                imageArr[x] = cv2.cvtColor(imageArr[x],cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imageArr)
        ver = hor
    if len(labels) != 0:
        eachImageWidth = int(ver.shape[1]/cols)
        eachImageHeight = int(ver.shape[0]/rows)
        for d in range(0,rows):
            for c in range(0,cols):
                cv2.rectangle(ver,(c*eachImageWidth,eachImageHeight*d),(c*eachImageWidth+len(labels[d][c])+13*25,30+eachImageHeight*d),(255,255,255),cv2.FILLED)
                cv2.putText(ver,labels[d][c],(eachImageWidth*c+10,eachImageHeight*d+20),cv2.FONT_HERSHEY_COMPLEX,0.7,(255,0,255),2)
    
    return ver