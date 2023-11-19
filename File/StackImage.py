import cv2
import numpy as np

# cap = cv2.VideoCapture(0)
# while True:
#     ref,frame = cap.read()
#     gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

#     cv2.imshow("test",gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()


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
        # img = cv2.imread('picshape.jpg')

def reccontour(contours):
    reccon = []
    for cnt in contours:
        peri = cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,0.01*peri,True)
        # x = approx.ravel()[0]
        # y = approx.ravel()[1] - 5
        if(len(approx) == 4 ):
            reccon.append(cnt)
    reccon = sorted(reccon,key=cv2.contourArea,reverse=True)
    return reccon

def reorder(mypoints):
    mypoint = mypoints.reshape((4,2))
    mypointnew = np.zeros((4,1,2),dtype=np.int32)
    # print(mypoint)
    add = mypoint.sum(1)
    # print(add)
    mypointnew[0] = mypoint[np.argmin(add)] #[0,0]
    mypointnew[3] = mypoint[np.argmax(add)] #[w,h]
    diff = np.diff(mypoint,axis=1)
    mypointnew[1] = mypoint[np.argmin(diff)] #[w,0]
    mypointnew[2] = mypoint[np.argmax(diff)] #[0,h]
    # print(diff)
    return mypointnew

def getCornerpoint(cont):
    peri = cv2.arcLength(cont,True)
    approxx = cv2.approxPolyDP(cont,0.02 * peri,True)
    return approxx

def splitbox(img):
    boxes = []
    rows = np.vsplit(img,5)
    for r in rows:
        cols = np.hsplit(r,5)
        for box in cols:
            boxes.append(box)
    # cv2.imshow("split",rows[0])
    return boxes

def showAnswer(img,myindex,grading,ans,question,choices):
    secW = int(img.shape[1]/question)
    secH = int(img.shape[0]/choices)

    for x in range(0,question):
        myAns = myindex[x]
        cx = (myAns*secW) + secW // 2
        cy = (x*secH) + secH//2

        if grading[x] == 1:
            checkcolor = (0,255,0)
        else:
            checkcolor = (0,0,255)
            collectAns = ans[x]
            cxw = (collectAns*secW) + secW // 2
            cyw = (x*secH) + secH//2
            cv2.circle(img,(cxw,cyw),20,(0,255,0),cv2.FILLED)

        cv2.circle(img,(cx,cy),50,checkcolor,cv2.FILLED)
    return img

#=======initial setting===================
img = cv2.imread('D:/VSCODE/OpenCVProject/Image/exam.jpg')
width = 700
height = 700
question = 5
choices = 5
ans = [1,2,0,1,4]
#=======================================

#===========basic setting=========
img = cv2.resize(img,dsize= (width,height))
imgcontours = img.copy()
imgfinal = img.copy()
imgBigcontours = img.copy()
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray,(5,5),0)
canny = cv2.Canny(blur,10,50)
#==================================

contours,hierarchy = cv2.findContours(canny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(imgcontours,contours,-1,(0,255,0),5)

blank = np.zeros_like(img)

#=====get area contour================
reccon = reccontour(contours)
biggestContour = getCornerpoint(reccon[0])
gradepoint = getCornerpoint(reccon[1])
#====================================
 
if biggestContour.size != 0 and gradepoint.size != 0:
    cv2.drawContours(imgBigcontours,biggestContour,-1,(0,0,255),20)
    cv2.drawContours(imgBigcontours,gradepoint,-1,(0,0,255),20)

    #=======wrap image of exam===============
    biggestContour = reorder(biggestContour)
    pt1 = np.float32(biggestContour)
    pt2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(pt1,pt2)
    imgWrap = cv2.warpPerspective(img,matrix,(width,height))
    
    #========wrap image of grad inut========
    gradepoint = reorder(gradepoint)
    ptg1 = np.float32(gradepoint)
    ptg2 = np.float32([[0,0],[325,0],[0,150],[325,150]])
    matrixg = cv2.getPerspectiveTransform(ptg1,ptg2)
    imgWrapg = cv2.warpPerspective(img,matrixg,(325,150))


    #=====apply threshold from wrap image============
    imgWrapGRay = cv2.cvtColor(imgWrap,cv2.COLOR_BGR2GRAY)
    imgThresh = cv2.threshold(imgWrapGRay,150,255,cv2.THRESH_BINARY_INV)[1]

    #====split image ===================
    boxes = splitbox(imgThresh)
    mypixel = np.zeros((question,choices))
    countRow = 0
    countCol = 0
    for choice in boxes:
        totalpixel = cv2.countNonZero(choice)
        mypixel[countRow][countCol] = totalpixel
        countCol += 1
        if countCol == choices:
            countRow += 1
            countCol = 0

    #get max contour for check true false============
    myindex = []
    for x in range(0,question):
        arr = mypixel[x]
        myindexval = np.where(arr == np.amax(arr))
        myindex.append(myindexval[0][0])


    #====== check score add to list================
    grading = []
    for x in range(0,question):
        if ans[x] == myindex[x]:
            grading.append(1)
        else:
            grading.append(0)


    #========sum score=========
    score = sum(grading)/question * 100
    # print(score)
    
    
    #=========display score check============
    imgresult = imgWrap.copy()
    imgresult = showAnswer(imgresult,myindex,grading,ans,question,choices)

    #=======unwrap image exam==========
    imgRawDraw = np.zeros_like(imgWrap)
    imgRawDraw = showAnswer(imgRawDraw,myindex,grading,ans,question,choices)
    invmatrix = cv2.getPerspectiveTransform(pt2,pt1)
    invimgWrap = cv2.warpPerspective(imgRawDraw,invmatrix,(width,height))
   
   #===========unwrap image score========
    imgRawGrad = np.zeros_like(imgWrapg)
    cv2.putText(imgRawGrad,str(int(score)) + "%",(50,100),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,255),2)
    invmatrixg = cv2.getPerspectiveTransform(ptg2,ptg1)
    invimgWrapg = cv2.warpPerspective(imgRawGrad,invmatrixg,(width,height))

    #======overlay image eam and score
    imgfinal = cv2.addWeighted(imgfinal,1,invimgWrap,1,0)
    imgfinal = cv2.addWeighted(imgfinal,1,invimgWrapg,1,0)

imgarr = [[img,gray,blur,canny],
          [imgcontours,imgBigcontours,imgWrap,imgThresh],
          [imgresult,imgRawDraw,invimgWrap,imgfinal]]

label = (["rawimage","gray","blur","canny"],
        ["contours","bigcontours","wrapimage","Threshhold"],
        ["result","imagedraw","imgwrap","imagefinal"])

stk = stackimage(0.3,(imgarr),label)
cv2.imshow("test",stk)
# cv2.imshow("final",imgfinal)
# cv2.imshow("grade",imgWrapg)
cv2.waitKey(0)