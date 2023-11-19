import cv2
from imutils.video import FPS , WebcamVideoStream 
from scipy.spatial import distance as dist
import scipy
import time
import imutils
from imutils import contours , perspective
import numpy as np

def nothing(x):
    pass


def initializeTrackbar(val = 0):
    cv2.namedWindow("TrackBar")
    cv2.resizeWindow("TrackBar",360,200)
    cv2.createTrackbar("Threshold1","TrackBar",1,100,nothing)
    cv2.createTrackbar("Threshold2","TrackBar",1,100,nothing)


    # cv2.createTrackbar("Threshold2","TrackBar",200,255,nothing)

def valTackbar():
    Threhold1 = cv2.getTrackbarPos("Threshold1","TrackBar")
    Threshold2 = cv2.getTrackbarPos("Threshold2","TrackBar")
    src = Threhold1,Threshold2
    return src

def midpoint(ptA, ptB):
	return ((ptA[0] + ptB[0]) * 0.5, (ptA[1] + ptB[1]) * 0.5)


def reorder(mypoints):
    mypoint = mypoints.reshape((4,2))
    mypointnew = np.zeros((4,2),dtype=np.int32)
    add = mypoint.sum(1)
    mypointnew[0] = mypoint[np.argmin(add)]
    mypointnew[3] = mypoint[np.argmax(add)]
    diff = np.diff(mypoint,axis=1)
    mypointnew[1] = mypoint[np.argmin(diff)]
    mypointnew[2] = mypoint[np.argmax(diff)]
    return mypointnew

def test(arr1,arr2):
    p1 = np.array(arr1)
    p2 = np.array(arr2)
    d = np.linalg.norm(p2-p1)
    return d


def object_detect():
    vs = WebcamVideoStream(src=1).start()
  
    new_frame_time = 0
    prev_frame_time = 0
    fr = 0
    
    fps = FPS().start()
    # initializeTrackbar()

    while(True):
        new_frame_time = time.time()
        pixelsPerMetric = None
        frame = vs.read()
        frame = imutils.resize(frame,width = 600)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray,(7,7),0)

        #  ======= find corner ================
        # thres = valTackbar()
        imgCanny = cv2.Canny(blur,80,200)
        kernal = np.ones((2,2))
        dial = cv2.dilate(imgCanny,kernal,iterations = 1)
        imgThres = cv2.erode(dial,kernal,iterations = 1)

        #  ======= find area contour ============
        # _,thresh = cv2.threshold(blur,240,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)


        try:
            imgcontours = cv2.findContours(imgThres,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            imgcontours = imutils.grab_contours(imgcontours)
            (imgcontours , _) = contours.sort_contours(imgcontours)

            for cnt in imgcontours:
                area = cv2.contourArea(cnt)
                if area < 100:
                    continue
                rect = cv2.minAreaRect(cnt)
                (x,y) , (w,h) , angle = rect
                box = cv2.BoxPoints(rect) if imutils.is_cv2() else cv2.boxPoints(rect) 
                # box = np.array(box, dtype="int")
                # box = np.int0(box)q
                # box = reorder(box)
                box = perspective.order_points(box)
                cv2.circle(frame,(int(x),int(y)),5,(0,0,255),-1)
                cv2.polylines(frame,[box.astype("int")],True,(0,255,0),2)

                for (x, y) in box:
                    cv2.circle(frame, (int(x), int(y)), 5, (0, 255, 64), -1)

                # ====== generate grid in rectacgle ========
                (tl, tr, br, bl) = box
                (tltrX, tltrY) = midpoint(tl, tr)
                (blbrX, blbrY) = midpoint(bl, br)

                (tlblX, tlblY) = midpoint(tl, bl)
                (trbrX, trbrY) = midpoint(tr, br)


                #  ======= draw midpoint =======
                cv2.circle(frame, (int(tltrX), int(tltrY)), 0, (0, 255, 64), 0)
                cv2.circle(frame, (int(blbrX), int(blbrY)), 0, (0, 255, 64), 0)
                cv2.circle(frame, (int(tlblX), int(tlblY)), 0, (0, 255, 64), 0)
                cv2.circle(frame, (int(trbrX), int(trbrY)), 0, (0, 255, 64), 0)

                # ===== draw line between modpoint ===========

                cv2.line(frame, (int(tltrX), int(tltrY)), (int(blbrX), int(blbrY)),(255, 255, 255), 1)
                cv2.line(frame, (int(tlblX), int(tlblY)), (int(trbrX), int(trbrY)),(255, 255, 255), 1)


                # dA = test([tltrX, tltrY], [blbrX, blbrY])
                # dB = test([tlblX, tlblY], [trbrX, trbrY])

                dA = dist.euclidean((tltrX, tltrY), (blbrX, blbrY))
                dB = dist.euclidean((tlblX, tlblY), (trbrX, trbrY))

                if pixelsPerMetric is None:
                    pixelsPerMetric = dB / 20
                    # pixelsPerMetric = dB / ((thres[0] * 10) + (thres[1] * 0.1))

                dimA = dA / pixelsPerMetric
                dimB = dB / pixelsPerMetric

                cv2.putText(frame, "{:.1f}mm".format(dimA),
                    (int(tltrX - 15), int(tltrY - 10)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)
                cv2.putText(frame, "{:.1f}mm".format(dimB),
                    (int(trbrX + 10), int(trbrY)), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2)

                # x,y,w,h = cv2.boundingRect(cnt)
                # cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            # peri = cv2.arcLength(cnt,True)
            # approx = cv2.approxPolyDP(cnt,0.01*peri,True)
            # cv2.drawContours(frame,[approx],0,(0,0,0),2)
            # x = approx.ravel()[0]
            # y = approx.ravel()[1] - 5
        except Exception as e:
            print(e)
            continue




        if (new_frame_time-prev_frame_time > 0):
            fr = 1/(new_frame_time-prev_frame_time)

        prev_frame_time = new_frame_time

        cv2.putText(frame,"FPS: {:.2f}".format(fr),(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2)
        cv2.imshow('frame', frame)
        # cv2.imshow('imgCanny', imgCanny)
        # cv2.imshow('dial', dial)
        # cv2.imshow('blur', blur)
        cv2.imshow('thresh', imgThres)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

    fps.stop()

    cv2.destroyAllWindows()
    vs.stop()

def cal1():
    a = (1, 2, 3)
    b = (4, 5, 6)
    dst = dist.euclidean(a, b)
    print(dst)



def main():
    object_detect()
    # cal1()



if __name__ == "__main__":
    main()