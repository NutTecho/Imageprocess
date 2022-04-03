from datetime import datetime
import cv2
import numpy as np
from imutils.video import FPS , WebcamVideoStream , FileVideoStream
import imutils
import time
from tesserocr import PyTessBaseAPI ,RIL ,PSM , iterate_level
from PIL import Image


def Concept1():
    stream = cv2.VideoCapture(0)
    fps = FPS().start()

    while True:
        (grabbed , frame) = stream.read()
        frame = imutils.resize(frame,width = 400)

        cv2.putText(frame,"Hello",(10,30),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1)
        cv2.imshow('frame', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

    fps.stop()
    print("[INFO] elasped time : {:.2f}" .format(fps.elapsed()))
    print("[INFO] approx FPS : {:.2f}" .format(fps.fps()))

    stream.release()
    cv2.destroyAllWindows()

def Concept2():
    vs = WebcamVideoStream(src=0).start()
    # vs = FileVideoStream("D:\VSCODE\OpenCVProject\env\highway.mp4").start()
    fps = FPS().start()
    new_frame_time = 0
    prev_frame_time = 0
    ocrResult = ""
    time.sleep(1.0)

    while True:

        new_frame_time = time.time()
        frame = vs.read()
        himg,wimg,_ =  frame.shape
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        reframe = imutils.resize(gray,width = 800)
        # finalframe = np.dstack([reframe, reframe, reframe])
        _,thresh = cv2.threshold(reframe,0,255,cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
        dialat = cv2.dilate(thresh,kernel,iterations=2)
        result = 255 - cv2.bitwise_and(dialat,thresh)

        if (new_frame_time-prev_frame_time > 0):
            fr = 1/(new_frame_time-prev_frame_time)

        prev_frame_time = new_frame_time

        with PyTessBaseAPI(path="D:/VSCODE/OpenCVProject/tessdata-main", psm=PSM.AUTO_OSD ,lang ="eng") as api:
            api.SetVariable("save_blob_choices","T")
            ri=api.GetIterator()
            api.Recognize()
            level = RIL.SYMBOL
            api.SetVariable('preserve_interword_spaces', '1')
            api.SetImage(Image.fromarray(frame))
            # gt = api.GetUTF8Text()
            # cv2.putText(frame,gt,(50,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
            boxs = api.GetComponentImages(RIL.WORD,True)
            # print(boxs)q
            # for r in iterate_level(ri,level):
            #     symbol = r.GetUTF8Text(level)
            #     conf =  r.Confidence(level)
            #     if symbol:
            #         print(u'symbol {}, conf: {}'.format(symbol, conf), end='')
            #     indent = False
            #     ci = r.GetChoiceIterator()
            #     for c in ci:
            #         if indent:
            #             print('\t\t ', end='')
            #         print('\t- ', end='')
            #         choice = c.GetUTF8Text()  # c == ci
            #         print(u'{} conf: {}'.format(choice, c.Confidence()))
            #         indent = True
            #     print('---------------------------------------------')
            for i,(im,b,_,_) in enumerate(boxs):
                x,y,w,h = b['x'],b['y'],b['w'],b['h']
                ocrResult = api.GetUTF8Text()
                # conf = api.MeanTextConf()
                # print( "data out : " + ocrResult)
                cv2.rectangle(frame,(x,y), (x+w,y+h),(255,255,),2)
                
                # print(u"Box[{0}]: x={x}, y={y}, w={w}, h={h}, "
                #     "confidence: {1}, text: {2}".format(i, conf, ocrResult, **b))
            cv2.putText(frame,ocrResult,(10,himg - 25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1) 

        cv2.putText(frame, "FPS: {:.2f}".format(fr),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)	
        cv2.imshow('result', result)
        cv2.imshow('frame', frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        fps.update()

    fps.stop()
    print("[INFO] elasped time : {:.2f}" .format(fps.elapsed()))
    print("[INFO] approx FPS : {:.2f}" .format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()

def Concept3(frame):
    # print(tesserocr.tesseract_version())  # print tesseract-ocr version
    # print(tesserocr.get_languages())  # prints tessdata path and list of available languages
    
    with PyTessBaseAPI(path="D:/VSCODE/OpenCVProject/tessdata-main",lang ="eng") as api:
    # for img in images:
        # api.SetVariable('preserve_interword_spaces', '1')
        himg,wimg,_ =  frame.shape
        api.SetImage(Image.fromarray(frame))
        gt = api.GetUTF8Text()
        print(gt)
        boxs = api.GetComponentImages(RIL.WORD,True)
        # print(boxs)

        for i,(im,b,_,_) in enumerate(boxs):
        # print(b)
        # b = b.split(' ')
        # print(b)
            x,y,w,h = b['x'],b['y'],b['w'],b['h']
            cv2.rectangle(frame,(x,y), (x+w,y+h),(0,0,255),2)
            cv2.putText(frame,gt,(x,himg-y+25),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),1)

        # api.SetImageFile(img)
        # print(api.GetUTF8Text())
    #     print(api.AllWordConfidences())
    # image = Image.open("sample.jpg")
    # print(tesserocr.image_to_text(image))

if __name__ == "__main__":
    Concept2()