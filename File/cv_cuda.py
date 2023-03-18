from operator import truediv
from unittest import result
import cv2
import numpy as np
from imutils.video import FPS , WebcamVideoStream , FileVideoStream
import imutils
import time
from tesserocr import PyTessBaseAPI ,RIL ,PSM , iterate_level
from PIL import Image

# vdo = WebcamVideoStream(src=0).start()
# fps = FPS().start()
# ocrResult = ""
# vdo = cv2.VideoCapture(0)
# fps = vdo.get(cv2.CAP_PROP_FPS)
# num_frame = vdo.get(cv2.CAP_PROP_FRAME_COUNT)
# new_frame_time = 0
# prev_frame_time = 0
# gpu_frame = cv2.cuda_GpuMat()

# while True:
#     new_frame_time = time.time()
#     frame = vdo.read()
#     gpu_frame.upload(frame)
#     resize = cv2.cuda.resize(gpu_frame,(640,480))
#     gray = cv2.cuda.cvtColor(resize,cv2.COLOR_BGR2GRAY)
#     _,thresh = cv2.cuda.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV )
#     # kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(15,3))
#     # dialat = cv2.dilate(thresh,kernel,iterations=2)
#     # result = 255 - cv2.cuda.bitwise_and(dialat,thresh)
#     # luv = cv2.cuda.cvtColor(resize, cv2.COLOR_BGR2LUV)
#     # hsv = cv2.cuda.cvtColor(resize, cv2.COLOR_BGR2HSV)
#     # gray = cv2.cuda.cvtColor(resize, cv2.COLOR_BGR2GRAY)
    

#     # resize.download()
#     # luv = luv.download()
#     # hsv = hsv.download()
#     gray = gray.download()
#     thresh = thresh.download()

#     # frame = vdo.read()

#     # with PyTessBaseAPI(path="D:/VSCODE/OpenCVProject/tessdata-main", psm=PSM.AUTO_OSD ,lang ="eng") as api:
#     #     api.SetVariable("save_blob_choices","T")
#     #     ri=api.GetIterator()
#     #     api.Recognize()
#     #     level = RIL.SYMBOL
#     #     api.SetVariable('preserve_interword_spaces', '1')
#     #     api.SetImage(Image.fromarray(frame))
#     #     # gt = api.GetUTF8Text()
#     #     # cv2.putText(frame,gt,(50,30),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),1)
#     #     boxs = api.GetComponentImages(RIL.WORD,True)
#     #     for i,(im,b,_,_) in enumerate(boxs):
#     #             x,y,w,h = b['x'],b['y'],b['w'],b['h']
#     #             ocrResult = api.GetUTF8Text()
#     #             # conf = api.MeanTextConf()
#     #             # print( "data out : " + ocrResult)
#     #             cv2.rectangle(frame,(x,y), (x+w,y+h),(255,255,),2)
    
#     if (new_frame_time-prev_frame_time > 0):
#         fr = 1/(new_frame_time-prev_frame_time)
#     prev_frame_time = new_frame_time
#         # fr = (cv2.getTickCount() - starttime)/ cv2.getTickFrequency()
#     cv2.putText(thresh, "FPS: {:.2f}".format(fr),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)	
#     cv2.imshow("thresh",thresh)
#     cv2.imshow("gray",gray)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#     fps.update()

# fps.stop()
# print("[INFO] elasped time : {:.2f}" .format(fps.elapsed()))
# print("[INFO] approx FPS : {:.2f}" .format(fps.fps()))
# # vdo.release()    
# vdo.stop()
# cv2.destroyAllWindows()

def test():
    video_stream = WebcamVideoStream(src=0).start()
    fps = FPS().start()
    fps.start()
    new_frame_time = 0
    prev_frame_time = 0
    # gpu_frame = cv2.cuda_GpuMat()
    while True:
        frame = video_stream.read()
        row,column = frame.shape[:2]
        bg = np.empty((row,column,3),np.uint8)
        stream = cv2.cuda_Stream()
        # cv2.dnn.readNet
       
        cuda_bg = cv2.cuda_GpuMat(row,column,cv2.CV_8UC3)
        cuda_bg.upload(frame,stream)

        gray = cv2.cuda.cvtColor(cuda_bg,cv2.COLOR_BGR2GRAY,stream = stream)
        gray.download(stream,bg)
        stream.waitForCompletion()

        # cv2.putText(frame, "FPS: {:.2f}".format(fr),(10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)	
        cv2.imshow("frame",frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break               
        fps.update()

    fps.stop()
    print("[INFO] elasped time : {:.2f}" .format(fps.elapsed()))
    print("[INFO] approx FPS : {:.2f}" .format(fps.fps()))
    # vdo.release()    
    video_stream.stop()
    cv2.destroyAllWindows()



class get_color_spaces(object):
    def __init__(self, cuda=False):
        self.cuda = cuda
        self._initialized = False


    def _initialize_color_space(self, bgr_frame):
        print("Allocating Memory")
        self.rows, self.columns = bgr_frame.shape[:2]
        self.bgr_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        self.hsv_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        self.lab_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        self.YCrCb_frame = np.empty((self.rows, self.columns, 3), np.uint8)
        if self.cuda:
            self.stream = cv2.cuda_Stream()
            self.cuda_bgr_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                  cv2.CV_8UC3)
            self.cuda_hsv_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                  cv2.CV_8UC3)
            self.cuda_lab_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                  cv2.CV_8UC3)
            self.cuda_YCrCb_frame = cv2.cuda_GpuMat(self.rows, self.columns,
                                                    cv2.CV_8UC3)
        print("Finished Allocating Memory")


    def do_color_spaceing(self,frame):
        if not self._initialized:
            self._initialize_color_space(frame)
            self._initialized = True
        self.bgr_frame = frame
        if self.cuda:
            self.cuda_bgr_frame.upload(self.bgr_frame, self.stream)
            self.cuda_hsv_frame = cv2.cuda.cvtColor(self.cuda_bgr_frame,
                                                    cv2.COLOR_BGR2HSV,
                                                    stream=self.stream)
            self.cuda_lab_frame = cv2.cuda.cvtColor(self.cuda_bgr_frame,
                                                    cv2.COLOR_BGR2Lab,
                                                    stream=self.stream)
            self.cuda_YCrCb_frame = cv2.cuda.cvtColor(self.cuda_bgr_frame,
                                                      cv2.COLOR_BGR2YCrCb,
                                                      stream=self.stream)
            self.cuda_hsv_frame.download(self.stream, self.hsv_frame)
            self.cuda_lab_frame.download(self.stream, self.lab_frame)
            self.cuda_YCrCb_frame.download(self.stream, self.YCrCb_frame)
            self.stream.waitForCompletion()
        else:
            self.hsv_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2HSV)
            self.lab_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2Lab)
            self.YCrCb_frame = cv2.cvtColor(self.bgr_frame, cv2.COLOR_BGR2YCrCb)

        return (self.hsv_frame, self.lab_frame, self.YCrCb_frame)

def main():
    """Run color space application"""
    text = "Color Space"
    FILE = "race.mp4"
    CUDA = False
    fps = FPS().start()
    try:
        # with WebcamVideoStream(src=0).start() as video_stream,\
        #         Streamer() as streamer:
        #     # check video stream input fps
        #     print(video_stream._thread._fps)
        #     time.sleep(2.0)
            video_stream = WebcamVideoStream(src=0).start()
  
            color_space = get_color_spaces(cuda=CUDA)
            fps.start()
            while True:
                frame = video_stream.read()
                hsv, lab, YCrCb = color_space.do_color_spaceing(frame)
                combined = np.hstack((frame, hsv))
                combined_2 = np.hstack((lab, YCrCb))
                combined_3 = np.vstack((combined, combined_2))
                # streamer.send_data(combined_3)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break               
                fps.update()
                # if streamer.check_exit():
                #     break

    finally:
        fps.stop()
        print("[INFO] elasped time : {:.2f}" .format(fps.elapsed()))
        print("[INFO] approx FPS : {:.2f}" .format(fps.fps()))
        print("Program Ending")




if __name__ == "__main__":
    # main()
    # print(cv2.cuda_DeviceInfo()[0])
    test()