import cv2
import numpy as np
import inspect
from pprint import pprint


pprint(inspect.getmembers(cv2.cuda_GpuMat()))
# cap = cv2.VideoCapture(0)
# ret, frame1 = cap.read()
# prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
# # hsv = np.zeros_like(frame1)
# # hsv[...,1] = 255
# gpu_frame = cv2.cuda_GpuMat()
# gpu_frame.upload(frame1)

# gpu_hsv = cv2.cuda_GpuMat(gpu_frame.size(),cv2.CV_32FC3)

# while(1):

#     ret, frame2 = cap.read()
#     next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)

#     # Calculates dense optical flow by Farneback method
#     flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
#     gpu_flow = cv2.cuda_Fareback

#     # Computes the magnitude and angle of the 2D vectors
#     mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    
#     # Sets image hue according to the optical flow 
#     # direction
#     hsv[...,0] = ang*180/np.pi/2
    
#     # Sets image value according to the optical flow
#     # magnitude (normalized)
#     hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
    
#     # Converts HSV to RGB (BGR) color representation
#     rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

#     cv2.imshow("frame2",rgb)
#     k = cv2.waitKey(30) & 0xff
#     if k == 27:
#         break
#     # elif k == ord(‘s’):
#     # cv2.imwrite(‘opticalfb.png’,frame2)
#     # cv2.imwrite(‘opticalhsv.png’,rgb)

#     # Updates previous frame
#     prvs = next


# cap.release()
# cv2.destroyAllWindows()