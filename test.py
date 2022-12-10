import numpy as np
import cv2
import matplotlib.pyplot as plt
# read image through command line
img = cv2.imread("/home/dtan/catkin_ws/src/deliver_rb/IMG_8611.JPG")
h, w = int(img.shape[0]), int(img.shape[1])

ratio = w/h
start_point =(400,900) #x,y
h_cut = 2700

img = img[start_point[1]:start_point[1]+h_cut,start_point[0]:start_point[0]+int(h_cut*ratio)] # cut follow y:y+h,x:x+w
img = cv2.resize(img, (w//4,h//4))

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (7,7),0)
sobelx = abs(cv2.Sobel(gray,cv2.CV_64F,1,0,ksize=5))
sobely = abs(cv2.Sobel(gray,cv2.CV_64F,0,1,ksize=5))
plt.subplot(1,3,1)
plt.imshow(sobelx,cmap='gray')
plt.subplot(1,3,2)
plt.imshow(sobely,cmap='gray')
fn = np.sqrt(sobelx**2+sobely**2)
max = fn.max()
plt.subplot(1,3,3)
plt.imshow((fn/max)*255,cmap='gray')

plt.show()