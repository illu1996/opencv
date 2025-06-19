import numpy as np #numpy 모듈
import cv2 #cv2 모듈


print("cv2 version: "+cv2.__version__) #cv2 version

img_path = "./RAW_02-66_0309.bmp"

img_rgb = cv2.imread(img_path) #이미지 RGB로 읽기
img_gray = cv2.imread(img_path) #이미지 GRAY로 읽기

