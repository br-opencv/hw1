import cv2
import os

img = cv2.imread("2.jpg")
if img is None:
    print(f"無法載入影像 2.jpg")

width = int(img.shape[1] * 0.5)
height = int(img.shape[0] * 0.5)
dim = (width, height)

resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
cv2.imwrite("low_res_2.jpg", resized)