import numpy as np
import cv2 
import matplotlib.pyplot as plt
import os
from PIL import Image


print(f"OpenCV version: {cv2.__version__}")

output_dir = "output"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
print(f"Output directory: {output_dir}")

for i in range(1, 4):
    img = cv2.imread(f"{i}.jpg")
    if img is None:
        print(f"無法載入影像 {i}.jpg")
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{output_dir}/gray_{i}.jpg", gray)
    # plt.imshow(gray, cmap='gray')


# 濾波處理
for i in range(1, 4):
    img = cv2.imread(f"{i}.jpg")
    if img is None:
        print(f"無法載入影像 {i}.jpg")
        continue
    filtered = cv2.GaussianBlur(img, (5, 5), 0)
    cv2.imwrite(f"{output_dir}/filtered_{i}.jpg", filtered)

plt.show()
