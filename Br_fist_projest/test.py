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
filtered_imgs = []
for i in range(1, 4):
    img = cv2.imread(f"{i}.jpg")
    if img is None:
        print(f"無法載入影像 {i}.jpg")
        continue
    filtered = cv2.GaussianBlur(img, (5, 5), 0)
    filtered_imgs.append(filtered)
    cv2.imwrite(f"{output_dir}/filtered_{i}.jpg", filtered)

#自適應直方圖均衡化
clahe_imgs = []
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
for i in range(1, 4):
    gray = cv2.cvtColor(filtered_imgs[i-1], cv2.COLOR_BGR2GRAY)
    clahe_img = clahe.apply(gray)
    clahe_imgs.append(clahe_img)
    cv2.imwrite(f"{output_dir}/clahe_{i}.jpg", clahe_img)

# 邊緣檢測
low_threshold = 180
high_threshold = 160

edges_imgs = []
for i in range(1, 4):
    gray = cv2.cvtColor(filtered_imgs[i-1], cv2.COLOR_BGR2GRAY)
    
    # # Sobel 邊緣檢測
    # sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    # sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    # cv2.imwrite(f"{output_dir}/sobelx_{i}.jpg", sobelx)
    # cv2.imwrite(f"{output_dir}/sobely_{i}.jpg", sobely)

    # # Laplacian 邊緣檢測
    # laplacian = cv2.Laplacian(gray, cv2.CV_64F , 1)
    # cv2.imwrite(f"{output_dir}/laplacian_{i}.jpg", laplacian)

    # Canny 邊緣檢測
    edges = cv2.Canny(clahe_imgs[i-1], low_threshold, high_threshold)
    edges = cv2.Canny(filtered_imgs[i-1], low_threshold, high_threshold)

    edges_imgs.append(edges)
    cv2.imwrite(f"{output_dir}/edges_{i}.jpg", edges)

# 二值化
# 有使用canny，所以不再使用二值化

# 型態學處理
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
dilation_imgs = []
erosion_imgs = []
for i in range(1, 4):
    edges = edges_imgs[i-1]
    dilation = cv2.dilate(edges, kernel, iterations=1)
    dilation_imgs.append(dilation)
    cv2.imwrite(f"{output_dir}/dilation_{i}.jpg", dilation)

    erosion = cv2.erode(dilation, kernel, iterations=1)
    erosion_imgs.append(erosion)
    cv2.imwrite(f"{output_dir}/erosion_{i}.jpg", erosion)



# 使用霍夫變換進行直線檢測

for i in range(1, 4):
    img = cv2.imread(f"{i}.jpg")
    if img is None:
        print(f"無法載入影像 {i}.jpg")
        continue
    edges = edges_imgs[i-1]
    erosion = erosion_imgs[i-1]
    dilation = dilation_imgs[i-1]
    # lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=160, maxLineGap=6)
    # lines = cv2.HoughLinesP(erosion, 1, np.pi/180, 100, minLineLength=175, maxLineGap=10)
    lines = cv2.HoughLinesP(dilation, 1, np.pi/180, 100, minLineLength=160, maxLineGap=5)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 4)
    else:
        print(f"無法檢測到直線 {i}.jpg")
    cv2.imwrite(f"{output_dir}/output{i}.jpg", img)


# 繪製車道線


plt.show()
