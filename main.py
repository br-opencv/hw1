import cv2
import numpy as np
import os

folder_path = 'output'
os.makedirs(folder_path, exist_ok=True)

# 1. 載入影像
n = 3
max_width = 1080
max_height = 720

for num in range(1, n + 1):
    filename = f'{num}.jpg'
    img = cv2.imread(filename)

    if img is None:
        print(f"無法載入影像 {filename}")
        continue

    h, w = img.shape[:2]

    # 計算比例
    scale = min(max_width / w, max_height / h)

    # 計算尺寸
    new_width = int(w * scale)
    new_height = int(h * scale)

    # 縮放
    img = cv2.resize(img, (new_width, new_height))

    # 2. 灰階轉換
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(folder_path + f'/gray_{num}.jpg', gray)

    # 3. 濾波
    filtered = cv2.GaussianBlur(gray, (7, 7), 0)
    cv2.imwrite(folder_path + f'/filter_{num}.jpg', filtered)

    # 4. 邊緣檢測
    low_threshold = 100
    high_threshold = 150
    edges = cv2.Canny(filtered, low_threshold, high_threshold)
    cv2.imwrite(folder_path + f'/edge_{num}.jpg', edges)

    # 5. 二值化
    ret, binary = cv2.threshold(edges, 150, 255, cv2.THRESH_BINARY)
    cv2.imwrite(folder_path + f'/bin_{num}.jpg', binary)

    # 6. 型態學處理
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
    cv2.imwrite(folder_path + f'/morphology_{num}.jpg', morph)

    # 7. 直線偵測
    lines = cv2.HoughLinesP(morph, rho=3, theta=np.pi/180, threshold=100, minLineLength=120, maxLineGap=10)
    line_img = img.copy()
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
    else:
        print("未偵測到直線")

    cv2.imwrite(folder_path + f'/line_{num}.jpg', line_img)