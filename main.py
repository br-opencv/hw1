import cv2
import numpy as np
# 輸入影像(1.jpg) -> 灰階轉換(gray_1.jpg) -> 濾波(filter_1.jpg) -> 邊緣檢測(edge _1.jpg) -> 二值化(bin_1.jpg) -> 形態學(morphology_1.jpg) -> 直線偵測 -> 繪製車道線(line_1.jpg)

# 1. 載入輸入影像
img = cv2.imread('1.jpg')
if img is None:
    print("無法載入影像 1.jpg")
    exit()

# 2. 灰階轉換
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite('gray_1.jpg', gray)

# 3. 濾波 (使用高斯模糊降低雜訊)
# 注意: 根據影像特性，kernel 大小可以調整 (例如 (5,5) )
filtered = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imwrite('filter_1.jpg', filtered)

# 4. 邊緣檢測 (使用 Canny 邊緣檢測)
# 可根據影像情況調整 low_threshold 與 high_threshold
low_threshold = 50
high_threshold = 100
edges = cv2.Canny(filtered, low_threshold, high_threshold)
cv2.imwrite('edge_1.jpg', edges)

# 5. 二值化 (這裡直接使用全局閥值，也可用adaptiveThreshold)
# 由於 Canny 已經產生的影像為二值影像，可視需求做額外量化處理
ret, binary = cv2.threshold(edges, 50, 255, cv2.THRESH_BINARY)
cv2.imwrite('bin_1.jpg', binary)

# 6. 型態學處理 (使用閉運算來補足斷裂處，連接邊線)
# 定義結構元，大小可根據實際情況調整
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,5))
morph = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=1)
cv2.imwrite('morphology_1.jpg', morph)

# 7. 直線偵測 (使用霍夫直線變換)
# 邊線偵測影像為 morph
lines = cv2.HoughLinesP(morph, 
                        rho=1, 
                        theta=np.pi/180, 
                        threshold=50, 
                        minLineLength=50, 
                        maxLineGap=10)

# 建立一個複製影像用來繪製直線，使用彩色方便觀察
line_img = img.copy()

if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 繪製直線，顏色設定為紅色，粗細 3 像素
        cv2.line(line_img, (x1, y1), (x2, y2), (0, 0, 255), 3)
else:
    print("未偵測到直線")

cv2.imwrite('line_1.jpg', line_img)

# 如果有需要也可顯示最終結果
cv2.namedWindow('Detected Lines', cv2.WINDOW_NORMAL)
cv2.imshow('Detected Lines', line_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
