# opencv

1. 읽기

```python
cv2.imread("path")
# numpy 배열로 로드하는 것이라 numpy가 필요하다?
```

2. 쓰기

```python
cv2.imwrite("filename", source)
```

3. 이미지 크기조절(resize)

```python
cv2.resize(img,(width,height))
```

4. 이미지 자르기(crop)

```python
img[50:200, 100:300]
```

### 중요

5. 색공간 변화 (RGB -> GRAY, HSV)

```python
cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
```

6. 블러링(가우시안, 평균, 미디언)

```python
blur = cv2.GaussianBlur(img, (5, 5), 0)
median = cv2.medianBlur(img, 5)
```

7. 엣지 검출(canny)

```python
edges = cv2.Canny(img, 100, 200)
```

8. 이진화(Thresholding)

```python
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
```

### 중요

9. 윤곽선 찾기 (Contour Detection)

```python
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img_contour = cv2.drawContours(img.copy(), contours, -1, (0, 255, 0), 2)
```
