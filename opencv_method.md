# opencv

1. 읽기

   ```python
   cv2.imread("path", "flag(optional)")

   path는 이미지 파일 경로를 의미하며,
   flags는 선택적으로 넣어주며,
   + cv2.IMREAD_COLOR = 1 : BGR 순서의 컬러(기본값)
   + cv2.IMREAD_GRAYSCALE = 0 : Grayscale로 읽음
   + cv2.IMREAD_UNCHANGED = -1 : 알파 채널/투명도를 포함하여 읽음

   성공시 Numpy.ndarray(이미지 배열)을 반환하며,
   실패하면 None을 반환
   ```

### 중요

5. 색공간 변화 (RGB -> GRAY, HSV)

   ```python
   cv2.cvtColor("src", "code", "dstCn")

   src는 이미지이며,
   code는 변환 코드이며,
   + cv2.COLOR_BGR2RGB     # BGR → RGB
   + cv2.COLOR_RGB2BGR     # RGB → BGR
   + cv2.COLOR_BGR2GRAY    # BGR → 그레이스케일
   + cv2.COLOR_GRAY2BGR    # 그레이스케일 → BGR
   등으로 각각의 원본2변환본으로 작성한다.
   dstCn 은 채널 수이다.
   ```

6. 블러링(가우시안, 평균, 미디언)

   ```python
   cv2.GaussianBlur('src', 'ksize', 'sigmaX', 'sigmaY', 'borderType')

   scr는 입력이미지,
   ksize는 커널 크기로 (width, height)이며, 홀수 여야한다.
   sigmaX,Y 는 표준편차로 각각 지정하지 않으면 0이거나 X와 동일하다.
   borderType은 경계 처리방법을 의미한다.
   #부드러운 블러처리

   cv2.medianBlur('src', 'ksize')

   scr는 입력이미지,
   ksize는 커널 크기로 (width, height)이며, 홀수 여야한다.
   #ksize에 따라 작은 흑백점이나 강한 노이즈 제거에 효과적이다.
   ```

7. 엣지 검출(canny)\_ 자동 임계값 설정

   ```python
   cv2.Canny('image', 'threshold1', 'threshold2', 'apertureSize', 'L2gradient')

   image: 입력 이미지 (그레이스케일)
   threshold1: 첫 번째 임계값 (낮은 임계값)
   threshold2: 두 번째 임계값 (높은 임계값)
   apertureSize: Sobel 커널 크기 (기본값: 3)
   L2gradient: 그래디언트 크기 계산 방법
   ```

8. 이진화(Thresholding)

   ```python
   cv2.threshold('src', 'thresh', 'maxval', 'type')

   src: 입력 이미지 (그레이스케일)
   thresh: 임계값
   maxval: 임계값 초과시 할당값
   type: 이진화 타입
   + cv2.THRESH_BINARY     # 임계값 초과: maxval, 이하: 0
   + cv2.THRESH_BINARY_INV # 임계값 초과: 0, 이하: maxval
   + cv2.THRESH_TRUNC # 임계값 초과: thresh, 이하: 원래값
   + cv2.THRESH_TOZERO # 임계값 초과: 원래값, 이하: 0
   + cv2.THRESH_TOZERO_INV # 임계값 초과: 0, 이하: 원래값
   + cv2.THRESH_OTSU # Otsu 자동 임계값
   + cv2.THRESH_TRIANGLE # Triangle 자동 임계값

   이를 제외하고 np.where()로도 가능하다.
   차이와 장단점은 opencv.py line75에서 확인 가능하다.
   ```

9. 윤곽선 찾기 (Contour Detection)

   ```python
   cv2.findContours('image', 'mode', 'method')

   image: 이진 이미지
   mode: 윤곽선 검색 모드
   + cv2.RETR_EXTERNAL     # 가장 외부 윤곽선만
   + cv2.RETR_LIST         # 모든 윤곽선 (계층 무시)
   + cv2.RETR_CCOMP        # 2레벨 계층
   + cv2.RETR_TREE         # 전체 계층 트리
   + cv2.RETR_FLOODFILL    # 플러드필 마스크용
   method: 윤곽선 근사 방법
   + cv2.CHAIN_APPROX_NONE     # 모든 점 저장
   + cv2.CHAIN_APPROX_SIMPLE   # 압축하여 저장 (권장)
   + cv2.CHAIN_APPROX_TC89_L1  # Teh-Chin 근사?
   + cv2.CHAIN_APPROX_TC89_KCOS # Teh-Chin 근사?
   ```

10. 윤곽선 그리기

    ```python
    cv2.drawContours(image, contours, contourIdx, color, thickness, lineType, hierarchy, maxLevel, offset)

    image: 그릴 대상 이미지
    contours: 윤곽선 리스트
    contourIdx: 그릴 윤곽선 인덱스 (-1: 모든 윤곽선)
    color: 색상 (BGR)
    thickness: 선 두께 (-1: 채우기)
    lineType: 선 타입
    hierarchy: 계층 정보
    maxLevel: 그릴 최대 레벨
    offset: 윤곽선 이동
    ```
