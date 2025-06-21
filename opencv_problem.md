2.1 Pycharm Program 설치하기

-. Pycharm New Project 생성

-. OpenCV, numpy 등 필요 라이브러리 설치하기

2.2 RAW_02-66_0309 Image 읽기

-. Python OpenCV Library 활용

-. img_rgb = cv2.imread(image_path)

-. img_gray = cv2.imread(image_path, 0)

2.3 Binary Image 생성하기

-. img_bin = cv2.where(img_gray < 100, 255, 0).astype(np.uint8)

2.4 Contour detection based on the img_bin image

2.5 Draw contours on the img_rgb image

전체 과정, 코드 주석 및 결과 이미지 포함 보고서로 작성하기
