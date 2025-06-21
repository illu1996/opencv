# OpenCV 를 활용한 Contour 검출

## 개요

```python
본 프로젝트는 OpenCV 라이브러리를 활용하여 이미지 처리 및 윤곽선 검출을 수행하는 프로젝트입니다.

진행 순서는 아래와 같습니다.
1. Grayscale 이미지 생성
2. Binary 이미지 생성
3. Contour 검출
4. 시각화 및 저장

이를 통해 결과를 확인하며, 이미지 처리의 기본 프로세스를 파악할 수 있습니다.
```

### 개발 환경

```bash
1.	Python 3.13.5
2.	opencv-python 4.11.0.86
```

### 실행 절차

1. 프로젝트 폴더 내부 파일들이 존재하는지 확인

   ```bash
   opencv/opencv.py
   opencv/requirements.txt
   ```

2. 사용할 이미지 프로젝트 폴더에 존재하는지 확인
   ```bash
   ex_image.png
   ```
3. 파이썬 가상환경 설정하기 ( 본 단계는 사용자의 컴퓨터 환경에 따라 달라집니다. )
   ```bash
   >>bash
   python -m venv venv
   ```
4. 필요 모듈 설치하기
   ```bash
   >>bash
   pip install -r requirements.txt
   ```
5. 파이썬 인터프리터 설정 확인하기
6. 프로젝트 실행하기
   ```bash
   /opencv
   python opencv.py
   ```

## 조건별 실행 절차

1. main 프로젝트 실행

   ```python
   /opencv/opencv.py
   line 368

   if __name__ == "__main__":
      main(image_path, "main") # main 실행
   ```

2. 추가 기능 2 : Contour의 모양 분류 기능 실행

   ```python
   /opencv/opencv.py
   line 368

   if __name__ == "__main__":
      main(image_path, "shape") # 추가 기능 2 실행
   ```

3. 추가 기능 1 : 각 임계점 별 Contour 개수 확인 기능 실행

   ```python
   /opencv/opencv.py
   line 368

   if __name__ == "__main__":
      other_thresholds(image_path) # 추가 기능 1 실행
   ```

4. 추가 기능 3 : 바이너리 이미지 전 후의 가우시안 블러 처리 기능 실행

   ```python
   /opencv/opencv.py
   line 368

   if __name__ == "__main__":
      gaussian_blur(image_path) # 추가 기능 3 실행
   ```
