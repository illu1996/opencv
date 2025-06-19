# 실행 순서

1. python 3.13.5 install
2. venv 가상환경 만들기
3. pip install numpy
4. pip install opencv-python

python opencv 는 numpy 없이 실행이 어렵다
컴퓨터마다 인터프리터의 문제로 .venv 가상환경을 만드느냐 venv 가상환경을 만드느냐에 따라 모듈을 불러오지 못하는 환경

# wafer 의 불량 검출 프로세스

센서/카메라 촬영 -> 이미지 전처리 -> 특징 추출 -> 불량 검출 -> 분류/리포트

### 이미지 전처리

노이즈 제거, 명암/밝기 정규화, 웨이퍼 영역 마스킹(배경 제거), 정렬(웨이퍼 위치 오차 보정)
