# openCV 컨투어 검출
# 이지혁
# 최초 코드 작성일 : 2025.06.19
# 수정 코드 작성일 : 2025.06.20
# other_thresholds 기능 추가 : 2025.06.19
import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def main(image_path, mode="main"):
    print("=== main() 시작 ===")
    mode = mode
    print("\n1단계 : 이미지 읽기")
    try:
        # 원본 이미지 읽기
        image_origin = cv2.imread(image_path)
        
        # 없다면 에러처리
        if image_origin is None:
            raise FileNotFoundError(f'이미지를 찾을 수 없습니다 -> 파일경로 : {image_path}')

        # 그레이 스케일 이미지 생성
        image_gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"이미지 파일을 성공적으로 읽었습니다 -> 파일경로 : {image_path}")
        print(f"원본 이미지 크기 -> 세로: {image_origin.shape[0]} 가로: {image_origin.shape[1]}")
        print(f"그레이스케일 이미지 크기 -> 세로: {image_gray.shape[0]} 가로: {image_gray.shape[1]}")
        
    except FileNotFoundError as e:
        print(f"error : {e}")
        return
    
    print("\n2단계 : 이진 이미지 생성 시작")
    image_binary = set_image_binary(image_gray) # 이진 이미지 생성 함수 호출
    if image_binary is None:
        print("이진 이미지 생성에 실패했습니다.\n함수를 종료합니다.")
        return
    print(f"이진 이미지 생성 완료\nThreshold 값 -> 100")
    print(f"이진 이미지 크기 -> 세로: {image_binary.shape[0]} 가로: {image_binary.shape[1]}")
    
    print("\n3단계 : 컨투어 검출 시작")
    

    result_contour = detect_contours(image_binary)
    if result_contour is None:
        print("컨투어가 검출되지 않았습니다.")
    else:
        print(f"컨투어 검출 완료, 총 {len(result_contour)}개의 컨투어가 검출되었습니다.")
    
    print("\n4단계 : 결과 이미지 출력")
    image_contour = cv2.cvtColor(image_origin.copy(), cv2.COLOR_BGR2RGB)
    
    if (mode == "shape"):
        print("\n추가 기능2: Contours 모양에 따른 분류 시작")
        classify_shape_contours(image_contour ,result_contour)
    else:
        # 컨투어 green으로 그림
        cv2.drawContours(image_contour, result_contour, -1, (0, 255, 0), 2)
    display_result(mode, image_origin, image_gray, image_binary, image_contour)

    
    print("\n5단계 : 결과 이미지 저장 시작")
    if (save_results(mode, image_origin, image_gray, image_binary, image_contour) == None):
        print("결과 이미지 저장에 실패했습니다.")
    else:
        print("결과 이미지 저장 완료")
    print("=== main() 종료 ===")

def set_image_binary(image_gray): # 2단계; 이진 이미지 생성 함수
    
    # Threshold을 이용한 이진 이미지, Threshold 값 100
    # 여기서 흑백 반전을 하는 이유 = 검은 배경의 흰점 처리가 더 유리하다
    # openCV 의 findContours() 함수는 흰색 픽셀(255)를 컨투어(윤곽선)으로 인식하고 검은픽셀(0)을 배경으로 간주
    
    #region np, cv2 두 가지 방법으로 이진 이미지 생성(장,단점 비교) 택 1
    # np.where
    ### 장점 : 조건의 다양화(ex gray< 80, gray > 200 등) 가능, ### 단점 : 파이썬 루프 기반(상대적 느림)
    # cv2.threshold
    ### 장점 : C++ 기반(상대적 빠름), ### 단점 : 조건의 다양화 불가(단순 임계값만 가능)
    
    # cv2.threshold를 이용한다면 Thresh_BINARY_INV 옵션을 사용하여 흰색 픽셀을 0(검정)으로
    # 검은색 픽셀을 255(흰색)로 반전해주는 것이 countours 검출에 유리하다
    #endregion
    
    image_binary = np.where(image_gray < 100, 255, 0).astype(np.uint8)
    # _, image_binary = cv2.threshold(image_gray, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 이진 이미지 검증 0과 255가 아닌 값이 있을 경우 NONE
    if not np.all(np.isin(image_binary, [0, 255])):
        print("이진 이미지 생성 실패: 0과 255 이외의 값이 존재합니다.")
        return None
    
    return image_binary

def detect_contours(image_binary): # 3단계; 컨투어 검출 함수
    #region 컨투어 검출
    # cv2.RETR_EXTERNAL: 가장 외곽 컨투어(255)만 검출
    # cv2.CHAIN_APPROX_SIMPLE: 컨투어 포인트를 압축하여 저장
    #endregion
    contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"검출된 컨투어 개수: {len(contours)}")
    
    if len(contours) == 0:
        return None
    else:
        for i, contour in enumerate(contours):
            # 컨투어의 면적 계산
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            print(f"컨투어 {i+1}: 면적 = {area:.2f}, 둘레 = {perimeter:.2f}")
        return contours

def display_result(mode, image_origin, image_gray, image_binary, image_contour): # 4단계 결과 이미지 출력 함수
    
    print("결과 이미지 생성중")
    plt.figure(figsize=(12, 10))
    plt.suptitle("OpenCV Contours Detection Result", fontsize=16)
    plt.subplot(2, 2, 1)
    plt.title("1. Original Image")
    plt.imshow(cv2.cvtColor(image_origin, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(2, 2, 2)
    plt.title("2. Grayscale Image")
    plt.imshow(image_gray, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 3)
    plt.title("3. Binary Image; Threshold = 100")
    plt.imshow(image_binary, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 2, 4)
    plt.title("4. Contour Detection Result")
    plt.imshow(image_contour)
    plt.axis('off')
    
    if(mode == "shape"):
        # 범례 추가
        legend_shapes = ["Rectangle", "Scratch", "Circle", "Noise", "Other"]
        legend_colors = [np.array(get_shape_color(shape))/255.0 for shape in legend_shapes]
        
        # 범례를 위한 더미 플롯
        for i, (shape, color) in enumerate(zip(legend_shapes, legend_colors)):
            plt.plot([], [], 'o', color=color, markersize=8, label=shape)
        
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    print("결과 이미지 생성 완료")
    print("plt 시작")
    plt.tight_layout()
    plt.show()
    print("plt 종료")

def save_results(mode, image_origin, image_gray, image_binary, image_contour): # 5단계 결과 이미지 저장 함수
    
    print("결과 이미지 저장중")
    folder_path = "./result_img"
    
    if(mode == "shape"):
        images = {
            "image_origin": image_origin,
            "image_gray": image_gray,
            "image_binary": image_binary,
            "image_contour_label": image_contour
        }
    else:
        images = {
            "image_origin": image_origin,
            "image_gray": image_gray,
            "image_binary": image_binary,
            "image_contour": image_contour
        }
    try:
        for key,value in images.items():
            cv2.imwrite(f"{folder_path}/{key}.png", value)
            print(f"{key}.png 저장 완료")
    except Exception as e:
        print(f"결과 이미지 저장 실패 {e}")
        return None
    return 1
    
def other_thresholds(image_path): #다른 임계값에 따른 이미지 확인 함수
    
    image_gray = cv2.imread(image_path, 0)
    print(f"받은 경로 : {image_path}")
    print(f"이미지 크기 : 세로 {image_gray.shape[0]}, 가로 {image_gray.shape[1]}")
    if image_gray is None:
        print("이미지를 불러올 수 없습니다.")
        return
    #임계값 50 150 200 확인
    thresholds = [50, ]
    image_binarys = []
    count_contours = []
    
    for thresh in thresholds:
        image_binary = np.where(image_gray < thresh, 255, 0).astype(np.uint8)
        image_binarys.append(image_binary)
        contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for i, contour in enumerate(contours):
            # 컨투어의 면적 계산
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            print(f"컨투어 {i+1}: 면적 = {area:.2f}, 둘레 = {perimeter:.2f}")
        count_contours.append(len(contours))
    
    plt.figure(figsize=(10, 10))
    
    for i,image_binary in enumerate(image_binarys):
        plt.subplot(2, 2, i + 1)
        plt.title(f"{i+1}. Threshold = {thresholds[i]}, Contour_Count : {count_contours[i]}")
        plt.imshow(image_binary, cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.show()

def classify_shape_contours(image_contour, contours): # 추가 기능2 :Contours 모양에 따른 분류하여 그리기
    shape_counts = {} #모양별 개수
    
    for i, contour in enumerate(contours):
        shape_name = classify_shape(contour) # 모양 분류
        if shape_name in shape_counts: # 모양별 개수 카운트
            shape_counts[shape_name] += 1
        else:
            shape_counts[shape_name] = 1
        
        color = get_shape_color(shape_name) # 컨투어 색깔별로 그리기
        cv2.drawContours(image_contour, [contour], -1, color, 2)
        print(f"컨투어 {i+1}: {shape_name}")
    
    # 검출된 모양 통계 출력
    print(f"\n=== 모양 분류 결과 ===")
    for shape, count in shape_counts.items():
        print(f"{shape}: {count}개")
    
    return image_contour
    
def classify_shape(contour): #추가 기능2 : 조건별 모양 분류 함수
    # 컨투어 근사화 및 면적,둘레
    epsilon = 0.02 * cv2.arcLength(contour, True) #윤곽선 둘레 단순화
    approx = cv2.approxPolyDP(contour, epsilon, True) #꼭지점을 위한 윤곽선 단순화
    area = cv2.contourArea(contour) #픽셀 면적
    perimeter = cv2.arcLength(contour, True) # 픽셀 둘레

    # 최소 면적 조건 (너무 작은 객체 제외)
    if area < 20: return "Noise"

    circularity = 4 * math.pi * area / (perimeter * perimeter) if perimeter > 0 else 0  # 원형도 계산 (4*π*면적/둘레²)
    area_to_perimeter_ratio = area / (perimeter * perimeter) if perimeter > 0 else 0 #선형도 판단
    # 바운딩 박스 정보
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(float(w) / h, float(h) / w) if min(w, h) != 0 else 1
    bbox_area = w * h     # 바운딩 박스 대비 실제 면적 비율 (얼마나 사각형에 가까운지)
    solidity = area / bbox_area if bbox_area > 0 else 0
    vertices = len(approx)    # 꼭짓점 개수
    
    # 스크래치 판별 조건들
    is_scratch = (
        aspect_ratio > 4.0 or  # 종횡비가 4:1 이상
        area_to_perimeter_ratio < 0.02 or  # 면적 대비 둘레가 매우 큰 경우 (얇은 형태)
        (aspect_ratio > 2.5 and solidity < 0.5)  # 종횡비가 높으면서 바운딩박스를 잘 채우지 못하는 경우
    )
    
    # Rectangle 판별 조건들
    is_rectangle = (
        vertices >= 4 and  # 4개 이상의 꼭짓점
        aspect_ratio <= 3.0 and  # 너무 길지 않은 형태
        solidity > 0.6 and  # 바운딩 박스를 잘 채우는 형태
        circularity < 0.6  # 원형이 아닌 형태
    )
    
    # Circle 판별 조건
    is_circle = (
        circularity > 0.7 and  # 높은 원형도
        aspect_ratio < 1.5 and  # 비교적 정사각형에 가까운 바운딩박스
        solidity > 0.7  # 바운딩박스를 잘 채우는 형태
    )
    
    # 분류 로직 (우선순위 순서)
    if is_scratch:
        return "Scratch"
    elif is_circle:
        return "Circle"
    elif is_rectangle:
        return "Rectangle"
    else:
        return "Other"  # 불규칙한 형태 또는 기타 결함

def get_shape_color(shape_name): # 추가 기능2: 모양에 따른 색상 반환 함수
    color_map = {
        "Rectangle": (0, 0, 255),     # 파랑
        "Scratch": (255, 0, 0),     # 빨강색 (스크래치)
        "Circle": (0, 255, 255),      # 청록
        "Noise": (255, 165, 0),     # 주황
        "Other": (0, 255, 0)       # 초록
    }
    return color_map.get(shape_name, (0, 0, 0)) # 없으면 검정 반환

if __name__ == "__main__":
    # image_path = "./RAW_02-66_0309.bm"  # 비정상 경로 테스트
    image_path = "RAW_02-66_0309.bmp"  # 이미지 파일 경로
    
    # 메인
    # 추가 기능 2를 보시려면 "main" 대신 "shape"로 변경하세요
    main(image_path, "main")
    
    # 추가 기능1
    # 왜 임계값이 100일까? 라는 질문에서 0 ~ 255다른 임계값과의 차이는??
    # other_thresholds(image_path)
    