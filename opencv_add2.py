# openCV 컨투어 검출 + 모양 분류
# 이지혁
# 최초 코드 작성일 : 2025.06.19
# 수정 코드 작성일 : 2025.06.20
# 모양 분류 기능 추가 : 2025.06.20

import numpy as np
import cv2
import matplotlib.pyplot as plt
import math


def main(image_path):
    print("=== main() 시작 ===")
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
        return
    else:
        print(f"컨투어 검출 완료, 총 {len(result_contour)}개의 컨투어가 검출되었습니다.")
    
    print("\n4단계 : 모양 분류 및 라벨링 시작")
    image_contour = cv2.cvtColor(image_origin.copy(), cv2.COLOR_BGR2RGB)
    # 컨투어 그리기 및 모양 분류
    image_labeled = classify_and_label_shapes(image_contour, result_contour)
    
    print("\n5단계 : 결과 이미지 출력")
    display_result(image_origin, image_gray, image_binary, image_labeled)
    
    print("\n6단계 : 결과 이미지 저장 시작")
    if (save_results(image_origin, image_gray, image_binary, image_labeled) == None):
        print("결과 이미지 저장에 실패했습니다.")
    else:
        print("결과 이미지 저장 완료")
    print("=== main() 종료 ===")


def classify_shape(contour):
    """컨투어의 모양을 분류하는 함수"""
    # 컨투어 근사화
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    # 면적과 둘레 계산
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    # 최소 면적 조건 (너무 작은 객체 제외)
    if area < 100:
        return "Noise"
    
    # 꼭짓점 개수로 기본 분류
    vertices = len(approx)
    
    # 원형도 계산 (4π*면적/둘레²)
    if perimeter > 0:
        circularity = 4 * math.pi * area / (perimeter * perimeter)
    else:
        circularity = 0
    
    # 종횡비 계산
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(float(w) / h, float(h) / w) if min(w, h) != 0 else 1
    
    # 선형도 계산 (얇고 긴 형태 판별)
    # 1. 종횡비가 매우 높은 경우 (길이 >> 너비)
    # 2. 면적 대비 둘레가 큰 경우 (얇은 형태)
    elongation_ratio = aspect_ratio  # 이미 max값으로 계산됨
    area_to_perimeter_ratio = area / (perimeter * perimeter) if perimeter > 0 else 0
    
    # 직선성 검사 (fitLine 사용)
    [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    
    # 직선으로부터의 평균 거리 계산
    distances = []
    for point in contour:
        px, py = point[0]
        # 점과 직선 사이의 거리 계산
        # 직선: (x-x0)/vx = (y-y0)/vy
        if vx != 0 and vy != 0:
            # 점에서 직선까지의 수직 거리
            dist = abs((py - y0) * vx - (px - x0) * vy) / math.sqrt(vx*vx + vy*vy)
            distances.append(dist)
    
    avg_distance_from_line = np.mean(distances) if distances else float('inf')
    max_distance_from_line = max(distances) if distances else float('inf')
    
    # 모양 분류 로직
    if vertices == 3:
        return "Triangle"
    elif vertices == 4:
        # 선형/스크래치 판별 조건들
        is_linear = (elongation_ratio > 4.0 or  # 종횡비가 4:1 이상
                    area_to_perimeter_ratio < 0.02 or  # 면적 대비 둘레가 매우 큰 경우
                    avg_distance_from_line < 3.0)  # 직선으로부터 평균 거리가 작은 경우
        
        if is_linear:
            return "Scratch"
        elif 0.85 <= (min(w, h) / max(w, h)) <= 1.0:  # 거의 정사각형
            return "Square"
        else:
            return "Rectangle"
    elif vertices == 5:
        return "Pentagon"
    elif vertices == 6:
        return "Hexagon"
    elif circularity > 0.7:  # 원형도가 높으면 원
        return "Circle"
    elif vertices > 6:
        # 타원과 원 구분
        if circularity > 0.5:
            return "Ellipse"
        else:
            return "Polygon"
    else:
        # 꼭짓점이 2개이거나 매우 적은 경우도 선형으로 분류
        if vertices <= 2 or elongation_ratio > 6.0:
            return "Line"
        return "Unknown"


def get_shape_color(shape_name):
    """모양에 따른 색상 반환"""
    color_map = {
        "Triangle": (255, 0, 0),      # 빨강
        "Square": (0, 255, 0),        # 초록
        "Rectangle": (0, 0, 255),     # 파랑
        "Scratch": (255, 100, 0),     # 주황색 (스크래치)
        "Line": (255, 165, 0),        # 오렌지 (라인)
        "Pentagon": (255, 255, 0),    # 노랑
        "Hexagon": (255, 0, 255),     # 마젠타
        "Circle": (0, 255, 255),      # 시안
        "Ellipse": (128, 0, 128),     # 보라
        "Polygon": (255, 165, 0),     # 주황
        "Noise": (128, 128, 128),     # 회색
        "Unknown": (64, 64, 64)       # 어두운 회색
    }
    return color_map.get(shape_name, (0, 0, 0))


def classify_and_label_shapes(image, contours):
    """모든 컨투어를 분류하고 라벨을 추가하는 함수"""
    shape_counts = {}
    
    for i, contour in enumerate(contours):
        # 모양 분류
        shape_name = classify_shape(contour)
        
        # 모양별 개수 카운트
        if shape_name in shape_counts:
            shape_counts[shape_name] += 1
        else:
            shape_counts[shape_name] = 1
        
        # 컨투어 그리기
        color = get_shape_color(shape_name)
        cv2.drawContours(image, [contour], -1, color, 2)
        
        # 라벨 텍스트 위치 계산 (컨투어의 중심)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            # 중심을 계산할 수 없으면 bounding box 중심 사용
            x, y, w, h = cv2.boundingRect(contour)
            cx = x + w // 2
            cy = y + h // 2
        
        # 텍스트 배경 사각형 그리기 (가독성 향상)
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        thickness = 2
        text = f"{shape_name}"
        
        # 텍스트 크기 계산
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        
        # 배경 사각형 그리기
        cv2.rectangle(image, 
                     (cx - text_width//2 - 5, cy - text_height//2 - 5),
                     (cx + text_width//2 + 5, cy + text_height//2 + 5),
                     (255, 255, 255), -1)
        
        # 텍스트 외곽선 (가독성 향상)
        cv2.putText(image, text, (cx - text_width//2, cy + text_height//2), 
                   font, font_scale, (0, 0, 0), thickness + 1)
        
        # 메인 텍스트
        cv2.putText(image, text, (cx - text_width//2, cy + text_height//2), 
                   font, font_scale, color, thickness)
        
        # 컨투어 정보 출력
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        print(f"컨투어 {i+1}: {shape_name}, 면적 = {area:.2f}, 둘레 = {perimeter:.2f}")
    
    # 검출된 모양 통계 출력
    print(f"\n=== 모양 분류 결과 ===")
    for shape, count in shape_counts.items():
        print(f"{shape}: {count}개")
    
    return image


def set_image_binary(image_gray):
    """2단계; 이진 이미지 생성 함수"""
    image_binary = np.where(image_gray < 100, 255, 0).astype(np.uint8)
    
    # 이진 이미지 검증 0과 255가 아닌 값이 있을 경우 NONE
    if not np.all(np.isin(image_binary, [0, 255])):
        print("이진 이미지 생성 실패: 0과 255 이외의 값이 존재합니다.")
        return None
    
    return image_binary


def detect_contours(image_binary):
    """3단계; 컨투어 검출 함수"""
    contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"검출된 컨투어 개수: {len(contours)}")
    
    if len(contours) == 0:
        return None
    else:
        return contours


def display_result(image_origin, image_gray, image_binary, image_labeled):
    """결과 이미지 출력 함수"""
    print("결과 이미지 생성중")
    plt.figure(figsize=(15, 12))
    plt.suptitle("OpenCV Contours Detection with Shape Classification", fontsize=16)
    
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
    plt.title("4. Shape Classification Result")
    plt.imshow(image_labeled)
    plt.axis('off')
    
    # 범례 추가
    legend_shapes = ["Triangle", "Square", "Rectangle", "Scratch", "Line", "Pentagon", 
                    "Hexagon", "Circle", "Ellipse", "Polygon"]
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


def save_results(image_origin, image_gray, image_binary, image_labeled):
    """결과 이미지 저장 함수"""
    print("결과 이미지 저장중")
    folder_path = "./result_img"
    
    # 폴더가 없으면 생성
    import os
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    images = {
        "image_origin": cv2.cvtColor(image_origin, cv2.COLOR_RGB2BGR),
        "image_gray": image_gray,
        "image_binary": image_binary,
        "image_labeled": cv2.cvtColor(image_labeled, cv2.COLOR_RGB2BGR)
    }
    
    try:
        for key, value in images.items():
            cv2.imwrite(f"{folder_path}/{key}.png", value)
            print(f"{key}.png 저장 완료")
    except Exception as e:
        print(f"결과 이미지 저장 실패 {e}")
        return None
    print("결과 이미지 저장 완료")


def other_thresholds(image_path):
    """다른 임계값에 따른 이미지 확인 함수"""
    image_gray = cv2.imread(image_path, 0)
    print(f"받은 경로 : {image_path}")
    print(f"이미지 크기 : 세로 {image_gray.shape[0]}, 가로 {image_gray.shape[1]}")
    
    if image_gray is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    # 임계값 50, 100, 150, 200 확인
    thresholds = [50, 100, 150, 200]
    image_binarys = []
    count_contours = []
    shape_results = []
    
    for thresh in thresholds:
        image_binary = np.where(image_gray < thresh, 255, 0).astype(np.uint8)
        image_binarys.append(image_binary)
        
        contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        count_contours.append(len(contours))
        
        # 각 임계값에서의 모양 분류 결과
        shapes = {}
        for contour in contours:
            shape = classify_shape(contour)
            shapes[shape] = shapes.get(shape, 0) + 1
        shape_results.append(shapes)
        
        print(f"\n임계값 {thresh}에서의 결과:")
        print(f"  총 컨투어 개수: {len(contours)}")
        for shape, count in shapes.items():
            print(f"  {shape}: {count}개")
    
    plt.figure(figsize=(15, 10))
    
    for i, image_binary in enumerate(image_binarys):
        plt.subplot(2, 2, i + 1)
        title = f"Threshold = {thresholds[i]}\nContours: {count_contours[i]}"
        if shape_results[i]:
            shape_info = ", ".join([f"{k}:{v}" for k, v in shape_results[i].items()])
            title += f"\n{shape_info}"
        plt.title(title)
        plt.imshow(image_binary, cmap='gray')
        plt.axis('off')
    
    plt.suptitle("Different Thresholds Comparison with Shape Classification", fontsize=14)
    plt.tight_layout()
    plt.show()


def analyze_shape_details(image_path):
    """상세한 모양 분석 함수 (디버깅 용)"""
    image_gray = cv2.imread(image_path, 0)
    if image_gray is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    image_binary = np.where(image_gray < 100, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(image_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print("=== 상세 모양 분석 결과 ===")
    for i, contour in enumerate(contours):
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        if perimeter > 0:
            circularity = 4 * math.pi * area / (perimeter * perimeter)
        else:
            circularity = 0
        
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = float(w) / h if h != 0 else 0
        
        shape = classify_shape(contour)
        
        print(f"\n컨투어 {i+1}:")
        print(f"  분류된 모양: {shape}")
        print(f"  꼭짓점 개수: {len(approx)}")
        print(f"  면적: {area:.2f}")
        print(f"  둘레: {perimeter:.2f}")
        print(f"  원형도: {circularity:.3f}")
        print(f"  종횡비: {aspect_ratio:.3f}")
        print(f"  길이 방향 비율: {max(w, h) / min(w, h) if min(w, h) != 0 else 0:.3f}")
        
        # 선형 특성 분석
        if shape in ["Scratch", "Line"]:
            [vx, vy, x0, y0] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
            distances = []
            for point in contour:
                px, py = point[0]
                if vx != 0 and vy != 0:
                    dist = abs((py - y0) * vx - (px - x0) * vy) / math.sqrt(vx*vx + vy*vy)
                    distances.append(dist)
            if distances:
                print(f"  직선으로부터 평균 거리: {np.mean(distances):.2f}")
                print(f"  직선으로부터 최대 거리: {max(distances):.2f}")


if __name__ == "__main__":
    image_path = "RAW_02-66_0309.bmp"  # 이미지 파일 경로
    
    # 메인 함수 실행
    main(image_path)
    
    # 추가 분석 함수들 (필요에 따라 주석 해제)
    # other_thresholds(image_path)
    # analyze_shape_details(image_path)