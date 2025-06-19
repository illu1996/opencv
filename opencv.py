# openCV 컨투어 검출
# 이지혁
# 최초 코드 작성일 : 2025.06.19
# 수정 코드 작성일 : 2025.06.19
import numpy as np
import cv2
import matplotlib.pyplot as plt


def main():
    print("\n1단계 : 이미지 읽기")
    # image_path = "./RAW_02-66_0309.bm"  # 비정상 경로 테스트
    image_path = "./RAW_02-66_0309.bmp"  # 이미지 파일 경로
    
    try:
        # 원본 이미지 읽기
        origin_image = cv2.imread(image_path)
        
        # 없다면 에러처리
        if origin_image is None:
            raise FileNotFoundError(f'이미지를 찾을 수 없습니다 -> 파일경로 : {image_path}')

        # 그레이 스케일 이미지 생성
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        print(f"이미지 파일을 성공적으로 읽었습니다 -> 파일경로 : {image_path}")
        print(f"원본 이미지 크기 -> 세로: {origin_image.shape[0]} 가로: {origin_image.shape[1]}")
        print(f"그레이스케일 이미지 크기 -> 세로: {gray_image.shape[0]} 가로: {gray_image.shape[1]}")
        
    except FileNotFoundError as e:
        print(f"error : {e}")
        return
    
    print("\n2단계 : 이진 이미지 생성 시작")
    binary_image = set_binary_image(gray_image)
    if binary_image is None:
        print("이진 이미지 생성에 실패했습니다.\n함수를 종료합니다.")
        return
    print(f"이진 이미지 생성 완료\nThresholding 값 -> 100")
    print(f"이진 이미지 크기 -> 세로: {binary_image.shape[0]} 가로: {binary_image.shape[1]}")
    
    print("\n3단계 : 컨투어 검출 시작")
    result_contour = detect_contours(binary_image)
    if result_contour is None:
        print("컨투어가 검출되지 않았습니다.")
    else:
        print(f"컨투어 검출 완료, 총 {len(result_contour)}개의 컨투어가 검출되었습니다.")
    
    print("\n4단계 : 결과 이미지 출력")
    
    
    
    
def set_binary_image(gray_image): # 1단계; 이진 이미지 생성 함수
    
    # Thresholing을 이용한 이진 이미지, Thresholing 값 100
    # 여기서 흑백 반전을 하는 이유 = 검은 배경의 흰점 처리가 더 유리하다
    # openCV 의 findCountours 함수는 흰색 픽셀(255)를 컨투어(윤곽선)으로 인식하고 검은픽셀(0)을 배경으로 간주
    
    #region np, cv2 두 가지 방법으로 이진 이미지 생성(장,단점 비교) 택 1
    # np.where
    # 장점 : 조건의 다양화(ex gray< 80, gray > 200 등) 가능
    # 단점 : 파이썬 루프 기반(상대적 느림)
    # cv2.threshold
    # 장점 : C++ 기반(상대적 빠름)
    # 단점 : 조건의 다양화 불가(단순 임계값만 가능)
    # cv2.threshold를 이용한다면 Thresh_BINARY_INV 옵션을 사용하여 흰색 픽셀을 0(검정)으로
    # 검은색 픽셀을 255(흰색)로 반전해주는 것이 countours 검출에 유리하다
    #endregion
    
    binary_image = np.where(gray_image < 100, 255, 0).astype(np.uint8)
    # _, binary_image = cv2.threshold(gray_image, 100, 255, cv2.THRESH_BINARY_INV)
    
    # 이진 이미지 검증 0과 255가 아닌 값이 있을 경우 NONE
    if not np.all(np.isin(binary_image, [0, 255])):
        print("이진 이미지 생성 실패: 0과 255 이외의 값이 존재합니다.")
        return None
    
    return binary_image

def detect_contours(binary_image): # 2단계; 컨투어 검출 함수
    #region 컨투어 검출
    # cv2.RETR_EXTERNAL: 가장 외곽 컨투어(255)만 검출
    # cv2.CHAIN_APPROX_SIMPLE: 컨투어 포인트를 압축하여 저장
    #endregion
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
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

if __name__ == "__main__":
    # 메인 프로그램 실행
    main()