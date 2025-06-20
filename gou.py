# OpenCV 컨투어 검출 실습
# 작성자: [학번] [이름]
# 작성일: 2025년 6월

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def main():
    """
    메인 함수: 이미지 처리 파이프라인 실행
    """
    print("=== OpenCV 컨투어 검출 실습 ===")
    
    # 2.2 이미지 읽기
    print("2.2 이미지 읽기 단계")
    image_path = "RAW_02-66_0309.jpg"  # 이미지 파일 경로
    
    try:
        # RGB 컬러 이미지 읽기
        img_rgb = cv2.imread(image_path)
        if img_rgb is None:
            raise FileNotFoundError(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        
        # 그레이스케일 이미지 읽기
        img_gray = cv2.imread(image_path, 0)
        
        print(f"✓ 이미지 로드 완료")
        print(f"  - 컬러 이미지 크기: {img_rgb.shape}")
        print(f"  - 그레이스케일 이미지 크기: {img_gray.shape}")
        
    except FileNotFoundError as e:
        print(f"❌ 오류: {e}")
        print("RAW_02-66_0309.jpg 파일을 프로젝트 폴더에 저장해주세요.")
        return
    
    # 2.3 이진 이미지 생성
    print("\n2.3 이진 이미지 생성 단계")
    
    # 임계값 100을 기준으로 이진화
    # 픽셀값 < 100 → 255(흰색), 픽셀값 ≥ 100 → 0(검정색)
    img_bin = np.where(img_gray < 100, 255, 0).astype(np.uint8)
    
    print(f"✓ 이진화 완료")
    print(f"  - 임계값: 100")
    print(f"  - 이진 이미지 크기: {img_bin.shape}")
    print(f"  - 흰색 픽셀 개수: {np.sum(img_bin == 255)}")
    print(f"  - 검정색 픽셀 개수: {np.sum(img_bin == 0)}")
    
    # 2.4 컨투어 검출
    print("\n2.4 컨투어 검출 단계")
    
    # 컨투어 검출
    # cv2.RETR_EXTERNAL: 가장 외곽 컨투어만 검출
    # cv2.CHAIN_APPROX_SIMPLE: 컨투어 포인트를 압축하여 저장
    contours, hierarchy = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    print(f"✓ 컨투어 검출 완료")
    print(f"  - 검출된 컨투어 개수: {len(contours)}")
    
    # 컨투어 정보 출력
    if contours:
        for i, contour in enumerate(contours[:5]):  # 상위 5개만 출력
            area = cv2.contourArea(contour)
            perimeter = cv2.arcLength(contour, True)
            print(f"  - 컨투어 {i+1}: 면적={area:.2f}, 둘레={perimeter:.2f}")
    
    # 2.5 컨투어 그리기
    print("\n2.5 컨투어 그리기 단계")
    
    # 원본 이미지 복사 (BGR → RGB 변환)
    img_result = cv2.cvtColor(img_rgb.copy(), cv2.COLOR_BGR2RGB)
    
    # 모든 컨투어를 초록색으로 그리기
    # contours: 컨투어 리스트
    # -1: 모든 컨투어 그리기 (특정 인덱스 지정 가능)
    # (0, 255, 0): 초록색 (RGB)
    # 2: 선 두께
    cv2.drawContours(img_result, contours, -1, (0, 255, 0), 2)
    
    print(f"✓ 컨투어 그리기 완료")
    print(f"  - 그려진 컨투어 개수: {len(contours)}")
    
    # 결과 시각화
    display_results(img_rgb, img_gray, img_bin, img_result, contours)
    
    # 결과 이미지 저장
    save_results(img_rgb, img_gray, img_bin, img_result)
    
    print("\n=== 실습 완료 ===")

def display_results(img_rgb, img_gray, img_bin, img_result, contours):
    """
    결과 이미지들을 시각화하는 함수
    
    Args:
        img_rgb: 원본 컬러 이미지 (BGR)
        img_gray: 그레이스케일 이미지
        img_bin: 이진 이미지
        img_result: 컨투어가 그려진 결과 이미지 (RGB)
        contours: 검출된 컨투어 리스트
    """
    # 2x2 서브플롯 생성
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('OpenCV 컨투어 검출 결과', fontsize=16, fontweight='bold')
    
    # 1. 원본 이미지 (BGR → RGB 변환)
    axes[0, 0].imshow(cv2.cvtColor(img_rgb, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title('1. 원본 이미지 (컬러)')
    axes[0, 0].axis('off')
    
    # 2. 그레이스케일 이미지
    axes[0, 1].imshow(img_gray, cmap='gray')
    axes[0, 1].set_title('2. 그레이스케일 이미지')
    axes[0, 1].axis('off')
    
    # 3. 이진 이미지
    axes[1, 0].imshow(img_bin, cmap='gray')
    axes[1, 0].set_title('3. 이진 이미지 (임계값: 100)')
    axes[1, 0].axis('off')
    
    # 4. 컨투어 검출 결과
    axes[1, 1].imshow(img_result)
    axes[1, 1].set_title(f'4. 컨투어 검출 결과 ({len(contours)}개)')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 컨투어 분석 결과 출력
    print_contour_analysis(contours)

def print_contour_analysis(contours):
    """
    컨투어 분석 결과를 출력하는 함수
    
    Args:
        contours: 검출된 컨투어 리스트
    """
    print("\n--- 컨투어 분석 결과 ---")
    if not contours:
        print("검출된 컨투어가 없습니다.")
        return
    
    # 면적 기준으로 정렬 (큰 순서대로)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    
    print(f"총 {len(contours)}개의 컨투어가 검출되었습니다.")
    print("\n상위 10개 컨투어 정보:")
    print("순위\t면적\t\t둘레\t\t포인트 수")
    print("-" * 50)
    
    for i, contour in enumerate(sorted_contours[:10]):
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        points = len(contour)
        print(f"{i+1}\t{area:.2f}\t\t{perimeter:.2f}\t\t{points}")
    
    # 통계 정보
    areas = [cv2.contourArea(c) for c in contours]
    print(f"\n면적 통계:")
    print(f"  - 최대 면적: {max(areas):.2f}")
    print(f"  - 최소 면적: {min(areas):.2f}")
    print(f"  - 평균 면적: {np.mean(areas):.2f}")
    print(f"  - 표준편차: {np.std(areas):.2f}")

def save_results(img_rgb, img_gray, img_bin, img_result):
    """
    결과 이미지들을 파일로 저장하는 함수
    
    Args:
        img_rgb: 원본 컬러 이미지 (BGR)
        img_gray: 그레이스케일 이미지
        img_bin: 이진 이미지
        img_result: 컨투어가 그려진 결과 이미지 (RGB)
    """
    try:
        # 그레이스케일 이미지 저장
        cv2.imwrite('result_grayscale.jpg', img_gray)
        
        # 이진 이미지 저장
        cv2.imwrite('result_binary.jpg', img_bin)
        
        # 컨투어 결과 이미지 저장 (RGB → BGR 변환)
        result_bgr = cv2.cvtColor(img_result, cv2.COLOR_RGB2BGR)
        cv2.imwrite('result_contours.jpg', result_bgr)
        
        print("\n✓ 결과 이미지 저장 완료:")
        print("  - result_grayscale.jpg")
        print("  - result_binary.jpg") 
        print("  - result_contours.jpg")
        
    except Exception as e:
        print(f"❌ 이미지 저장 중 오류 발생: {e}")

# 가우시안 분포 기반 이미지 처리
def gaussian_based_processing(image_path):
    """
    가우시안 분포를 적용한 이미지 처리 및 분석
    
    Args:
        image_path: 이미지 파일 경로
    """
    img_gray = cv2.imread(image_path, 0)
    if img_gray is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    print("\n=== 가우시안 분포 기반 이미지 처리 ===")
    
    # 1. 히스토그램 분석 및 가우시안 분포 피팅
    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    
    # 가우시안 분포 파라미터 추정
    mean_intensity = np.mean(img_gray)
    std_intensity = np.std(img_gray)
    
    print(f"이미지 픽셀 강도 통계:")
    print(f"  - 평균: {mean_intensity:.2f}")
    print(f"  - 표준편차: {std_intensity:.2f}")
    print(f"  - 최소값: {np.min(img_gray)}")
    print(f"  - 최대값: {np.max(img_gray)}")
    
    # 2. 가우시안 블러 적용 (다양한 커널 크기)
    gaussian_results = []
    kernel_sizes = [(3, 3), (5, 5), (9, 9), (15, 15)]
    
    for kernel_size in kernel_sizes:
        # 가우시안 블러 적용
        img_gaussian = cv2.GaussianBlur(img_gray, kernel_size, 0)
        
        # 이진화
        img_bin = np.where(img_gaussian < 100, 255, 0).astype(np.uint8)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        gaussian_results.append({
            'kernel_size': kernel_size,
            'processed_img': img_gaussian,
            'binary_img': img_bin,
            'contours': contours,
            'contour_count': len(contours)
        })
        
        print(f"가우시안 블러 ({kernel_size}): {len(contours)}개 컨투어 검출")
    
    # 3. 가우시안 분포 기반 적응적 임계값
    # 평균 ± k*표준편차를 임계값으로 사용
    adaptive_thresholds = []
    k_values = [0.5, 1.0, 1.5, 2.0]
    
    print(f"\n가우시안 분포 기반 적응적 임계값:")
    for k in k_values:
        thresh_lower = max(0, mean_intensity - k * std_intensity)
        thresh_upper = min(255, mean_intensity + k * std_intensity)
        
        # 이진화 (평균 ± k*σ 범위를 객체로 간주)
        img_adaptive = np.where((img_gray >= thresh_lower) & (img_gray <= thresh_upper), 255, 0).astype(np.uint8)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(img_adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        adaptive_thresholds.append({
            'k': k,
            'thresh_lower': thresh_lower,
            'thresh_upper': thresh_upper,
            'binary_img': img_adaptive,
            'contours': contours,
            'contour_count': len(contours)
        })
        
        print(f"  k={k}: 임계값 범위 [{thresh_lower:.1f}, {thresh_upper:.1f}], 컨투어: {len(contours)}개")
    
    # 4. 가우시안 혼합 모델 기반 이진화
    img_gmm = gaussian_mixture_segmentation(img_gray)
    contours_gmm, _ = cv2.findContours(img_gmm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"가우시안 혼합 모델: {len(contours_gmm)}개 컨투어 검출")
    
    # 5. 결과 시각화
    visualize_gaussian_results(img_gray, gaussian_results, adaptive_thresholds, img_gmm, 
                              mean_intensity, std_intensity)
    
    return gaussian_results, adaptive_thresholds

def gaussian_mixture_segmentation(img_gray):
    """
    가우시안 혼합 모델을 사용한 이미지 분할
    
    Args:
        img_gray: 그레이스케일 이미지
    
    Returns:
        이진화된 이미지
    """
    from sklearn.mixture import GaussianMixture
    
    # 픽셀 값을 1D 배열로 변환
    pixels = img_gray.flatten().reshape(-1, 1)
    
    # 가우시안 혼합 모델 학습 (2개 클러스터: 배경과 객체)
    gmm = GaussianMixture(n_components=2, random_state=42)
    gmm.fit(pixels)
    
    # 클러스터 예측
    labels = gmm.predict(pixels)
    
    # 결과를 원본 이미지 크기로 변환
    segmented = labels.reshape(img_gray.shape)
    
    # 더 작은 평균을 가진 클러스터를 객체(255)로, 큰 평균을 가진 클러스터를 배경(0)으로 설정
    means = gmm.means_.flatten()
    if means[0] < means[1]:
        # 클러스터 0이 어두운 영역 (객체)
        binary_result = np.where(segmented == 0, 255, 0).astype(np.uint8)
    else:
        # 클러스터 1이 어두운 영역 (객체)
        binary_result = np.where(segmented == 1, 255, 0).astype(np.uint8)
    
    print(f"가우시안 혼합 모델 파라미터:")
    print(f"  - 클러스터 0 평균: {means[0]:.2f}")
    print(f"  - 클러스터 1 평균: {means[1]:.2f}")
    
    return binary_result

def visualize_gaussian_results(img_gray, gaussian_results, adaptive_thresholds, img_gmm, 
                              mean_intensity, std_intensity):
    """
    가우시안 분포 기반 처리 결과를 시각화
    """
    # 1. 히스토그램과 가우시안 분포 비교
    plt.figure(figsize=(15, 12))
    
    # 히스토그램과 가우시안 분포
    plt.subplot(3, 4, 1)
    hist, bins = np.histogram(img_gray.flatten(), 256, [0, 256])
    plt.plot(hist)
    
    # 이론적 가우시안 분포 오버레이
    x = np.linspace(0, 255, 256)
    gaussian_theoretical = np.exp(-0.5 * ((x - mean_intensity) / std_intensity) ** 2)
    gaussian_theoretical = gaussian_theoretical / np.max(gaussian_theoretical) * np.max(hist)
    plt.plot(x, gaussian_theoretical, 'r--', label=f'가우시안 분포\n(μ={mean_intensity:.1f}, σ={std_intensity:.1f})')
    plt.title('픽셀 강도 히스토그램')
    plt.xlabel('픽셀 강도')
    plt.ylabel('빈도')
    plt.legend()
    
    # 2. 가우시안 블러 결과들
    for i, result in enumerate(gaussian_results):
        plt.subplot(3, 4, i + 2)
        plt.imshow(result['binary_img'], cmap='gray')
        plt.title(f'가우시안 블러 {result["kernel_size"]}\n컨투어: {result["contour_count"]}개')
        plt.axis('off')
    
    # 3. 적응적 임계값 결과들
    for i, result in enumerate(adaptive_thresholds):
        if i < 4:  # 처음 4개만 표시
            plt.subplot(3, 4, i + 6)
            plt.imshow(result['binary_img'], cmap='gray')
            plt.title(f'적응적 k={result["k"]}\n[{result["thresh_lower"]:.1f}, {result["thresh_upper"]:.1f}]')
            plt.axis('off')
    
    # 4. 가우시간 혼합 모델 결과
    plt.subplot(3, 4, 10)
    plt.imshow(img_gmm, cmap='gray')
    plt.title('가우시안 혼합 모델')
    plt.axis('off')
    
    # 5. 원본 이미지
    plt.subplot(3, 4, 11)
    plt.imshow(img_gray, cmap='gray')
    plt.title('원본 이미지')
    plt.axis('off')
    
    # 6. 성능 비교 차트
    plt.subplot(3, 4, 12)
    methods = ['원본'] + [f'Blur{r["kernel_size"]}' for r in gaussian_results] + [f'k={a["k"]}' for a in adaptive_thresholds] + ['GMM']
    counts = [0] + [r['contour_count'] for r in gaussian_results] + [a['contour_count'] for a in adaptive_thresholds] + [len(cv2.findContours(img_gmm, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0])]
    
    plt.bar(range(len(methods)), counts)
    plt.title('컨투어 검출 개수 비교')
    plt.xlabel('방법')
    plt.ylabel('컨투어 개수')
    plt.xticks(range(len(methods)), methods, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.show()

def analyze_gaussian_performance(image_path):
    """
    가우시안 기반 방법들의 성능을 정량적으로 분석
    """
    img_gray = cv2.imread(image_path, 0)
    if img_gray is None:
        return
    
    print("\n=== 가우시안 방법 성능 분석 ===")
    
    # 기준 방법 (원본 임계값 100)
    img_base = np.where(img_gray < 100, 255, 0).astype(np.uint8)
    contours_base, _ = cv2.findContours(img_base, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    methods = {
        '원본 (임계값 100)': img_base,
        '가우시안 블러 (5x5)': None,
        '적응적 임계값 (k=1.0)': None,
        '가우시안 혼합 모델': None
    }
    
    # 가우시안 블러 적용
    img_blur = cv2.GaussianBlur(img_gray, (5, 5), 0)
    img_blur_bin = np.where(img_blur < 100, 255, 0).astype(np.uint8)
    methods['가우시안 블러 (5x5)'] = img_blur_bin
    
    # 적응적 임계값
    mean_val, std_val = np.mean(img_gray), np.std(img_gray)
    thresh_lower = max(0, mean_val - 1.0 * std_val)
    thresh_upper = min(255, mean_val + 1.0 * std_val)
    img_adaptive = np.where((img_gray >= thresh_lower) & (img_gray <= thresh_upper), 255, 0).astype(np.uint8)
    methods['적응적 임계값 (k=1.0)'] = img_adaptive
    
    # 가우시안 혼합 모델
    img_gmm = gaussian_mixture_segmentation(img_gray)
    methods['가우시안 혼합 모델'] = img_gmm
    
    # 성능 지표 계산
    print(f"{'방법':<20} {'컨투어 수':<10} {'평균 면적':<12} {'표준편차':<12} {'최대 면적':<12}")
    print("-" * 70)
    
    for method_name, binary_img in methods.items():
        contours, _ = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            areas = [cv2.contourArea(cnt) for cnt in contours]
            mean_area = np.mean(areas)
            std_area = np.std(areas)
            max_area = np.max(areas)
        else:
            mean_area = std_area = max_area = 0
        
        print(f"{method_name:<20} {len(contours):<10} {mean_area:<12.1f} {std_area:<12.1f} {max_area:<12.1f}")

# 추가 기능: 다양한 임계값으로 실험
def experiment_with_thresholds(image_path):
    """
    다양한 임계값으로 이진화 결과를 비교하는 함수
    
    Args:
        image_path: 이미지 파일 경로
    """
    img_gray = cv2.imread(image_path, 0)
    if img_gray is None:
        print("이미지를 불러올 수 없습니다.")
        return
    
    # 다양한 임계값 테스트
    thresholds = [50, 100, 150, 200]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('다양한 임계값을 사용한 이진화 비교', fontsize=14)
    
    for i, thresh in enumerate(thresholds):
        row, col = i // 2, i % 2
        
        # 이진화
        img_bin = np.where(img_gray < thresh, 255, 0).astype(np.uint8)
        
        # 컨투어 검출
        contours, _ = cv2.findContours(img_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # 결과 표시
        axes[row, col].imshow(img_bin, cmap='gray')
        axes[row, col].set_title(f'임계값: {thresh} (컨투어: {len(contours)}개)')
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 메인 프로그램 실행
    # main()
    
    # 가우시안 분포 기반 처리 실험
    print("\n" + "="*50)
    gaussian_results, adaptive_results = gaussian_based_processing("RAW_02-66_0309.bmp")
    
    # 성능 분석
    analyze_gaussian_performance("RAW_02-66_0309.bmp")
    
    # 추가 실험 (선택사항)
    # experiment_with_thresholds("RAW_02-66_0309.jpg")