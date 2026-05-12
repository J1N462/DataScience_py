import cv2
import numpy as np

def nothing(x):
    pass

def resize_keep_ratio(img, max_width=1000, max_height=800):
    h, w = img.shape[:2]

    scale_w = max_width / w
    scale_h = max_height / h
    scale = min(scale_w, scale_h, 1.0)  # 원본보다 키우지는 않음

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(img, (new_w, new_h))
    return resized

# 1. 이미지 불러오기
img = cv2.imread('test_image.jpg')

if img is None:
    print("이미지를 불러오지 못했습니다. 'test_image.jpg' 경로를 확인하세요.")
    exit()

# 원본 비율 유지하면서 화면에 맞게 축소
img = resize_keep_ratio(img, max_width=1000, max_height=800)

# 2. 창 생성
cv2.namedWindow('Trackbars', cv2.WINDOW_NORMAL)
cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
cv2.namedWindow('Mask', cv2.WINDOW_NORMAL)
cv2.namedWindow('Edges', cv2.WINDOW_NORMAL)
cv2.namedWindow('Line Detection', cv2.WINDOW_NORMAL)

# HSV 범위 (색상, 채도, 밝기)
cv2.createTrackbar('H_min_1', 'Trackbars', 0, 179, nothing)
cv2.createTrackbar('H_max_1', 'Trackbars', 10, 179, nothing)
cv2.createTrackbar('H_min_2', 'Trackbars', 160, 179, nothing)
cv2.createTrackbar('H_max_2', 'Trackbars', 179, 179, nothing)
cv2.createTrackbar('S_min', 'Trackbars', 100, 255, nothing)
cv2.createTrackbar('S_max', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('V_min', 'Trackbars', 80, 255, nothing)
cv2.createTrackbar('V_max', 'Trackbars', 255, 255, nothing)

# Morphology / Canny / Hough
cv2.createTrackbar('Kernel', 'Trackbars', 5, 15, nothing)
cv2.createTrackbar('Canny1', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('Canny2', 'Trackbars', 150, 255, nothing)
#조명, 노이즈, 선 굵기에 따라 edge 강도가 다르기 때문에 트랙바로 만듦
# threshold가 낮으면 edge 많이 검출 오검출 증가
# threshold가 높으면 edge 깔끔 약한 선 놓칠 수 있음
cv2.createTrackbar('Hough_Th', 'Trackbars', 50, 200, nothing)
cv2.createTrackbar('MinLen', 'Trackbars', 50, 500, nothing)
cv2.createTrackbar('MaxGap', 'Trackbars', 100, 500, nothing)

img_h, img_w = img.shape[:2]
cv2.createTrackbar('ROI_Y', 'Trackbars', img_h // 2, img_h - 1, nothing)

while True:
    frame = img.copy()

    # 3. ROI 설정(아래만 보게하는거임 (하늘 이런거 제낄라고))
    roi_y = cv2.getTrackbarPos('ROI_Y', 'Trackbars')
    roi = frame[roi_y:, :]

    result_img = frame.copy()
    cv2.rectangle(result_img, (0, roi_y), (frame.shape[1], frame.shape[0]), (255, 0, 0), 2)

    # 4. Blur + HSV 변환
    blur = cv2.GaussianBlur(roi, (5, 5), 0) #자갈ㄴㄴ
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)

    # 5. 트랙바 값 읽기
    h_min_1 = cv2.getTrackbarPos('H_min_1', 'Trackbars')
    h_max_1 = cv2.getTrackbarPos('H_max_1', 'Trackbars')
    h_min_2 = cv2.getTrackbarPos('H_min_2', 'Trackbars')
    h_max_2 = cv2.getTrackbarPos('H_max_2', 'Trackbars')
    s_min = cv2.getTrackbarPos('S_min', 'Trackbars')
    s_max = cv2.getTrackbarPos('S_max', 'Trackbars')
    v_min = cv2.getTrackbarPos('V_min', 'Trackbars')
    v_max = cv2.getTrackbarPos('V_max', 'Trackbars')

    kernel_size = cv2.getTrackbarPos('Kernel', 'Trackbars')
    if kernel_size < 1:
        kernel_size = 1
    if kernel_size % 2 == 0:
        kernel_size += 1

    canny1 = cv2.getTrackbarPos('Canny1', 'Trackbars')
    canny2 = cv2.getTrackbarPos('Canny2', 'Trackbars')
    hough_th = cv2.getTrackbarPos('Hough_Th', 'Trackbars')
    min_len = cv2.getTrackbarPos('MinLen', 'Trackbars')
    max_gap = cv2.getTrackbarPos('MaxGap', 'Trackbars')

    # 6. 빨간색 2구간 마스킹
    #빨강을 잡을 때 보통: 구간 1: 0 ~ 10, 구간 2: 160 ~ 179
    lower1 = np.array([h_min_1, s_min, v_min])
    upper1 = np.array([h_max_1, s_max, v_max])

    lower2 = np.array([h_min_2, s_min, v_min])
    upper2 = np.array([h_max_2, s_max, v_max])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)
    mask = cv2.bitwise_or(mask1, mask2)

    # 7. Morphology
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) #흰색 노이즈를 먼저 깎아내고 다시 복원
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) #선 내부의 작은 구멍이나 끊김을 메움

    # 8. Edge Detection
    #이진화된 마스크에서 경계선만 추출. HoughLinesP는 보통 edge 이미지에서 잘 작동
    #mask를 바로 넣기보다 Canny를 거치는 게 일반적   
    edges = cv2.Canny(mask, canny1, canny2)

    # 9. 확률적 허프 변환 - 직선 형태의 선분 검출
    lines = cv2.HoughLinesP(
        edges,
        1,
        np.pi / 180,
        threshold=hough_th,
        minLineLength=min_len,
        maxLineGap=max_gap
    )

    center_x = img_w // 2
    cv2.line(result_img, (center_x, 0), (center_x, img_h), (0, 255, 255), 2)

    best_line = None
    best_length = 0

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # ROI에서 검출했기 때문에, y좌표를 원본 이미지 기준으로 다시 보정
            y1_full = y1 + roi_y
            y2_full = y2 + roi_y

            length = np.sqrt((x2 - x1) ** 2 + (y2_full - y1_full) ** 2)# 현재 선분의 길이 계산

            cv2.line(result_img, (x1, y1_full), (x2, y2_full), (0, 100, 0), 1)  # 검출된 모든 후보 선
            #가장 긴 선을 대표 선으로
            if length > best_length:
                best_length = length
                best_line = (x1, y1_full, x2, y2_full)

    if best_line is not None:
        x1, y1, x2, y2 = best_line
        cv2.line(result_img, (x1, y1), (x2, y2), (0, 255, 0), 3) #대표선 

        # 중심점 계산:
        line_cx = (x1 + x2) // 2  # 대표 선의 양 끝점 평균을 이용해 선의 중점을 구함
        line_cy = (y1 + y2) // 2
        cv2.circle(result_img, (line_cx, line_cy), 6, (0, 0, 255), -1)
        
        # 화면 중앙 대비 오차 계산:
        error_x = line_cx - center_x
        # 각도 계산:
        angle_rad = np.arctan2((y2 - y1), (x2 - x1))
        angle_deg = np.degrees(angle_rad) #좌우 이동 제어나 yaw 보정에 활용

        cv2.putText(result_img, f'Center Error X: {error_x}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, f'Angle: {angle_deg:.1f} deg', (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(result_img, f'Best Length: {best_length:.1f}', (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.line(result_img, (center_x, line_cy), (line_cx, line_cy), (255, 255, 0), 2)

    else:
        cv2.putText(result_img, 'No line detected', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow('Original', frame)
    cv2.imshow('Mask', mask)
    cv2.imshow('Edges', edges)
    cv2.imshow('Line Detection', result_img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()