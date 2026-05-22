import cv2
import numpy as np

# ==========================================
# 1. 가상 환경(맵) 설정
# ==========================================
GRID_SIZE = 5       # 5x5 격자
CELL_PX = 200       # 한 칸의 픽셀 크기
MAP_PX = GRID_SIZE * CELL_PX
VIEW_PX = 400       # 드론 카메라 뷰 크기

# 흰색 배경에 파란색 선 생성
world_map = np.ones((MAP_PX, MAP_PX, 3), dtype=np.uint8) * 255
LINE_COLOR = (255, 0, 0) # BGR 파란색 (보색 환경 가정)

for i in range(GRID_SIZE + 1):
    cv2.line(world_map, (i * CELL_PX, 0), (i * CELL_PX, MAP_PX), LINE_COLOR, 15)
    cv2.line(world_map, (0, i * CELL_PX), (MAP_PX, i * CELL_PX), LINE_COLOR, 15)

# 🚨 수정된 부분: ID 99번을 사용하기 위해 250개짜리 딕셔너리로 변경
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)

# 맵 곳곳에 흩어진 마커들
marker_positions = {(1, 1): 10, (4, 1): 15, (2, 3): 20, (0, 4): 5, (4, 4): 99} 

for (x, y), mid in marker_positions.items():
    marker_img = cv2.aruco.generateImageMarker(aruco_dict, mid, 100)
    marker_img_bgr = cv2.cvtColor(marker_img, cv2.COLOR_GRAY2BGR)
    
    cx, cy = x * CELL_PX, y * CELL_PX
    
    # 맵 범위를 벗어나지 않도록 안전한 좌표 계산
    y1, y2 = max(0, cy - 50), min(MAP_PX, cy + 50)
    x1, x2 = max(0, cx - 50), min(MAP_PX, cx + 50)
    
    # 마커 이미지도 잘려나간 만큼 맞춰서 자르기
    m_y1 = 50 - (cy - y1)
    m_y2 = 50 + (y2 - cy)
    m_x1 = 50 - (cx - x1)
    m_x2 = 50 + (x2 - cx)
    
    # 안전하게 맵에 합성
    world_map[y1:y2, x1:x2] = marker_img_bgr[m_y1:m_y2, m_x1:m_x2]

# ==========================================
# 2. 매핑 시스템 변수
# ==========================================
drone_logical_x = 0
drone_logical_y = 0
discovered_map = {}

# 카메라 위치 및 지그재그(Lawnmower) 주행 변수
cam_x = 0
cam_y = 0
speed = 15          # 비행 속도
direction_x = 1     # 1: 오른쪽으로 이동, -1: 왼쪽으로 이동
down_step_target = 0 # 아래로 내려가야 할 남은 픽셀

# ==========================================
# 3. 드론 비행 시뮬레이션 및 인식 루프
# ==========================================
print("드론 이륙! 지그재그(Lawnmower) 패턴으로 전 구역 탐색을 시작합니다...")

while True:
    # --- [A] 가상 비행 (지그재그 패턴 주행) ---
    if down_step_target > 0:
        # 아래로 이동 중
        cam_y += speed
        down_step_target -= speed
        if down_step_target <= 0: # 아래로 다 내려왔으면 방향 전환
            direction_x *= -1
    else:
        # 좌우로 이동 중
        cam_x += speed * direction_x
        
        # 오른쪽 끝에 도달했을 때
        if cam_x >= MAP_PX - VIEW_PX and direction_x == 1:
            cam_x = MAP_PX - VIEW_PX
            down_step_target = VIEW_PX // 2  # 시야가 겹치도록 반 칸만 내려감
            
        # 왼쪽 끝에 도달했을 때
        elif cam_x <= 0 and direction_x == -1:
            cam_x = 0
            down_step_target = VIEW_PX // 2

    # 전체 구역(맨 아랫줄) 탐색 완료 체크
    if cam_y >= MAP_PX - VIEW_PX:
        cam_y = MAP_PX - VIEW_PX
        print("\n✅ 전체 구역 탐색 완료! 비행을 종료하고 지도를 생성합니다.")
        break

    frame = world_map[cam_y:cam_y+VIEW_PX, cam_x:cam_x+VIEW_PX].copy()
    
    # --- [B] HSV 기반 라인 디텍팅 ---
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50])
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    
    center = VIEW_PX // 2
    actual_logical_x = round((cam_x + center) / CELL_PX)
    actual_logical_y = round((cam_y + center) / CELL_PX)
    
    if actual_logical_x != drone_logical_x or actual_logical_y != drone_logical_y:
        drone_logical_x = actual_logical_x
        drone_logical_y = actual_logical_y
        print(f"선 통과! 현재 좌표: ({drone_logical_x}, {drone_logical_y})")

    # --- [C] ArUco 마커 디텍팅 및 매핑 ---
    try:
        corners, ids, rejected = cv2.aruco.detectMarkers(frame, aruco_dict)
    except AttributeError:
        detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
        corners, ids, rejected = detector.detectMarkers(frame)
    
    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i in range(len(ids)):
            marker_id = ids[i][0]
            if (drone_logical_x, drone_logical_y) not in discovered_map:
                discovered_map[(drone_logical_x, drone_logical_y)] = marker_id
                print(f"📍 맵 업데이트: ({drone_logical_x}, {drone_logical_y}) ➔ 마커 ID {marker_id}")

    # --- [D] 시각화 ---
    cv2.circle(frame, (center, center), 5, (0, 0, 255), -1) # 드론 현재 위치
    cv2.putText(frame, f"Pos: ({drone_logical_x}, {drone_logical_y})", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    
    cv2.imshow("Drone Camera View (Zig-Zag Flying)", frame)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()

# ==========================================
# 4. 종료 후 맵 그림(팝업창) 생성 및 출력
# ==========================================
CELL_SIZE = 120
PADDING = 60

map_width = (GRID_SIZE * CELL_SIZE) + (PADDING * 2)
map_height = (GRID_SIZE * CELL_SIZE) + (PADDING * 2)

final_map_img = np.ones((map_height, map_width, 3), dtype=np.uint8) * 255

# 격자 그리기
for x in range(GRID_SIZE + 1):
    cv2.line(final_map_img, (PADDING + x * CELL_SIZE, PADDING), (PADDING + x * CELL_SIZE, map_height - PADDING), (200, 200, 200), 2)
for y in range(GRID_SIZE + 1):
    cv2.line(final_map_img, (PADDING, PADDING + y * CELL_SIZE), (map_width - PADDING, PADDING + y * CELL_SIZE), (200, 200, 200), 2)

# 마커 및 숫자 표시
for (x, y), mid in discovered_map.items():
    cx = PADDING + x * CELL_SIZE
    cy = PADDING + y * CELL_SIZE
    cv2.rectangle(final_map_img, (cx - 30, cy - 30), (cx + 30, cy + 30), (0, 200, 0), -1)
    cv2.rectangle(final_map_img, (cx - 30, cy - 30), (cx + 30, cy + 30), (0, 100, 0), 2)
    
    text = str(mid)
    text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(final_map_img, text, (cx - text_size[0] // 2, cy + text_size[1] // 2), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# 좌표축(X, Y) 라벨 표시
for x in range(GRID_SIZE + 1):
    cv2.putText(final_map_img, f"X:{x}", (PADDING + x*CELL_SIZE - 15, PADDING - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
for y in range(GRID_SIZE + 1):
    cv2.putText(final_map_img, f"Y:{y}", (PADDING - 40, PADDING + y*CELL_SIZE + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    

cv2.imshow("Final Generated Map", final_map_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

