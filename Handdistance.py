import cv2
import mediapipe as mp
import math

# MediaPipe 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# 웹캠 열기
cap = cv2.VideoCapture(0)

# 유클리드 거리 계산 함수
def calculate_distance(point1, point2, width, height):
    x1, y1 = int(point1.x * width), int(point1.y * height)
    x2, y2 = int(point2.x * width), int(point2.y * height)
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    return distance

# 손가락별 랜드마크 매핑
FINGER_TIPS = [
    mp_hands.HandLandmark.THUMB_TIP,
    mp_hands.HandLandmark.INDEX_FINGER_TIP,
    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
    mp_hands.HandLandmark.RING_FINGER_TIP,
    mp_hands.HandLandmark.PINKY_TIP
]
FINGER_MCPS = [
    mp_hands.HandLandmark.THUMB_MCP,
    mp_hands.HandLandmark.INDEX_FINGER_MCP,
    mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
    mp_hands.HandLandmark.RING_FINGER_MCP,
    mp_hands.HandLandmark.PINKY_MCP
]

with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # BGR을 RGB로 변환
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # 손 감지 수행
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                # 랜드마크 그리기
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # 화면 크기 가져오기
                height, width, _ = frame.shape

                # 각 손가락의 TIP-MCP 거리 계산
                for i in range(len(FINGER_TIPS)):
                    tip = hand_landmarks.landmark[FINGER_TIPS[i]]
                    mcp = hand_landmarks.landmark[FINGER_MCPS[i]]

                    # 거리 계산
                    distance = calculate_distance(tip, mcp, width, height)

                    # 손가락별 이름 표시
                    finger_name = ["Thumb", "Index", "Middle", "Ring", "Pinky"][i]

                    # 거리 및 손가락 이름 출력
                    cv2.putText(frame, f'{finger_name}: {int(distance)} px', 
                                (10, 30 + i * 30),  # 손가락별로 위치 조정
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 화면 출력
        cv2.imshow('Hand Tracking - All Fingers', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
