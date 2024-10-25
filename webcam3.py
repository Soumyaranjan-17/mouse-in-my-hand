import cv2
import mediapipe as mp
import pyautogui
import math

# Initialize MediaPipe Hand
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Capture video feed
cap = cv2.VideoCapture(0)
screen_width, screen_height = pyautogui.size()

# Function to calculate distance between two points
def calculate_distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

# Main loop
prev_x, prev_y = None, None  # Previous cursor position for relative movement
sensitivity = 2  # Adjust sensitivity to control cursor speed

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks
    result = hands.process(frame_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get landmark positions for index_tip and thumb_tip
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_joint7 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_PIP]
            index_joint6 = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_DIP]

            # Calculate index_tip position in screen coordinates
            x = int(index_tip.x * screen_width)
            y = int(index_tip.y * screen_height)

            # Relative movement based on previous cursor position
            if prev_x is None or prev_y is None:
                prev_x, prev_y = x, y

            dx, dy = (x - prev_x) * sensitivity, (y - prev_y) * sensitivity
            pyautogui.moveRel(dx, dy)
            prev_x, prev_y = x, y  # Update previous position

            # Calculate distances for click detection
            thumb_index_distance7 = calculate_distance(
                (thumb_tip.x, thumb_tip.y), (index_joint7.x, index_joint7.y)
            )
            thumb_index_distance6 = calculate_distance(
                (thumb_tip.x, thumb_tip.y), (index_joint6.x, index_joint6.y)
            )

            # Left click if thumb is close to index joint 7
            if thumb_index_distance7 < 0.05:  # Adjust threshold as needed
                pyautogui.click(button='left')
                cv2.putText(frame, 'Left Click', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Right click if thumb is close to index joint 6
            elif thumb_index_distance6 < 0.05:  # Adjust threshold as needed
                pyautogui.click(button='right')
                cv2.putText(frame, 'Right Click', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            # Draw landmarks and connections
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Hand Tracking", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
