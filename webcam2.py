import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import math

# Initialize MediaPipe for hand detection and drawing
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Set initial cursor position and parameters for sensitivity control
screen_width, screen_height = pyautogui.size()
prev_cursor_x, prev_cursor_y = screen_width / 2, screen_height / 2  # Start from the screen center
sensitivity_factor = 9.0  # Adjust to make the cursor more or less responsive
z_threshold = -0.2  # Threshold for detecting click based on finger height
left_click_triggered = False  # Flag to prevent multiple clicks from being registered

# Capture video feed
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and process the frame for MediaPipe
    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get normalized hand landmarks and convert to pixel values
            index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            index_x, index_y, index_z = int(index_tip.x * frame.shape[1]), int(index_tip.y * frame.shape[0]), index_tip.z

            # Calculate relative movement and sensitivity
            cursor_dx = (index_x - prev_cursor_x) / (sensitivity_factor + math.log1p(abs(index_x - prev_cursor_x)))
            cursor_dy = (index_y - prev_cursor_y) / (sensitivity_factor + math.log1p(abs(index_y - prev_cursor_y)))

            # Update cursor position
            cursor_x = prev_cursor_x + cursor_dx
            cursor_y = prev_cursor_y + cursor_dy

            # Control the system cursor
            pyautogui.moveTo(cursor_x, cursor_y)
            prev_cursor_x, prev_cursor_y = cursor_x, cursor_y

            # Click Detection based on height (z-axis) position
            if index_z < z_threshold:
                if not left_click_triggered:
                    pyautogui.click(button='left')
                    left_click_triggered = True
            else:
                left_click_triggered = False

    # Display the frame
    cv2.imshow("Hand Gesture Controlled Cursor", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
