import cv2
import mediapipe as mp
import pyautogui

# Initialize MediaPipe Hands and Drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Parameters
sensitivity = 0.8  # Adjust this value for sensitivity

# Capture video from the default camera
cap = cv2.VideoCapture(0)

# Use Hands module
with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:

    # Get initial cursor position
    last_x, last_y = pyautogui.position()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Cannot receive frame. Exiting ...")
            break

        # Flip the frame horizontally for a mirrored effect
        frame = cv2.flip(frame, 1)

        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the image and detect hands
        results = hands.process(rgb_frame)

        # Draw hand landmarks on the frame if hands are detected
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get the landmark positions
                landmarks = hand_landmarks.landmark

                # Extract positions of the required landmarks
                thumb_tip = landmarks[4]     # Thumb tip
                index_tip = landmarks[8]      # Index finger tip
                middle_tip = landmarks[12]    # Middle finger tip
                index_base = landmarks[6]     # Index finger MCP
                middle_base = landmarks[10]    # Middle finger MCP

                # Calculate cursor movement based on index finger position
                screen_width, screen_height = pyautogui.size()
                finger_x = int(index_tip.x * screen_width)
                finger_y = int(index_tip.y * screen_height)

                # Calculate movement
                delta_x = finger_x - last_x
                delta_y = finger_y - last_y

                # Adjust movement by sensitivity
                new_x = last_x + delta_x / sensitivity
                new_y = last_y + delta_y / sensitivity

                # Move the cursor
                pyautogui.moveTo(new_x, new_y)

                # Update last cursor position
                last_x, last_y = new_x, new_y

                # Gesture Recognition
                # Check for Pinch Out Gesture
                if (thumb_tip.x < index_tip.x) and (thumb_tip.y < index_tip.y):
                    cv2.putText(frame, "Gesture: Pinch Out", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Scroll up
                    pyautogui.scroll(10)

                # Check for Pinch In Gesture
                elif (thumb_tip.x > index_tip.x) and (thumb_tip.y > index_tip.y):
                    cv2.putText(frame, "Gesture: Pinch In", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    # Scroll down
                    pyautogui.scroll(-10)

                # Check for Index Finger Click
                elif (index_tip.y < index_base.y) and (middle_tip.y > middle_base.y):
                    cv2.putText(frame, "Gesture: Index Click", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyautogui.click()  # Simulate click

                # Check for Middle Finger Click
                elif (middle_tip.y < middle_base.y) and (index_tip.y > index_base.y):
                    cv2.putText(frame, "Gesture: Middle Click", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyautogui.click()  # Simulate click

                # Check for Index Finger Up Gesture
                elif index_tip.y < index_base.y:
                    cv2.putText(frame, "Gesture: Index Finger Up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Check for Middle Finger Up Gesture
                elif middle_tip.y < middle_base.y:
                    cv2.putText(frame, "Gesture: Middle Finger Up", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Display the processed frame
        cv2.imshow('Gesture Control', frame)

        # Press 'q' to exit the video window
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
