# 🖐️ Mouse in My Hand 🖱️

### Control your computer cursor with just your hand gestures! 🖐️🤖

This project allows you to use your index finger as a virtual mouse, providing a more natural and immersive way to control your computer. Perfect for accessibility and fun!

---

## 🌟 Features

- **Cursor Control**: Moves with the index finger tip 📍

- **Click Actions**: Left click with thumb-to-index finger joint touch (Joint 7), and right click with a thumb-to-index joint touch (Joint 6) 👍👎

- **Efficient Sensitivity**: Cursor speed responds to finger movement 🏎️

- **Python-Powered with OpenCV**: Seamlessly integrated with computer vision and real-time hand tracking

## 🚀 Getting Started

### Prerequisites

- **Python** (Version 3.7+ recommended)

- **Install Required Libraries**: 
```bash
    pip install opencv-python mediapipe pyautogui
```

### Setup

1\. **Clone the Repository**:
```bash
    git clone https://github.com/Soumyaranjan-17/mouse-in-my-hand.git

    cd mouse-in-my-hand
```

2\. **Run the Application**:
```bash
    python hand_cursor_control.py
```

3\. **Move the Cursor**: Move your index finger around to control the cursor.

4\. **Click Events**:

   - For a **left click**, touch the thumb to the base of your index finger (Joint 7).

   - For a **right click**, touch the thumb to the second joint of your index finger (Joint 6).

---

## 🤲 Contribution Guide

We're excited to welcome contributors during **Hacktoberfest** and beyond! 🎉

1\. **Fork** the repository.

2\. **Create a new branch** for your feature or fix:
```bash
    git checkout -b feature-name
```

3\. **Make your changes** and commit:
```bash
    git commit -m "Added an awesome feature!"
```

4\. **Push** the branch:
```bash
    git push origin feature-name
```

5\. **Create a Pull Request** -- don't forget to include details about your feature!

---

## 🤖 How It Works

Using **MediaPipe** for hand tracking and **OpenCV** for real-time image processing, the program detects hand landmarks and maps them to cursor movements and click actions. With **PyAutoGUI**, these gestures translate to real mouse actions on the screen.

---

## 👏 Acknowledgments

Big thanks to **MediaPipe**, **OpenCV**, and **PyAutoGUI** for making this innovative project possible!

---

Let the **Mouse in My Hand** make your computing experience hands-free and fun! 🚀