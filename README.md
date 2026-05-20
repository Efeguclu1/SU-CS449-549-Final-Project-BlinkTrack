# BlinkTrack

Hands-free interaction system enabling cursor control and selection through real-time gaze tracking and blink detection.
Developed for **CS449 / CS549 – Human–Computer Interaction** (Sabancı University).

## Overview
BlinkTrack explores a low-cost, webcam-based approach for hands-free interaction. The system tracks gaze direction via iris landmarks to move the cursor and detects intentional eye blinks (single, double, prolonged) to issue interaction commands such as clicking and menu selection.

The project targets accessibility scenarios where traditional mouse/keyboard input is unavailable, using only a standard webcam — no specialized eye-tracking hardware required.

## Features
- Real-time gaze landmark extraction via **MediaPipe FaceMesh** (iris + eye corners)
- Blink detection using the **Eye Aspect Ratio (EAR)** with single / double / prolonged classification
- Cursor control through head/gaze mapping (**pyautogui**)
- **Fitts' Law** target-acquisition demo with CSV logging for performance evaluation
- **Menu navigation demo** matching the Figma prototype (gaze hover + double-blink to select)
- Integrated end-to-end demo combining gaze cursor + blink-click
- Low-fidelity UI prototypes (Figma)

## Directory Structure
```
.
├── prototypes/
│   └── figma/                      # Low-fi UI mockups (PNG exports)
│       ├── Cursor tracking mockup.png
│       ├── blink flow  (interaction flow).png
│       └── gaze hover.png
└── src/
    ├── main.py                     # Demo launcher (menu of all demos)
    ├── blinktrack_integrated.py    # Full system: gaze cursor + double-blink click
    ├── fitts_demo.py               # Fitts' Law target study with CSV logging
    ├── menu_demo.py                # Menu navigation via gaze + blink
    ├── blink_detection/
    │   └── blink_test.py           # EAR-based blink detection test
    ├── cursor_control/
    │   └── cursor_test.py          # pyautogui cursor mapping test
    └── gaze_tracking/
        └── gaze_test.py            # MediaPipe FaceMesh gaze test
```

## Requirements
- Python 3.8+
- A working webcam
- macOS / Windows / Linux

### Python dependencies
```bash
pip install opencv-python mediapipe numpy pyautogui
```

> On macOS, grant **Camera** and **Accessibility** permissions to your terminal/IDE so `pyautogui` can move the cursor.

## Getting Started

Clone the repository:
```bash
git clone https://github.com/Efeguclu1/SU-CS449-549-Final-Project-BlinkTrack.git
cd SU-CS449-549-Final-Project-BlinkTrack
```

Launch the demo menu:
```bash
python src/main.py
```

Or run a specific demo directly:
```bash
python src/blinktrack_integrated.py   # gaze cursor + blink click
python src/fitts_demo.py              # Fitts' Law study
python src/menu_demo.py               # menu navigation
```

Press `q` (or `ESC`) inside any demo window to exit.

## How It Works
1. **Face & iris landmarks** — MediaPipe FaceMesh produces 478 landmarks per frame, including iris points (468–477).
2. **Gaze estimation** — the iris center is normalized against the eye corners to produce a 2D gaze vector, then mapped to screen coordinates.
3. **Blink detection** — the Eye Aspect Ratio (EAR) is computed from six landmarks per eye. EAR dropping below a threshold for N consecutive frames signals a blink; pairs of blinks within a short window are classified as a **double blink** (click).
4. **Action dispatch** — `pyautogui` moves the cursor and issues click events on detected gestures.

## Milestones
- **Milestone 1** — Figma prototypes, gaze landmark extraction, basic EAR blink detection, initial cursor experiments.
- **Milestone 2** — Integrated gaze-cursor + blink-click, menu navigation demo, Fitts' Law evaluation harness.

## Course
Sabancı University — CS449 / CS549 Human–Computer Interaction · Final Project.

## License
Academic project. See repository for licensing details.
