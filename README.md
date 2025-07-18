# Ask the Wizard

Ask the Wizard is a fun and interactive computer vision project that uses webcam, voice command support and keyboard shortcuts to deliver a magical user experience.

## Features

- **Face Detection:** Uses Haar cascades to detect faces in the webcam feed
- **Emotion Recognition:** Analyzes facial expressions using DeepFace and displays corresponding emojis
- **Background Change:** Switches your background with random images from the `bg/` folder
- **Time Filter:** Hide or reveal your identity with blur and sharpening effects
- **Voice Commands:** Control the application with your voice using speech recognition
- **Keyboard Shortcuts:** Instantly trigger actions with simple key presses

## Setup Intructions

### Prerequisites

- Python 3.7+
- Webcam

### Installation

Clone the repository:
```bash
git clone https://github.com/LeLuke007/Ask-the-Wizard.git
cd Ask-the-Wizard
```

Install dependencies:
```bash
pip install -r requirements.txt
```

Run the application:
```bash
python main.py
```

### Customization

- Add your own background images to the `bg/` folder
- Add / replace emoji images in the `img/` folder
- Update Haar cascade files in `haarcascades/` for improved detection

## Usage

1. **Voice Controls:** Speak commands like "change background", "mood", "change time", "home", or "exit"

2. **Keyboard Controls:**
    - `d` : Change background
    - `t` : Hide/reveal yourself (toggle time filter)
    - `m` : Discover your mood (emotion detection)
    - `h` : Return to home screen
    - `q` : Exit the application