# Face Recognition & Greeting

A Python-based face recognition system that captures face images, trains a model, and marks attendance automatically.

## Features
- **Capture Faces**: Record multiple face images for each person
- **Train Model**: Generate facial encodings using deep learning
- **Real-time Recognition**: Identify and mark attendance automatically
- **Timestamp Logging**: Records attendance with date and time
- **Easy to Use**: Simple command-line interface

## Setup (Windows / Python 3.12 tested)

### 1) Create and activate a virtual environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip


## Quick Start

### 1. Capture Faces
```bash
python capture_faces.py
```
Enter a person's name and press 'C' to capture photos (need at least 10).

### 2. Train Model
```bash
python train_model.py
```
This creates facial encodings from captured images.

### 3. Mark Attendance
```bash
python recognize_and_greet.py
```
Real-time face recognition and automatic attendance marking.

“This uses pretrained face encodings (no deep model training).”

## How It Works
1. **Face Capture**: Saves face images in `dataset/[name]/`
2. **Encoding**: Converts faces to 128-dimensional vectors
3. **Recognition**: Compares real-time faces with stored encodings
4. **Attendance**: Records recognized faces in `attendance.csv`

## Requirements
- Python 3.7+
- setuptools<81
- opencv-python>=4.5.0
- face-recognition>=1.3.0
- numpy>=1.19.0
- pyttsx3>=2.90
- pywin32
- dlib-bin>=20.0.0

Install: `pip install dlib-bin`
Install: `pip install -r requirements.txt`
Install: `pip install git+https://github.com/ageitgey/face_recognition_models`
Install: `pip install face-recognition==1.3.0 --no-deps`
## Windows / Python 3.12 Notes

This project depends on `face_recognition_models`, which currently uses `pkg_resources` from `setuptools`.  
On Python 3.12+, you may need to pin setuptools to a version that still includes `pkg_resources`:

```bash
pip install "setuptools<81"

## Output
Attendance marked in `attendance.csv`:
```
Name,Timestamp
Jayneel,2026-02-08 15:53:29


## GUI

bash
python app.py


## License
MIT License

Made by Jayneel with ❤️

## Isaac Notes
“On Python 3.12+, install setuptools<81 because face_recognition_models depends on pkg_resources.”
