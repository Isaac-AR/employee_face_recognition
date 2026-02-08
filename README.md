# Face Recognition Attendance System

A Python-based face recognition system that captures face images, trains a model, and marks attendance automatically.

## Features
- **Capture Faces**: Record multiple face images for each person
- **Train Model**: Generate facial encodings using deep learning
- **Real-time Recognition**: Identify and mark attendance automatically
- **Timestamp Logging**: Records attendance with date and time
- **Easy to Use**: Simple command-line interface

## Quick Start

### 1. Capture Faces
```bash
python3 capture_faces.py
```
Enter a person's name and press 'C' to capture photos (need at least 10).

### 2. Train Model
```bash
python3 train_model.py
```
This creates facial encodings from captured images.

### 3. Mark Attendance
```bash
python3 recognize_and_greet.py
```
Real-time face recognition and automatic attendance marking.

## How It Works
1. **Face Capture**: Saves face images in `dataset/[name]/`
2. **Encoding**: Converts faces to 128-dimensional vectors
3. **Recognition**: Compares real-time faces with stored encodings
4. **Attendance**: Records recognized faces in `attendance.csv`

## Requirements
- Python 3.7+
- opencv-python
- face-recognition
- numpy

Install: `pip install -r requirements.txt`

## Output
Attendance marked in `attendance.csv`:
```
Name,Timestamp
Jayneel,2026-02-08 15:53:29
```

## License
MIT License

Made by Jayneel with ❤️
