import face_recognition
import cv2
import pickle
import os
import numpy as np
import time
from speech import speak


def recognize_and_greet():
    #OLD """Recognize faces in real-time and mark attendance."""
    """Recognizes faces in real-time and greets the associated person"""
    
    # Load encodings
    if not os.path.exists("encodings.pickle"):
        print("Error: encodings.pickle not found!")
        print("Please run train_model.py first.")
        return
    
    with open("encodings.pickle", "rb") as f:
        data = pickle.load(f)
        known_encodings = data["encodings"]
        known_names = data["names"]
    
    print("Starting face recognition...\n")
    print("Press 'Q' to quit")
    
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open webcam!")
        speak("Error: Could not open webcam!")
        return
    
    # Tolerance for face matching (lower = more strict)
    tolerance = 0.6
    
    # Track recognized people in this session
    greeted_this_session = set()
    # Track the last time an unknown face was detected (so it won't keep printing the same message constantly)
    # Since computer time starts in 1970 this works as the last unknown being effectively never.
    UNKNOWN_SPEECH_COOLDOWN = 2.0
    last_unknown_print = 1000.0
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("Error: Failed to read from camera!")
            speak("Error: Failed to read from camera!")
            break
        
        # Resize for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        rgb_small_frame = np.ascontiguousarray(rgb_small_frame)
        
        # Detect faces and encodings
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_names = []
        face_distances = []
        
        for face_encoding in face_encodings:
            # Compare with known encodings
            distances = face_recognition.face_distance(known_encodings, face_encoding)
            name = "Unknown"
            confidence = 0
            
            if distances.size > 0:
                best_match_index = distances.argmin()
                if distances[best_match_index] < tolerance:
                    name = known_names[best_match_index]
                    confidence = 1 - distances[best_match_index]
            
            face_names.append(name)
            face_distances.append(confidence)
        
        # (OLD) Mark attendance
        #for name, confidence in zip(face_names, face_distances):
        #    if name != "Unknown" and name not in greeted_this_session:
        #        # Mark attendance
        #        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        #        with open("attendance.csv", "a") as f:
        #            f.write(f"{name},{timestamp}\n")
        #        print(f"✓ Attendance marked: {name} at {timestamp}")
        #        greeted_this_session.add(name)

        # Mark attendance
        for name, confidence in zip(face_names, face_distances):
            if name != "Unknown" and name not in greeted_this_session:
                # Mark attendance
                print(f"Hello {name}")
                speak(f"Hello {name}")
                greeted_this_session.add(name)
            elif name == "Unknown":
                current_time = time.time()
                if current_time - last_unknown_print > UNKNOWN_SPEECH_COOLDOWN:
                    print("Unknown face detected. Please return to the main menu and register your face so that I can mark your attendance")
                    speak("Unknown face detected. Please return to the main menu and register your face so that I can mark your attendance")
                    last_unknown_print = current_time
                      
        
        # Display results
        for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, face_distances):
            # Scale back up
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            
            # Color based on recognition
            if name == "Unknown":
                color = (0, 0, 255)  # Red
            else:
                color = (0, 255, 0)  # Green
            
            # Draw box
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            
            # Draw label
            label = f"Hello {name}" if name != "Unknown" else "Unknown"
            if confidence > 0:
                label += f" ({confidence:.2f})"
            cv2.putText(frame, label, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        cv2.imshow("Face Recognition - Press 'Q' to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print("\nFace recognition stopped.")
    speak("Face recognition stopped.")

if __name__ == "__main__":
    recognize_and_greet()
