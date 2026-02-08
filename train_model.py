import face_recognition
import os
import pickle
import numpy as np
from PIL import Image

def train_model():
    """Train the face recognition model using images in the dataset folder."""
    dataset_path = "dataset"
    
    if not os.path.exists(dataset_path):
        print(f"Error: '{dataset_path}' folder not found!")
        print("Please run capture_faces.py first to capture face images.")
        return
    
    known_encodings = []
    known_names = []
    
    print("Training face recognition model...\n")
    
    # Iterate through each person's folder
    for person_name in os.listdir(dataset_path):
        person_path = os.path.join(dataset_path, person_name)
        
        if not os.path.isdir(person_path):
            continue
        
        print(f"Processing '{person_name}'...")
        person_encoding_count = 0
        
        # Iterate through each image
        for image_name in os.listdir(person_path):
            image_path = os.path.join(person_path, image_name)
            
            # Filter for image files
            if not image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            
            try:
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if face_encodings:
                    # Use the first face detected
                    encoding = face_encodings[0]
                    known_encodings.append(encoding)
                    known_names.append(person_name)
                    person_encoding_count += 1
                else:
                    print(f"  ⚠ No face detected in {image_name}")
            
            except Exception as e:
                print(f"  ✗ Error processing {image_name}: {str(e)}")
        
        print(f"  ✓ {person_encoding_count} encoding(s) created for '{person_name}'\n")
    
    if known_encodings:
        # Save encodings to pickle file
        data = {"encodings": known_encodings, "names": known_names}
        with open("encodings.pickle", "wb") as f:
            pickle.dump(data, f)
        print(f"✓ Model trained successfully!")
        print(f"✓ Total encodings: {len(known_encodings)}")
        print(f"✓ Saved to 'encodings.pickle'")
    else:
        print("Error: No valid face encodings found. Please capture more images.")

if __name__ == "__main__":
    train_model()
