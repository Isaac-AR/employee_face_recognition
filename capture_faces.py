import cv2
import os

def capture_faces():
    """Capture face images for a person and save them to the dataset folder."""
    name = input("Enter person name: ").strip()
    
    if not name:
        print("Error: Name cannot be empty!")
        return
    
    save_path = os.path.join("dataset", name)
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    
    if not cam.isOpened():
        print("Error: Could not open webcam. Please check your camera connection.")
        return
    
    print(f"\nCapturing faces for '{name}'...")
    print("Press 'C' to capture a photo (need 10 photos)")
    print("Press 'Q' to skip this person")
    
    count = 0
    
    while True:
        ret, frame = cam.read()
        
        if not ret:
            print("Error: Failed to read from camera!")
            break
        
        # Display current count on frame
        cv2.putText(frame, f"Photos: {count}/10", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'C' to capture, 'Q' to quit", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        cv2.imshow("Capture Faces - Press 'C' to capture, 'Q' to quit", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c') or key == ord('C'):
            img_path = os.path.join(save_path, f"img{count}.jpg")
            cv2.imwrite(img_path, frame)
            print(f"✓ Saved {img_path}")
            count += 1
        
        elif key == ord('q') or key == ord('Q'):
            print(f"Stopped capturing for '{name}'. {count} images saved.")
            break
        
        if count >= 10:
            print(f"✓ Captured 10 images for '{name}'!")
            break
    
    cam.release()
    cv2.destroyAllWindows()
    print(f"Dataset collection complete for '{name}'. {count} image(s) saved to '{save_path}'.\n")

if __name__ == "__main__":
    capture_faces()
