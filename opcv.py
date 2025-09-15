import cv2
import os
import time
import matplotlib.pyplot as plt
from roboflow import Roboflow
import numpy as np

def main():
    # ===============================
    # Roboflow Setup with Error Handling
    # ===============================
    try:
        print("Loading Roboflow workspace...")
        rf = Roboflow(api_key="jsMpdaGvUYSD2zSsW1SJ")   # your API key

        print("Loading Roboflow project...")
        project = rf.workspace("fsfhdg").project("my-first-project-dvz6b")
        model = project.version(1).model   
        
        print("Roboflow model loaded successfully!")
    except Exception as e:
        print(f"Error initializing Roboflow: {e}")
        print("Please check your API key, workspace name, and project name.")
        return

    # ===============================
    # Run Real-Time Webcam Detection with matplotlib display
    # ===============================
    print("Starting webcam... Close the matplotlib window to quit")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not access webcam")
        return

    # Set a reasonable frame size
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Create matplotlib figure
    plt.ion()
    fig, ax = plt.subplots(1, 1)
    
    # To avoid saving too many frames, we'll process only every few frames
    frame_counter = 0
    process_every_n_frames = 10  # Process every 10th frame to reduce load
    
    try:
        while plt.fignum_exists(fig.number):
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame")
                break

            frame_counter += 1
            
            # Only process every Nth frame to reduce load on API
            if frame_counter % process_every_n_frames == 0:
                try:
                    # Save current frame temporarily
                    cv2.imwrite("temp_frame.jpg", frame)

                    # Get predictions from Roboflow with confidence threshold
                    preds = model.predict("temp_frame.jpg", confidence=40, overlap=30).json()

                    # Draw boxes on detected objects
                    if "predictions" in preds:
                        for pred in preds["predictions"]:
                            x, y, w, h = pred["x"], pred["y"], pred["width"], pred["height"]
                            label = pred["class"]
                            confidence = pred["confidence"]

                            # Convert center x,y to top-left & bottom-right
                            x1, y1 = int(x - w/2), int(y - h/2)
                            x2, y2 = int(x + w/2), int(y + h/2)

                            # Draw box + label
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            label_text = f"{label} ({confidence:.2f})"
                            cv2.putText(frame, label_text, (x1, y1-10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    
                    # Remove temporary file
                    if os.path.exists("temp_frame.jpg"):
                        os.remove("temp_frame.jpg")
                        
                except Exception as e:
                    print(f"Error during prediction: {e}")
                    # Continue running even if there's an error with prediction

            # Convert BGR to RGB for matplotlib
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update the matplotlib display
            ax.clear()
            ax.imshow(frame_rgb)
            ax.set_title("Roboflow Detection - Close window to exit")
            ax.axis('off')
            plt.pause(0.01)
            
            # Check if user wants to close
            if not plt.fignum_exists(fig.number):
                break
                
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        cap.release()
        plt.ioff()
        plt.close()
        
        # Clean up temporary file if it exists
        if os.path.exists("temp_frame.jpg"):
            os.remove("temp_frame.jpg")

if __name__ == "__main__":
    main()