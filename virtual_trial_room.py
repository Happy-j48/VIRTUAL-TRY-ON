import cv2
import mediapipe as mp
import numpy as np
import time
import threading

# Initialize Mediapipe Pose
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Known width of the reference object in centimeters (e.g., a credit card is 8.5 cm)
REFERENCE_WIDTH_CM = 8.5

def compute_real_size(landmark1, landmark2, frame_shape, ref_pixel_width):
    # Calculate distance in pixels between two landmarks
    landmark1_px = (int(landmark1.x * frame_shape[1]), int(landmark1.y * frame_shape[0]))
    landmark2_px = (int(landmark2.x * frame_shape[1]), int(landmark2.y * frame_shape[0]))
    pixel_distance = np.linalg.norm(np.array(landmark2_px) - np.array(landmark1_px))
    
    # Calculate real-world distance in centimeters
    cm_per_pixel = REFERENCE_WIDTH_CM / ref_pixel_width
    real_distance_cm = pixel_distance * cm_per_pixel
    
    return real_distance_cm

def estimate_tshirt_size(shoulder_width_cm, waist_width_cm):
    # Example size estimation logic based on measurements
    if shoulder_width_cm < 40 and waist_width_cm < 70:
        return "Small"
    elif shoulder_width_cm < 45 and waist_width_cm < 80:
        return "Medium"
    elif shoulder_width_cm < 50 and waist_width_cm < 90:
        return "Large"
    else:
        return "Extra Large"

def overlay_tshirt(frame, landmarks, tshirt_img, scale_factor=1.5):  # Increased scale factor for bigger T-shirt
    # Get the shoulder and hip landmarks
    shoulder_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
    shoulder_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
    hip_left = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP.value]
    hip_right = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP.value]

    # Calculate the width and height of the T-shirt area, increasing size by scale_factor
    shoulder_width = int(abs(shoulder_right.x - shoulder_left.x) * frame.shape[1] * scale_factor)
    height = int(abs(hip_left.y - shoulder_left.y) * frame.shape[0] * scale_factor)  # Use shoulder-to-hip distance for height

    # Ensure that the T-shirt width and height are valid
    if shoulder_width <= 0 or height <= 0:
        print("Invalid shoulder width or height detected, skipping overlay.")
        return frame

    # Resize the T-shirt image to fit the body
    tshirt_resized = cv2.resize(tshirt_img, (shoulder_width, height))

    # Check if the T-shirt image has an alpha channel (RGBA)
    if tshirt_resized.shape[2] == 3:  # No alpha channel
        # Add an alpha channel (fully opaque)
        alpha_channel = np.ones((tshirt_resized.shape[0], tshirt_resized.shape[1], 1), dtype=tshirt_resized.dtype) * 255
        tshirt_resized = np.concatenate((tshirt_resized, alpha_channel), axis=2)

    # Calculate the position to overlay the T-shirt
    x = int((shoulder_left.x + shoulder_right.x) / 2 * frame.shape[1]) - int(shoulder_width / 2)  # Centering the T-shirt on the body
    y = int(shoulder_left.y * frame.shape[0]) - int(tshirt_resized.shape[0] / 4)  # Adjust position for better fit

    # Ensure the T-shirt fits within the frame boundaries
    if x < 0:  # Prevent T-shirt from going off the left side of the frame
        x = 0
    if x + shoulder_width > frame.shape[1]:  # Prevent T-shirt from going off the right side of the frame
        x = frame.shape[1] - shoulder_width
    if y + height > frame.shape[0]:  # Ensure T-shirt doesn't go off the bottom of the frame
        height = frame.shape[0] - y
        tshirt_resized = cv2.resize(tshirt_img, (shoulder_width, height))

    # Overlay the T-shirt on the frame
    alpha_tshirt = tshirt_resized[:, :, 3] / 255.0
    for c in range(0, 3):
        frame[y:y+height, x:x+shoulder_width, c] = (alpha_tshirt * tshirt_resized[:, :, c] + 
                                                    (1 - alpha_tshirt) * frame[y:y+height, x:x+shoulder_width, c])

    return frame

# Load multiple T-shirt images with transparency (PNG format)
tshirt_images = [
    cv2.imread('tshirt1.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt2.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt3.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt4.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt5.png', cv2.IMREAD_UNCHANGED),
    cv2.imread('tshirt6.png', cv2.IMREAD_UNCHANGED),
]

# Initialize variables for automatically changing T-shirts
tshirt_index = 0
change_interval = 3  # Time interval (in seconds) to automatically change the T-shirt

def run_virtual_trial_room():
    global tshirt_index
    # Capture Video from Webcam
    cap = cv2.VideoCapture(0)

    # Set the OpenCV window to be maximized
    cv2.namedWindow("Virtual Trial Room", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty("Virtual Trial Room", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    last_change_time = time.time()  # Track the time for automatic change

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert the frame to RGB for Mediapipe processing
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)

            if results.pose_landmarks:
                # Draw pose landmarks on the frame
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                # Assume reference object is visible and measured in pixels
                ref_pixel_width = 150  # Example value; replace with actual measurement

                # Compute body measurements in centimeters
                shoulder_width_cm = compute_real_size(results.pose_landmarks.landmark[11], results.pose_landmarks.landmark[12], frame.shape, ref_pixel_width)
                waist_width_cm = compute_real_size(results.pose_landmarks.landmark[23], results.pose_landmarks.landmark[24], frame.shape, ref_pixel_width)

                # Estimate T-shirt size
                tshirt_size = estimate_tshirt_size(shoulder_width_cm, waist_width_cm)

                # Overlay the T-shirt on the frame
                frame = overlay_tshirt(frame, results.pose_landmarks, tshirt_images[tshirt_index])

                # Display measurements and estimated size on the frame
                cv2.putText(frame, f'Shoulder Width: {shoulder_width_cm:.2f} cm', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'Waist Width: {waist_width_cm:.2f} cm', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, f'T-Shirt Size: {tshirt_size}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # Show the frame with annotations
            cv2.imshow('Virtual Trial Room', frame)

            # Automatically change T-shirt every 'change_interval' seconds
            if time.time() - last_change_time > change_interval:
                tshirt_index = (tshirt_index + 1) % len(tshirt_images)  # Change the T-shirt
                last_change_time = time.time()  # Reset the time tracker

            # Exit the loop when 'q' or 'ESC' is pressed
            if cv2.waitKey(1) & 0xFF in [27, ord('q')]:
                break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

# Run the virtual trial room in a separate thread
thread = threading.Thread(target=run_virtual_trial_room)
thread.start()
