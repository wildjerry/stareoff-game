import cv2
import dlib
import imutils
import numpy as np
import matplotlib.pyplot as plt
from imutils import face_utils
import time

# Function to display the image using matplotlib
def show_frame_matplotlib(frame):
    # Convert the frame from BGR (OpenCV format) to RGB (matplotlib format)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    plt.imshow(frame_rgb)
    plt.axis('off')  # Hide the axis for a cleaner view
    plt.show(block=False)  # Show the frame without blocking the main program
    plt.pause(0.001)  # Short pause to allow the figure to refresh

# Define constants for blink detection parameters
EYE_AR_THRESH = 0.2  # Threshold for the Eye Aspect Ratio (EAR) below which a blink is detected
EYE_AR_CONSEC_FRAMES = 0.75  # Minimum consecutive duration (seconds) of frames with EAR below threshold to detect blink
#frame_rate = 30  # Assumed frame rate of 30 frames per second, used to calculate blink duration threshold

# Initialize dlib's face detector and facial landmark predictor model
print("[INFO] Loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()  # Face detection model
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")  # 68-point facial landmarks model

# Function to calculate Eye Aspect Ratio (EAR) for blink detection
def eye_aspect_ratio(eye):
    # Compute the distances between the vertical eye landmarks
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    # Compute the distance between the horizontal eye landmarks
    C = np.linalg.norm(eye[0] - eye[3])
    # Calculate the EAR, a measure of openness of the eye
    ear = (A + B) / (2.0 * C)
    return ear

# Initialize variables for tracking blinks
frame_count = 0  # Total frames processed
blink_detected = False  # Flag to indicate if a blink has been detected
consec_frame_count = 0  # Counter for consecutive frames where EAR is below threshold

# Start the video stream and allow the camera to warm up
vs = cv2.VideoCapture(0)
time.sleep(2.0)

frame_time_running_average = 1/25

last_time=time.time()

# Main loop to process video frames
start_time = time.time()
while True:
    ret, frame = vs.read()  # Capture a frame
    cur_time = time.time()
    frame_time = cur_time - last_time
    last_time = cur_time

    frame_time_running_average = 0.9*frame_time_running_average + 0.1*frame_time
    frame_rate = 1/frame_time_running_average

    if not ret:
        break  # Exit if the frame could not be captured

    frame = imutils.resize(frame, width=600)  # Resize frame for faster processing
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    # Loop over each detected face
    for rect in rects:
        shape = predictor(gray, rect)  # Detect facial landmarks
        shape = face_utils.shape_to_np(shape)  # Convert landmarks to NumPy array

        # Extract coordinates for left and right eyes
        left_eye = shape[36:42]
        right_eye = shape[42:48]

        # Calculate EAR for both eyes and average them
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        ear = (left_ear + right_ear) / 2.0

        # Draw landmarks on eyes for visual reference
        for (x, y) in left_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw circles on left eye
        for (x, y) in right_eye:
            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)  # Draw circles on right eye

        # Check if EAR is below blink threshold
        if ear < EYE_AR_THRESH:
            consec_frame_count += 1  # Increment consecutive frame count
        else:
            consec_frame_count = 0  # Reset if EAR goes above threshold

        # If EAR below threshold for sufficient time, detect a blink
        if consec_frame_count / frame_rate > EYE_AR_CONSEC_FRAMES:
            blink_detected = True

        # Display EAR value on the frame for reference
        cv2.putText(frame, f"EAR: {ear:.2f} frame_rate: {frame_rate:.2f} consec_frame_count: {consec_frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame using matplotlib
    show_frame_matplotlib(frame)

    # If blink detected, exit loop and close video
    if blink_detected:
        print("Blink detected, exiting game...")
        plt.close('all')  # Close all matplotlib windows
        break

    # Optional: Allow manual exit if 'q' is pressed
    if plt.waitforbuttonpress(timeout=0.01):
        plt.close('all')
        print("User exited with key press.")
        break

# Release resources
vs.release()
cv2.destroyAllWindows()
