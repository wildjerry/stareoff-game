import cv2
import dlib
import imutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from imutils import face_utils
import time
import sqlite3
from uuid import uuid4


#frame_artist = None #https://stackoverflow.com/questions/28692246/matplotlib-draw-is-slow-in-loop-when-it-showing-an-image

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
blink_detected = False  # Flag to indicate if a blink has been detected
consec_frame_count = 0  # Counter for consecutive frames where EAR is below threshold

db_connection = sqlite3.connect('stare.db')
db_cursor = db_connection.cursor()

player_uuid = str(uuid4())

def add_to_db(score, face):
    #based on https://www.geeksforgeeks.org/how-to-insert-image-in-sqlite-using-python/
    sqlite_insert_blob_query = """ INSERT INTO Leaderboard
                                 (image, score, uuid) VALUES (?, ?, ?)"""
    data_tuple = (face, score, player_uuid)
    db_cursor.execute(sqlite_insert_blob_query, data_tuple)
    db_connection.commit()

def display_leaderboard():
    sqlite_query = '''
        WITH RankedLeaderboard AS (
            SELECT *, RANK() OVER (ORDER BY score DESC) AS rank
            FROM Leaderboard
        ),
        Top5 AS (
            SELECT * FROM RankedLeaderboard
            WHERE rank <= 5
        )
        SELECT * FROM Top5
        UNION ALL
        SELECT * FROM RankedLeaderboard
        WHERE uuid = ? AND uuid NOT IN (SELECT uuid FROM Top5);

    '''
    db_cursor.execute(sqlite_query, [player_uuid])
    leaderboard = db_cursor.fetchall()

    fig, ax = plt.subplots(3,2)

    for a in ax.flat:
        a.axis("off")

    for i, row in enumerate(leaderboard):
        lb_img = cv2.imdecode( np.frombuffer(row[0], dtype=np.uint8 ), cv2.IMREAD_COLOR)
        lb_img = cv2.cvtColor(lb_img, cv2.COLOR_BGR2RGB)
        lb_score = row[1]
        lb_uuid = row[2]
        lb_rank = row[3]

        r, c = divmod(i, 2)
        ax[r,c].imshow(lb_img)
        ax[r, c].set_title(f"#{lb_rank}: {lb_score:.2f} seconds", fontsize=10)
    plt.show()




# Start the video stream and allow the camera to warm up
vs = cv2.VideoCapture(0)
time.sleep(2.0)

frame_time_running_average = 1 #starting point for average-a really high default works well because it stablizes in around 10-15 frames no matter what, but those 10-15 frames are a lot quicker at high FPS.


last_time=time.time()

leaderboard_image = None

rects = None #a rectangle for each face, from the detector

# Main loop to process video frames
start_time = time.time()

while True:
    plt.clf()
    plt.cla()

    frame_rate = 1/frame_time_running_average

    frame_begin = time.time()

    ret, full_frame = vs.read()  # Capture a frame

    if not ret:
        break  # Exit if the frame could not be captured

    frame = imutils.resize(full_frame, width=600)  # Resize frame for faster processing

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for face detection
    gray = cv2.equalizeHist(gray)

    # Detect faces in the grayscale frame
    rects = detector(gray, 0)

    if leaderboard_image is None and len(rects)!=0:
        leaderboard_image = cv2.imencode('.jpg', full_frame)[1].tobytes()

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
        cv2.putText(frame, f"EAR: {ear:.2f} frame_rate: {frame_rate:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display the frame using matplotlib
    show_frame_matplotlib(frame)

    if blink_detected:
        add_to_db(time.time()-start_time, leaderboard_image)
    
    if blink_detected or plt.waitforbuttonpress(timeout=0.01):
        plt.close()
        display_leaderboard()
        break;

    if rects: #don't count frames when no face is detected, because they're a lot faster and will shift the timing
        frame_time = time.time() - frame_begin
        frame_time_running_average = 0.7*frame_time_running_average + 0.3*frame_time
        

# Release resources
vs.release()
cv2.destroyAllWindows()
