import cv2
import datetime
count = 0

while True:
    vs = cv2.VideoCapture(0)
    ret, full_frame = vs.read()  # Capture a frame

    if not ret:
        print('no image; exiting')
        break  # Exit if the frame could not be captured
    
    count += 1

    if count%10 ==0:#if count is a multiple of 10
        print(f'{count=} at {datetime.datetime.now().strftime("%I:%M%p:%S on %B %d, %Y")}')

print(f'exiting at {datetime.datetime.now().strftime("%I:%M%p:%S on %B %d, %Y")} with {count=}')