import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the video camera from your laptop.")
    exit()

# Read the first frame and detect motion
ret, frame1 = cap.read()
ret, frame2 = cap.read()

while cap.isOpened():
    # Calculate the abs difference between the two frames
    diff = cv2.absdiff(frame1, frame2)

    # Covert the difference to grayscale
    gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Blur grayscale image to reduce noise
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # Apply binary threshold to the blurred image
    _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)

    # Dilate the threshold image to fill in holes
    dilated = cv2.dilate(thresh, None, iterations=3)

    # Find contours in the dilated image
    contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangles around detected contours
    for contour in contours:
        if cv2.contourArea(contour) < 700:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(frame1, (x,y), (x + w, y + h), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow("Motion Detector", frame1)

    # Read the next frame
    frame1 = frame2
    ret, frame2 = cap.read()

    # Exit the loop, if "q" is pressed
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break


# Release the video capture object and close windows
cap.release()
cv2.destroyWindow()
