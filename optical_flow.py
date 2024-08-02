import cv2
import numpy as np

# Initialize laptop camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the camera from your laptop.")
    exit()

# Initialize the first frame and detect motion
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
hsv = np.zeros_like(frame1)
hsv[..., 1] = 255

# Compute and display optical flow using Farneback method
while cap.isOpened():
    ret, frame2 = cap.read()
    if not ret:
        break

    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # Calculate optical flow
    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # Display flow matrix in terminal
    print(flow)

    # Compute the magnitude and angle of 2D vectors
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # Overlay the flow vectors on frame
    frame2 = cv2.addWeighted(frame2, 1, bgr, 2, 0)
    cv2.imshow("Optical flow", frame2)

    prvs = next

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()