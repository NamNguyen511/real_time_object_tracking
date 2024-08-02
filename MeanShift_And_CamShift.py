import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the video camera from your laptop.")
    exit()

# Read the first frame
ret, frame = cap.read()
if not ret:
    print("Failed to read from camera")
    exit()

# Select the ROI (Region Of Interest) for tracking

x, y, w, h = 300, 200, 200, 200
track_window = (x, y, w, h)

# Set up the ROI for tracking
roi_frame = frame[y:y+h, x:x+w]
hsv_roi = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv_roi, np.array((0., 60., 32.)), np.array((180., 255., 255.)))
roi_hist = cv2.calcHist([hsv_roi], [0], mask, [180], [0,180])
cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)

# Set up the termination criteria: either 10 iterations or move by at least 1 pt
term_crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)

    # Apply MeanShift to get the new location
    ret, track_window = cv2.meanShift(dst, track_window, term_crit)
    if ret:
        x, y, w, h = track_window
        img2 = cv2.rectangle(frame, (x, y), (x + w, y + h), 255, 2)
        cv2.imshow("MeanShift Tracking", img2)
    else:
        print("MeanShift Tracking Failed ")

    # Apply CamShift to get the new location
    # ret1, track_window = cv2.CamShift(dst, track_window, term_crit)
    # pts = cv2.boxPoints(ret1)
    # pts = np.int0(pts)
    # img3 = cv2.polylines(frame, [pts], True, 255, 2)
    # cv2.imshow("CamShift Tracking", img3)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()