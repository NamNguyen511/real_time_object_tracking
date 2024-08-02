import cv2
import numpy as np

# Initialize the camera
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open the video camera from your laptop.")
    exit()

backSubMOG2 = cv2.createBackgroundSubtractorMOG2()
backSubKNN = cv2.createBackgroundSubtractorKNN()

while cap.isOpened():
    ret, frame = cap.read()
    if frame is None:
        break

    fgMaskMOG2 = backSubMOG2.apply(frame)
    fgMaskKNN = backSubKNN.apply(frame)

    # Find contours in the mask
    contoursMOG2, _ = cv2.findContours(fgMaskMOG2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw rectangle around detected contours
    for contourMOG2 in contoursMOG2:
        if cv2.contourArea(contourMOG2) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contourMOG2)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(frame, str(cap.get(cv2.CAP_PROP_POS_FRAMES)), (15,15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0))

    contoursKNN, _ = cv2.findContours(fgMaskKNN, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contourKNN in contoursKNN:
        if cv2.contourArea(contourKNN) < 500:
            continue
        x, y, w, h = cv2.boundingRect(contourKNN)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    cv2.imshow("Frame", frame)
    cv2.imshow("FG MASK MOG2", fgMaskMOG2)
    cv2.imshow("FG MASK KNN", fgMaskKNN)

    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()