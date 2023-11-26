import cv2
import numpy as np

cap = cv2.VideoCapture(0)

while(cap.isOpened()):
    ret, frame = cap.read()

    if ret == True:
        frame = cv2.flip(frame, 1)
        cv2.putText(frame, 'Deep learning', (100, 100), 1, 3.0, (0,0,255), 5 )
        cv2.imshow('Output', frame)

        if cv2.waitKey(2) == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()