import cv2
import numpy as np
cv2.namedWindow('out', cv2.WINDOW_NORMAL)

img = np.zeros([2000, 2000, 3], dtype='uint8')
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, "Deep learning", (200, 200), font, 2, (0, 0, 255), 8)
img = cv2.line(img, (0, 0), (500, 300), (0, 0, 255), 9)
cv2.imshow('out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
