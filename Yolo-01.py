import cv2
import numpy as np

cap = cv2.VideoCapture(2)
coco = "coco.names"
coco_class = []

with open(coco, "rt") as f:
    coco_class = f.read().rstrip('\n').split('\n')

print(coco_class)
print(len(coco_class))

while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('output', frame)

    if cv2.waitKey(2) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


