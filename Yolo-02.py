import cv2
import numpy as np

cap = cv2.VideoCapture(2)
coco = "coco.names"
coco_class = []
net_config = "yolov3-tiny.cfg"
net_weights = "yolov3-tiny.weights"

with open(coco, "rt") as f:
    coco_class = f.read().rstrip('\n').split('\n')

print(coco_class)
print(len(coco_class))

net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while (cap.isOpened()):
    ret, frame = cap.read()
    cv2.imshow('output', frame)

    if cv2.waitKey(2) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


