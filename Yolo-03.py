import cv2
import numpy as np

cap = cv2.VideoCapture(0)
coco = "coco.names"
coco_class = []
net_config = "yolov3-tiny.cfg"
net_weights = "yolov3-tiny.weights"
blob_size = 320

with open(coco, "rt") as f:
    coco_class = f.read().rstrip('\n').split('\n')

print(coco_class)
print(len(coco_class))

net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while (cap.isOpened()):
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(blob_size, blob_size), mean=(0,0,0), swapRB=True, crop=False)
    for image in blob:
        for k, b in enumerate(image):
            cv2.imshow(str(k), b)

    net.setInput(blob)
    cv2.imshow('output', frame)

    if cv2.waitKey(2) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


