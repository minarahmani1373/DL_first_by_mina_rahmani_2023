import cv2
import numpy as np

cap = cv2.VideoCapture(0)
coco = "coco.names"
coco_class = []
net_config = "yolov3.cfg"
net_weights = "yolov3.weights"
blob_size = 320

with open(coco, "rt") as f:
    coco_class = f.read().rstrip('\n').split('\n')

# print(coco_class)
# print(len(coco_class))

net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

while (cap.isOpened()):
    ret, frame = cap.read()
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255, size=(blob_size, blob_size), mean=(0,0,0), swapRB=True, crop=False)
    # for image in blob:
    #     for k, b in enumerate(image):
    #         cv2.imshow(str(k), b)
    net.setInput(blob)
    # print(net.getUnconnectedOutLayersNames())
    out_names = net.getUnconnectedOutLayersNames()
    output = net.forward(out_names)
    # print(len(output))
    # print(output)
    # print(type(output))
    print(output[0].shape)
    print(output[1].shape)
    print(output[2].shape)

    # print(output[2].shape)
    # cv2.imshow('output', frame)

    if cv2.waitKey(2) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


