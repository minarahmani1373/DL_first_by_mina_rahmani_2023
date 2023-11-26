import cv2
import numpy as np

cap = cv2.VideoCapture(0)
coco = "coco.names"
coco_class = []
net_config = "yolov3.cfg"
net_weights = "yolov3.weights"
blob_size = 320
conf_threshold = 0.5
nms_threshold = 0.3

with open(coco, "rt") as f:
    coco_class = f.read().rstrip('\n').split('\n')

# print(coco_class)
# print(len(coco_class))
net = cv2.dnn.readNetFromDarknet(net_config, net_weights)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
######################################################
def findObject(output, image):
    image_h, image_w, image_c = image.shape
    bboxes = [] # good boxes
    class_ids = []
    confidences = []

    for member in output:
        for detect_vector in member:
            scores = detect_vector[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > conf_threshold:
                w, h = int(detect_vector[2] * image_w), int(detect_vector[3] * image_h)
                x, y = int((detect_vector[0]*image_w)-w/2), int((detect_vector[1]*image_h)-h/2)
                bboxes.append([x, y, w, h])
                class_ids.append(class_id)
                confidences.append(float(confidence))

    indices = cv2.dnn.NMSBoxes(bboxes, confidences, conf_threshold, nms_threshold)
    # print(indices)
    for i in indices:
        bbox = bboxes[i]
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
        cv2.putText(image, f'{coco_class[class_ids[i]].upper()} {int(confidences[i]*100)}%',
                    (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
###################################################################################################

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
    # print(output[2].shape)
    findObject(output, frame)
    cv2.imshow('output', frame)
    if cv2.waitKey(2) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


