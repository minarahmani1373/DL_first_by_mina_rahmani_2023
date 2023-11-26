import cv2

cap = cv2.VideoCapture('vtest.avi')
fourcc = cv2.VideoWriter.fourcc('X', 'V', 'I', 'D')
out = cv2.VideoWriter('output.avi', fourcc, 30.0, (768, 576))

while (cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:

        # print(frame)
        print(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        print(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out.write(frame)
        cv2.imshow("OutputVideoX", frame)
        if cv2.waitKey(1) == ord('q'):
            break
    else:
        break

cap.release()
out.release()
cv2.destroyAllWindows()
