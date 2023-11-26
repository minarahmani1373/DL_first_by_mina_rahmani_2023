import cv2
import numpy as np

events = [i for i in dir(cv2) if 'EVENT' in i]
print(events)

def ms_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        font = cv2.FONT_HERSHEY_SIMPLEX
        txt = '(' + str(x) + ', ' + str(y) + ')'
        cv2.putText(img, txt, (x, y), font, 0.5, (0, 100,200))
        cv2.circle(img, (x, y), 3, (0,0,255), -1)
        points.append((x, y))
        if len(points) >= 2:
            cv2.line(img, points[-1], points[-2], (0, 225, 255))
        cv2.imshow(out, img)

    if event == cv2.EVENT_RBUTTONDOWN:
        blue = img[y, x, 0]
        green = img[y, x, 1]
        red = img[y, x, 2]
        print(blue, ',', green, ',', red)
        color_selected = np.zeros((300, 300, 3), np.uint8)
        print(color_selected)
        color_selected[:] = [blue, green, red]
        cv2.imshow('Your Color', color_selected)


cv2.namedWindow('out', cv2.WINDOW_NORMAL)
points = []
img = cv2.imread('Data_Image.jpg')
out = 'Output'
cv2.imshow(out, img)
# cv2.setMouseCallback(out, ms_event)
cv2.waitKey(0)
cv2.destroyAllWindows()






