import cv2

cv2.namedWindow('out', cv2.WINDOW_NORMAL)
img = cv2.imread('Data_Image.jpg')
img = cv2.line(img, (0, 0), (500, 300), (0, 0, 255), 9)
img = cv2.arrowedLine(img, (200, 200), (600, 959), (0, 255, 128), 9)
x1 = (int)(img.shape[1]/2)
y1 = (int)(img.shape[0]/2)
x2 = (int)(img.shape[1])
y2 = (int)(img.shape[0])
img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 229, 204))
img = cv2.circle(img, (2500, 2500), 500, (0, 0, 255), 10)

cv2.imshow('out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()