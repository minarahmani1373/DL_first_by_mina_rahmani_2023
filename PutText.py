import cv2

cv2.namedWindow('out', cv2.WINDOW_NORMAL)
img = cv2.imread('Data_Image.jpg')
font = cv2.FONT_HERSHEY_SIMPLEX
img = cv2.putText(img, "Deep learning", (1000, 1000), font, 5, (0, 0, 255), 10)
cv2.imshow('out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
