import cv2
cv2.namedWindow("output", cv2.WINDOW_NORMAL)

img = cv2.imread('Data_Image.jpg')
cv2.imshow('output', img)

key = cv2.waitKey(0)

if key == 27:
    cv2.destroyAllWindows()

elif key == ord('s'):
    cv2.imwrite("Data_copy.jpg", img)
    cv2.destroyAllWindows()




