import cv2
cv2.namedWindow('out', cv2.WINDOW_NORMAL)
img = cv2.imread('Data_Image.jpg')
pat = img[160:260, 1000:1155]
# color = [0,0,255]
img[836:936, 248:403] = pat
cv2.imshow('out', img)
cv2.waitKey(0)
cv2.destroyAllWindows()